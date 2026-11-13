"""Phase 2 tests – Telemetry Engine & LLM Task Deconstructor."""

import pytest

from app.schema import (
    FixedLecture,
    OutboundSignals,
    ScheduledBlock,
    StudyTask,
)
from app.engine.telemetry import (
    compute_workload_density,
    compute_velocity_slip,
    generate_study_block_invitations,
    compute_outbound_signals,
)
from app.engine.deconstructor import (
    MicroTask,
    DeconstructionResult,
    _deterministic_split,
    deconstruct_task,
    deconstruct_tasks,
)


# ===================================================================
# Helpers
# ===================================================================

def _task(**overrides) -> StudyTask:
    defaults = dict(
        task_id="t1",
        title="Review Ch. 1",
        source="user",
        estimated_mins=50,
        priority_weight=3.0,
        cognitive_load="medium",
        status="pending",
    )
    defaults.update(overrides)
    return StudyTask(**defaults)


def _block(**overrides) -> ScheduledBlock:
    defaults = dict(
        task_id="t1",
        title="Review Ch. 1",
        date="2026-11-02",
        start="09:00",
        end="09:25",
    )
    defaults.update(overrides)
    return ScheduledBlock(**defaults)


# ===================================================================
# A. Workload Density Metrics
# ===================================================================

class TestWorkloadDensity:
    def test_density_no_lectures_no_gcal(self):
        """With no fixed blocks, density = study_mins / 840."""
        blocks = [_block(start="09:00", end="09:50")]  # 50 mins
        result = compute_workload_density(
            scheduled_blocks=blocks,
            university_schedule=[],
            gcal_busy_blocks=[],
            target_date="2026-11-02",
        )
        assert result["total_free_mins"] == 840
        assert result["scheduled_study_mins"] == 50
        assert result["workload_density"] == pytest.approx(50 / 840, abs=0.001)

    def test_density_with_lectures(self):
        """Lectures reduce total_free_mins."""
        lec = FixedLecture(
            id="l1", day="Mon", date="2026-11-02",
            start="08:00", end="10:00", title="Math",
        )
        blocks = [_block(start="10:00", end="10:25")]  # 25 mins
        result = compute_workload_density(
            scheduled_blocks=blocks,
            university_schedule=[lec],
            gcal_busy_blocks=[],
            target_date="2026-11-02",
        )
        assert result["lecture_mins"] == 120
        assert result["total_free_mins"] == 840 - 120
        assert result["scheduled_study_mins"] == 25

    def test_density_with_gcal(self):
        """Gcal busy blocks reduce total_free_mins."""
        gcal = [{"start": "12:00", "end": "13:00"}]
        blocks = [_block(start="09:00", end="09:25")]
        result = compute_workload_density(
            scheduled_blocks=blocks,
            university_schedule=[],
            gcal_busy_blocks=gcal,
            target_date="2026-11-02",
        )
        assert result["gcal_busy_mins"] == 60
        assert result["total_free_mins"] == 840 - 60

    def test_density_ignores_other_dates(self):
        """Blocks on a different date should not count."""
        blocks = [_block(start="09:00", end="09:25", date="2026-11-03")]
        result = compute_workload_density(
            scheduled_blocks=blocks,
            university_schedule=[],
            gcal_busy_blocks=[],
            target_date="2026-11-02",
        )
        assert result["scheduled_study_mins"] == 0
        assert result["workload_density"] == 0.0

    def test_density_returns_all_keys(self):
        result = compute_workload_density([], [], [], "2026-11-02")
        expected_keys = {
            "target_date", "total_operating_mins", "lecture_mins",
            "gcal_busy_mins", "total_free_mins", "scheduled_study_mins",
            "workload_density",
        }
        assert set(result.keys()) == expected_keys

    def test_density_filters_gcal_by_target_date(self):
        """GCal busy blocks explicitly on another date must not reduce free time."""
        gcal = [
            {"date": "2026-11-03", "start": "09:00", "end": "12:00"},
            {"start": "2026-11-04T10:00:00+02:00", "end": "2026-11-04T12:00:00+02:00"},
            {"start": "14:00", "end": "15:00"},  # implicitly target_date
        ]
        result = compute_workload_density([], [], gcal, "2026-11-02")
        assert result["gcal_busy_mins"] == 60
        assert result["total_free_mins"] == 840 - 60

    def test_density_merges_overlapping_lecture_and_gcal_intervals(self):
        """Overlapping lecture and GCal event must not be double-counted."""
        lec = FixedLecture(
            id="l1", day="Mon", date="2026-11-02",
            start="09:00", end="11:00", title="Algorithms",
        )
        # Same lecture synced in GCal 10:00-12:00 -> union is 09:00-12:00 (180 mins)
        gcal = [{"date": "2026-11-02", "start": "10:00", "end": "12:00"}]
        blocks = [_block(start="13:00", end="14:00")]
        result = compute_workload_density(
            scheduled_blocks=blocks,
            university_schedule=[lec],
            gcal_busy_blocks=gcal,
            target_date="2026-11-02",
        )
        assert result["lecture_mins"] == 120
        assert result["gcal_busy_mins"] == 120
        assert result["total_free_mins"] == 840 - 180
        assert result["workload_density"] == pytest.approx(60 / (840 - 180), abs=0.001)

    def test_density_clips_to_operating_window_and_cross_midnight(self):
        """Events extending outside 08:00-22:00 or across midnight clip accurately."""
        gcal = [
            {"date": "2026-11-02", "start": "06:00", "end": "09:00"},  # clips to 08:00-09:00 (60m)
            {
                "start": "2026-11-02T20:30:00+02:00",
                "end": "2026-11-03T01:00:00+02:00",  # clips to 20:30-22:00 (90m)
            },
        ]
        result = compute_workload_density([], [], gcal, "2026-11-02")
        assert result["gcal_busy_mins"] == 150
        assert result["total_free_mins"] == 840 - 150


# ===================================================================
# B. Velocity Slip Detection
# ===================================================================

class TestVelocitySlip:
    def test_no_tasks_no_slip(self):
        result = compute_velocity_slip([])
        assert result["slip_ratio"] == 0.0
        assert result["deviation_exceeded"] is False
        assert result["alert_payload"] is None

    def test_all_completed_no_slip(self):
        tasks = [
            _task(task_id="t1", status="completed"),
            _task(task_id="t2", status="completed"),
        ]
        result = compute_velocity_slip(tasks)
        assert result["completion_rate"] == 1.0
        assert result["slip_ratio"] == 0.0
        assert result["deviation_exceeded"] is False

    def test_some_missed_below_threshold(self):
        """1 missed out of 10 = 10% slip, below 20% threshold."""
        tasks = (
            [_task(task_id=f"c{i}", status="completed") for i in range(9)]
            + [_task(task_id="m1", status="missed")]
        )
        result = compute_velocity_slip(tasks)
        assert result["completion_rate"] == pytest.approx(0.9, abs=0.01)
        assert result["slip_ratio"] == pytest.approx(0.1, abs=0.01)
        assert result["deviation_exceeded"] is False
        assert result["alert_payload"] is None

    def test_many_missed_exceeds_threshold(self):
        """5 missed out of 10 = 50% slip, above 20% threshold."""
        tasks = (
            [_task(task_id=f"c{i}", status="completed") for i in range(5)]
            + [_task(task_id=f"m{i}", status="missed") for i in range(5)]
        )
        result = compute_velocity_slip(tasks)
        assert result["completion_rate"] == 0.5
        assert result["slip_ratio"] == 0.5
        assert result["deviation_exceeded"] is True
        assert result["alert_payload"] is not None
        assert result["alert_payload"]["signal_type"] == "VELOCITY_SLIP_ALERT"

    def test_exactly_at_threshold_not_exceeded(self):
        """Exactly 20% slip should NOT trigger (threshold is >20%)."""
        tasks = (
            [_task(task_id=f"c{i}", status="completed") for i in range(4)]
            + [_task(task_id="m1", status="missed")]
        )
        result = compute_velocity_slip(tasks)
        assert result["completion_rate"] == pytest.approx(0.8, abs=0.01)
        assert result["slip_ratio"] == pytest.approx(0.2, abs=0.01)
        assert result["deviation_exceeded"] is False

    def test_pending_tasks_ignored(self):
        """Pending tasks are not attempted – should not count."""
        tasks = [
            _task(task_id="c1", status="completed"),
            _task(task_id="p1", status="pending"),
            _task(task_id="p2", status="pending"),
        ]
        result = compute_velocity_slip(tasks)
        assert result["attempted_count"] == 1
        assert result["completed_count"] == 1
        assert result["slip_ratio"] == 0.0

    def test_future_scheduled_task_does_not_trigger_velocity_slip(self):
        """Freshly scheduled tasks whose end time has NOT passed yet must not count as slip."""
        from datetime import datetime
        from zoneinfo import ZoneInfo

        now_cairo = datetime(2026, 11, 13, 10, 0, tzinfo=ZoneInfo("Africa/Cairo"))
        tasks = [
            _task(
                task_id="future_today",
                status="scheduled",
                scheduled_date="2026-11-13",
                scheduled_end="14:00",
            ),
            _task(
                task_id="future_tomorrow",
                status="scheduled",
                scheduled_date="2026-11-14",
                scheduled_end="09:00",
            ),
        ]
        result = compute_velocity_slip(tasks, now=now_cairo)
        assert result["attempted_count"] == 0
        assert result["slip_ratio"] == 0.0
        assert result["deviation_exceeded"] is False
        assert result["alert_payload"] is None

    def test_overdue_scheduled_task_counts_in_velocity_slip(self):
        """Scheduled tasks whose end time <= now (or on a past date) DO count if not completed."""
        from datetime import datetime
        from zoneinfo import ZoneInfo

        now_cairo = datetime(2026, 11, 13, 15, 0, tzinfo=ZoneInfo("Africa/Cairo"))
        tasks = [
            _task(
                task_id="overdue_today",
                status="scheduled",
                scheduled_date="2026-11-13",
                scheduled_end="14:00",
            ),
            _task(
                task_id="overdue_yesterday",
                status="scheduled",
                scheduled_date="2026-11-12",
                scheduled_end="21:00",
            ),
        ]
        result = compute_velocity_slip(tasks, now=now_cairo)
        assert result["attempted_count"] == 2
        assert result["completed_count"] == 0
        assert result["slip_ratio"] == 1.0
        assert result["deviation_exceeded"] is True
        assert result["alert_payload"]["signal_type"] == "VELOCITY_SLIP_ALERT"


# ===================================================================
# C. Study Block Invitations
# ===================================================================

class TestStudyBlockInvitations:
    def test_generates_invitation_per_block(self):
        tasks = [
            _task(task_id="t1", title="Review Ch. 1", topic_ref="ch1"),
            _task(task_id="t2", title="Practice Problems", topic_ref="prob"),
        ]
        blocks = [
            _block(task_id="t1", start="09:00", end="09:25"),
            _block(task_id="t2", start="10:00", end="10:25"),
        ]
        invitations = generate_study_block_invitations(blocks, tasks)
        assert len(invitations) == 2
        assert invitations[0]["task_id"] == "t1"
        assert invitations[0]["topic_ref"] == "ch1"
        assert invitations[1]["task_id"] == "t2"

    def test_invitation_contains_required_fields(self):
        tasks = [_task(task_id="t1", topic_ref="algebra", source="life_coach",
                       cognitive_load="high", parent_milestone_id="m1")]
        blocks = [_block(task_id="t1", start="09:00", end="09:25", date="2026-11-02", part=2)]
        invitations = generate_study_block_invitations(blocks, tasks)
        inv = invitations[0]
        assert inv["block_date"] == "2026-11-02"
        assert inv["block_start"] == "09:00"
        assert inv["block_end"] == "09:25"
        assert inv["block_part"] == 2
        assert inv["task_title"] == "Review Ch. 1"
        assert inv["source"] == "life_coach"
        assert inv["cognitive_load"] == "high"
        assert inv["parent_milestone_id"] == "m1"

    def test_missing_task_skipped(self):
        """Blocks referencing unknown task_ids are silently skipped."""
        blocks = [_block(task_id="unknown")]
        invitations = generate_study_block_invitations(blocks, [])
        assert len(invitations) == 0


# ===================================================================
# D. Aggregate Outbound Signals
# ===================================================================

class TestOutboundSignals:
    def test_happy_path_no_alerts(self):
        tasks = [_task(status="completed")]
        blocks = [_block()]
        signals = compute_outbound_signals(
            scheduled_blocks=blocks,
            tasks=tasks,
            university_schedule=[],
            gcal_busy_blocks=[],
            target_date="2026-11-02",
        )
        assert isinstance(signals, OutboundSignals)
        assert signals.to_adaptive_manager is None
        assert signals.to_life_coach is None
        assert len(signals.to_ai_tutor) == 1
        assert "workload_density" in signals.workload_density_metrics

    def test_feasibility_flags_propagated(self):
        flags = {"is_feasible": False, "unscheduled_task_ids": ["t1"]}
        signals = compute_outbound_signals(
            scheduled_blocks=[],
            tasks=[],
            university_schedule=[],
            gcal_busy_blocks=[],
            target_date="2026-11-02",
            feasibility_flags=flags,
        )
        assert signals.to_adaptive_manager == flags
        assert signals.to_life_coach is not None
        assert signals.to_life_coach["signal_type"] == "MILESTONE_FEASIBILITY_REPORT"

    def test_velocity_slip_alert_when_no_feasibility_flags(self):
        tasks = (
            [_task(task_id=f"c{i}", status="completed") for i in range(3)]
            + [_task(task_id=f"m{i}", status="missed") for i in range(7)]
        )
        signals = compute_outbound_signals(
            scheduled_blocks=[],
            tasks=tasks,
            university_schedule=[],
            gcal_busy_blocks=[],
            target_date="2026-11-02",
        )
        assert signals.to_adaptive_manager is not None
        assert signals.to_adaptive_manager["signal_type"] == "VELOCITY_SLIP_ALERT"
        # No feasibility issue → no life coach signal
        assert signals.to_life_coach is None

    def test_simultaneous_feasibility_and_velocity_slip_not_swallowed(self):
        """P1: Both feasibility issues and velocity slip must surface together."""
        tasks = (
            [_task(task_id=f"c{i}", status="completed") for i in range(2)]
            + [_task(task_id=f"m{i}", status="missed") for i in range(8)]
        )
        signals = compute_outbound_signals(
            scheduled_blocks=[],
            tasks=tasks,
            university_schedule=[],
            gcal_busy_blocks=[],
            target_date="2026-11-02",
            feasibility_flags={
                "signal_type": "FEASIBILITY_INFEASIBLE",
                "flags": ["DEADLINE_VIOLATION: task m0"],
            },
        )
        assert signals.to_life_coach is not None
        assert signals.to_life_coach["signal_type"] == "MILESTONE_FEASIBILITY_REPORT"
        assert signals.to_adaptive_manager is not None
        assert signals.to_adaptive_manager["signal_type"] == "FEASIBILITY_INFEASIBLE"
        assert "velocity_slip_alert" in signals.to_adaptive_manager
        assert signals.to_adaptive_manager["velocity_slip_alert"]["signal_type"] == "VELOCITY_SLIP_ALERT"
        assert signals.to_adaptive_manager["active_signals"] == [
            "FEASIBILITY_INFEASIBLE",
            "VELOCITY_SLIP_ALERT",
        ]


# ===================================================================
# E. Deterministic Fallback Splitter
# ===================================================================

class TestDeterministicSplit:
    def test_single_chunk_when_fits(self):
        """Task of 20 mins with 25 min focus → 1 chunk."""
        task = _task(estimated_mins=20)
        result = _deterministic_split(task, focus_duration_mins=25)
        assert len(result.micro_tasks) == 1
        assert result.micro_tasks[0].estimated_mins == 20

    def test_even_split(self):
        """100 mins / 25 focus → 4 chunks of 25."""
        task = _task(estimated_mins=100)
        result = _deterministic_split(task, focus_duration_mins=25)
        assert len(result.micro_tasks) == 4
        for mt in result.micro_tasks:
            assert mt.estimated_mins == 25

    def test_uneven_split_preserves_total(self):
        """53 mins / 25 focus → 3 chunks, total still 53."""
        task = _task(estimated_mins=53)
        result = _deterministic_split(task, focus_duration_mins=25)
        assert len(result.micro_tasks) == 3
        total = sum(mt.estimated_mins for mt in result.micro_tasks)
        assert total == 53

    def test_preserves_cognitive_load_and_topic(self):
        task = _task(cognitive_load="high", topic_ref="linear-algebra")
        result = _deterministic_split(task, focus_duration_mins=25)
        for mt in result.micro_tasks:
            assert mt.cognitive_load == "high"
            assert mt.topic_ref == "linear-algebra"

    def test_parent_fields_set(self):
        task = _task(task_id="parent1", title="Big Project")
        result = _deterministic_split(task, focus_duration_mins=25)
        assert result.parent_task_id == "parent1"
        assert result.parent_title == "Big Project"

    def test_titles_include_part_numbers(self):
        task = _task(estimated_mins=75, title="Study OS")
        result = _deterministic_split(task, focus_duration_mins=25)
        assert "Part 1/3" in result.micro_tasks[0].title
        assert "Part 3/3" in result.micro_tasks[2].title


# ===================================================================
# F. deconstruct_task public API (fallback path)
# ===================================================================

class TestDeconstructTask:
    def test_uses_fallback_when_no_api_key(self, monkeypatch):
        """Without GEMINI_API_KEY, should use deterministic fallback."""
        monkeypatch.setattr("app.config.GEMINI_API_KEY", "")
        task = _task(estimated_mins=60)
        result = deconstruct_task(task, focus_duration_mins=25)
        assert isinstance(result, DeconstructionResult)
        assert len(result.micro_tasks) >= 1
        total = sum(mt.estimated_mins for mt in result.micro_tasks)
        assert total == 60

    def test_force_fallback_flag(self, monkeypatch):
        """force_fallback=True should bypass LLM even with a key."""
        monkeypatch.setattr("app.config.GEMINI_API_KEY", "fake-key")
        task = _task(estimated_mins=50)
        result = deconstruct_task(task, focus_duration_mins=25, force_fallback=True)
        assert isinstance(result, DeconstructionResult)
        assert len(result.micro_tasks) == 2

    def test_reads_model_name_from_config(self, monkeypatch):
        """GEMINI_MODEL_NAME should be read from app.config (not hardcoded)."""
        from app import config
        assert hasattr(config, "GEMINI_MODEL_NAME")
        assert isinstance(config.GEMINI_MODEL_NAME, str)
        assert len(config.GEMINI_MODEL_NAME) > 0

    def test_llm_deconstruct_primary_model_succeeds(self, monkeypatch):
        """P2: Primary Gemini model succeeds and sets parent metadata cleanly."""
        import sys
        import types

        invoked_models = []

        class FakeStructured:
            def invoke(self, messages):
                return DeconstructionResult(
                    parent_task_id="ignored",
                    parent_title="ignored",
                    micro_tasks=[
                        MicroTask(title="Step A", estimated_mins=30, cognitive_load="high", topic_ref="dl_1"),
                        MicroTask(title="Step B", estimated_mins=30, cognitive_load="medium", topic_ref="dl_2"),
                    ],
                )

        class FakeLLM:
            def __init__(self, model, **kwargs):
                invoked_models.append(model)

            def with_structured_output(self, schema):
                return FakeStructured()

        fake_mod = types.ModuleType("langchain_google_genai")
        fake_mod.ChatGoogleGenerativeAI = FakeLLM
        monkeypatch.setitem(sys.modules, "langchain_google_genai", fake_mod)
        monkeypatch.setattr("app.config.GEMINI_API_KEY", "test-fake-key")
        monkeypatch.setattr("app.config.GEMINI_MODEL_NAME", "primary-gemini")
        monkeypatch.setattr("app.config.GEMINI_FALLBACK_MODEL_NAME", "fallback-gemini")

        task = _task(task_id="big_dl", title="Deep Learning Project", estimated_mins=60)
        res = deconstruct_task(task, focus_duration_mins=30)
        assert invoked_models == ["primary-gemini"]
        assert res.parent_task_id == "big_dl"
        assert res.parent_title == "Deep Learning Project"
        assert [mt.title for mt in res.micro_tasks] == ["Step A", "Step B"]

    def test_llm_deconstruct_fallback_model_used_when_primary_invoke_fails(self, monkeypatch):
        """P2: When primary model invoke fails, fallback model in candidate_models succeeds."""
        import sys
        import types

        invoked_models = []

        class FakeStructured:
            def __init__(self, model_name):
                self._model = model_name

            def invoke(self, messages):
                if self._model == "primary-gemini":
                    raise RuntimeError("404 primary model unavailable")
                return DeconstructionResult(
                    parent_task_id="big_fb",
                    parent_title="Fallback Topic",
                    micro_tasks=[
                        MicroTask(title="Part X", estimated_mins=25, cognitive_load="high"),
                        MicroTask(title="Part Y", estimated_mins=25, cognitive_load="medium"),
                    ],
                )

        class FakeLLM:
            def __init__(self, model, **kwargs):
                self._model = model
                invoked_models.append(model)

            def with_structured_output(self, schema):
                return FakeStructured(self._model)

        fake_mod = types.ModuleType("langchain_google_genai")
        fake_mod.ChatGoogleGenerativeAI = FakeLLM
        monkeypatch.setitem(sys.modules, "langchain_google_genai", fake_mod)
        monkeypatch.setattr("app.config.GEMINI_API_KEY", "test-fake-key")
        monkeypatch.setattr("app.config.GEMINI_MODEL_NAME", "primary-gemini")
        monkeypatch.setattr("app.config.GEMINI_FALLBACK_MODEL_NAME", "fallback-gemini")

        task = _task(task_id="big_fb", title="Fallback Topic", estimated_mins=50)
        res = deconstruct_task(task, focus_duration_mins=25)
        assert invoked_models == ["primary-gemini", "fallback-gemini"]
        assert [mt.title for mt in res.micro_tasks] == ["Part X", "Part Y"]

    def test_llm_deconstruct_falls_back_to_deterministic_split_when_all_models_fail(self, monkeypatch):
        """P2: If all candidate models fail on invoke, safely degrade to _deterministic_split."""
        import sys
        import types

        invoked_models = []

        class FakeStructured:
            def invoke(self, messages):
                raise RuntimeError("API quota exhausted")

        class FakeLLM:
            def __init__(self, model, **kwargs):
                invoked_models.append(model)

            def with_structured_output(self, schema):
                return FakeStructured()

        fake_mod = types.ModuleType("langchain_google_genai")
        fake_mod.ChatGoogleGenerativeAI = FakeLLM
        monkeypatch.setitem(sys.modules, "langchain_google_genai", fake_mod)
        monkeypatch.setattr("app.config.GEMINI_API_KEY", "test-fake-key")
        monkeypatch.setattr("app.config.GEMINI_MODEL_NAME", "primary-gemini")
        monkeypatch.setattr("app.config.GEMINI_FALLBACK_MODEL_NAME", "fallback-gemini")

        task = _task(task_id="big_det", title="Algorithms Chapter", estimated_mins=50)
        res = deconstruct_task(task, focus_duration_mins=25)
        assert invoked_models == ["primary-gemini", "fallback-gemini"]
        assert len(res.micro_tasks) == 2
        assert sum(mt.estimated_mins for mt in res.micro_tasks) == 50
        assert "Part 1/2" in res.micro_tasks[0].title

    def test_unmocked_live_llm_suppressed_during_pytest_even_with_api_key(self, monkeypatch):
        """P2: Under PYTEST_CURRENT_TEST with an unmocked client, no network call runs."""
        monkeypatch.setattr("app.config.GEMINI_API_KEY", "sk-real-looking-key")
        monkeypatch.setenv("PYTEST_CURRENT_TEST", "tests/test_phase2.py::test_suppression")
        task = _task(task_id="offline_guard", title="Linear Algebra", estimated_mins=50)
        res = deconstruct_task(task, focus_duration_mins=25)
        assert len(res.micro_tasks) == 2
        assert sum(mt.estimated_mins for mt in res.micro_tasks) == 50


# ===================================================================
# G. Batch deconstruction
# ===================================================================

class TestDeconstructTasks:
    def test_batch_returns_one_result_per_task(self, monkeypatch):
        monkeypatch.setattr("app.config.GEMINI_API_KEY", "")
        tasks = [
            _task(task_id="t1", estimated_mins=50),
            _task(task_id="t2", estimated_mins=75),
            _task(task_id="t3", estimated_mins=20),
        ]
        results = deconstruct_tasks(tasks, focus_duration_mins=25)
        assert len(results) == 3
        assert results[0].parent_task_id == "t1"
        assert results[1].parent_task_id == "t2"
        assert results[2].parent_task_id == "t3"

    def test_batch_total_minutes_preserved(self, monkeypatch):
        monkeypatch.setattr("app.config.GEMINI_API_KEY", "")
        tasks = [
            _task(task_id="t1", estimated_mins=53),
            _task(task_id="t2", estimated_mins=110),
        ]
        results = deconstruct_tasks(tasks, focus_duration_mins=25)
        for task, result in zip(tasks, results):
            total = sum(mt.estimated_mins for mt in result.micro_tasks)
            assert total == task.estimated_mins


# ===================================================================
# H. DeconstructionResult / MicroTask schema tests
# ===================================================================

class TestSchemas:
    def test_micro_task_defaults(self):
        mt = MicroTask(title="Part 1", estimated_mins=25)
        assert mt.cognitive_load == "medium"
        assert mt.topic_ref is None

    def test_deconstruction_result_empty(self):
        dr = DeconstructionResult(parent_task_id="p1", parent_title="T")
        assert dr.micro_tasks == []
