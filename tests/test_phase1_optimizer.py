"""Phase 1 tests – Config, Schemas, and Deterministic Schedule Optimizer."""

import pytest

from app.config import GEMINI_MODEL_NAME
from app.schema import (
    AdaptiveModelContext,
    FixedLecture,
    ScheduleRequest,
    StudyTask,
    OutboundSignals,
)
from app.engine.optimizer import (
    _hhmm_to_mins,
    _mins_to_hhmm,
    _Interval,
    _subtract_intervals,
    optimize_schedule,
)


# ===================================================================
# Helper factories
# ===================================================================

def _make_task(**overrides) -> StudyTask:
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


def _make_request(
    tasks=None,
    lectures=None,
    gcal_busy=None,
    adaptive_overrides=None,
    target_date="2026-11-02",
) -> ScheduleRequest:
    adaptive_kw = adaptive_overrides or {}
    return ScheduleRequest(
        target_date=target_date,
        tasks=tasks or [],
        university_schedule=lectures or [],
        gcal_busy_blocks=gcal_busy or [],
        adaptive_model=AdaptiveModelContext(**adaptive_kw),
    )


# ===================================================================
# A. Config tests
# ===================================================================

class TestConfig:
    def test_model_name_has_default(self):
        """GEMINI_MODEL_NAME should have a sensible default."""
        assert isinstance(GEMINI_MODEL_NAME, str)
        assert len(GEMINI_MODEL_NAME) > 0


# ===================================================================
# B. Schema validation tests
# ===================================================================

class TestSchemas:
    def test_study_task_defaults(self):
        t = _make_task()
        assert t.status == "pending"
        assert t.gcal_event_id is None

    def test_study_task_priority_bounds(self):
        with pytest.raises(Exception):
            _make_task(priority_weight=0.5)
        with pytest.raises(Exception):
            _make_task(priority_weight=5.5)

    def test_adaptive_model_defaults(self):
        ctx = AdaptiveModelContext()
        assert ctx.effective_focus_duration_mins == 25
        assert ctx.max_daily_study_mins == 240
        assert ctx.current_system_mode == "NORMAL"

    def test_outbound_signals_defaults(self):
        sig = OutboundSignals()
        assert sig.to_adaptive_manager is None
        assert sig.to_ai_tutor == []

    def test_fixed_lecture_creation(self):
        lec = FixedLecture(
            id="lec1", day="Monday", date="2026-11-02",
            start="09:00", end="10:30", title="Linear Algebra",
        )
        assert lec.start == "09:00"


# ===================================================================
# C. Internal helper tests
# ===================================================================

class TestHelpers:
    def test_hhmm_to_mins(self):
        assert _hhmm_to_mins("00:00") == 0
        assert _hhmm_to_mins("08:00") == 480
        assert _hhmm_to_mins("22:00") == 1320

    def test_mins_to_hhmm(self):
        assert _mins_to_hhmm(0) == "00:00"
        assert _mins_to_hhmm(480) == "08:00"
        assert _mins_to_hhmm(1320) == "22:00"

    def test_subtract_intervals_no_overlap(self):
        free = [_Interval(480, 1320)]
        busy = [_Interval(0, 60)]
        result = _subtract_intervals(free, busy)
        assert len(result) == 1
        assert result[0].start == 480

    def test_subtract_intervals_full_overlap(self):
        free = [_Interval(480, 600)]
        busy = [_Interval(480, 600)]
        result = _subtract_intervals(free, busy)
        assert len(result) == 0

    def test_subtract_intervals_split(self):
        free = [_Interval(480, 1320)]
        busy = [_Interval(600, 660)]  # 10:00-11:00
        result = _subtract_intervals(free, busy)
        assert len(result) == 2
        assert result[0] == _Interval(480, 600)
        assert result[1] == _Interval(660, 1320)


# ===================================================================
# D. Optimizer – Step 1: Constraint Ingestion
# ===================================================================

class TestConstraintIngestion:
    def test_empty_schedule_gives_full_day(self):
        req = _make_request(tasks=[_make_task(estimated_mins=25)])
        res = optimize_schedule(req)
        assert len(res.scheduled_blocks) == 1

    def test_lecture_blocks_subtracted(self):
        """A lecture from 08:00-10:00 should push the first block to >= 10:00."""
        lec = FixedLecture(
            id="l1", day="Monday", date="2026-11-02",
            start="08:00", end="10:00", title="Physics",
        )
        task = _make_task(estimated_mins=25)
        req = _make_request(tasks=[task], lectures=[lec])
        res = optimize_schedule(req)
        assert len(res.scheduled_blocks) == 1
        assert _hhmm_to_mins(res.scheduled_blocks[0].start) >= 600  # >= 10:00

    def test_gcal_busy_blocks_subtracted(self):
        gcal = [{"start": "08:00", "end": "12:00"}]
        task = _make_task(estimated_mins=25)
        req = _make_request(tasks=[task], gcal_busy=gcal)
        res = optimize_schedule(req)
        assert _hhmm_to_mins(res.scheduled_blocks[0].start) >= 720  # >= 12:00


# ===================================================================
# E. Optimizer – Step 2: Duration & Chunking
# ===================================================================

class TestDurationChunking:
    def test_task_split_into_focus_chunks(self):
        """A 50-min task with 25-min focus should yield 2 blocks."""
        task = _make_task(estimated_mins=50)
        req = _make_request(
            tasks=[task],
            adaptive_overrides={"effective_focus_duration_mins": 25},
        )
        res = optimize_schedule(req)
        blocks = [b for b in res.scheduled_blocks if b.task_id == "t1"]
        assert len(blocks) == 2
        assert blocks[0].part == 1
        assert blocks[1].part == 2

    def test_break_between_chunks(self):
        """10-minute break should separate consecutive chunks."""
        task = _make_task(estimated_mins=50)
        req = _make_request(
            tasks=[task],
            adaptive_overrides={"effective_focus_duration_mins": 25},
        )
        res = optimize_schedule(req)
        blocks = sorted(res.scheduled_blocks, key=lambda b: b.start)
        end_first = _hhmm_to_mins(blocks[0].end)
        start_second = _hhmm_to_mins(blocks[1].start)
        assert start_second - end_first >= 10

    def test_preemptive_restructure_inflates_duration(self):
        """In PREEMPTIVE_RESTRUCTURE mode, 50 mins * 1.2 = 60 mins → 3 chunks @ 25 min focus."""
        task = _make_task(estimated_mins=50)
        req = _make_request(
            tasks=[task],
            adaptive_overrides={
                "effective_focus_duration_mins": 25,
                "current_system_mode": "PREEMPTIVE_RESTRUCTURE",
                "task_delay_factor": 1.2,
            },
        )
        res = optimize_schedule(req)
        blocks = [b for b in res.scheduled_blocks if b.task_id == "t1"]
        # 60 / 25 = 2 full + 1 of 10 min = 3 chunks
        assert len(blocks) == 3


# ===================================================================
# F. Optimizer – Step 3: Prioritization
# ===================================================================

class TestPrioritization:
    def test_assessment_remediation_first(self):
        """Tasks from assessment_remediation source should be scheduled first."""
        t_user = _make_task(
            task_id="user1", source="user",
            priority_weight=5.0, estimated_mins=25,
        )
        t_remed = _make_task(
            task_id="remed1", source="assessment_remediation",
            priority_weight=1.0, estimated_mins=25,
        )
        req = _make_request(tasks=[t_user, t_remed])
        res = optimize_schedule(req)
        assert res.scheduled_blocks[0].task_id == "remed1"

    def test_earlier_deadline_scheduled_first(self):
        t_late = _make_task(
            task_id="late", deadline="2026-11-10T23:59:59",
            estimated_mins=25, priority_weight=3.0,
        )
        t_early = _make_task(
            task_id="early", deadline="2026-11-03T23:59:59",
            estimated_mins=25, priority_weight=3.0,
        )
        req = _make_request(tasks=[t_late, t_early])
        res = optimize_schedule(req)
        assert res.scheduled_blocks[0].task_id == "early"

    def test_higher_weight_wins_on_tie(self):
        t_low = _make_task(
            task_id="low", priority_weight=2.0,
            deadline="2026-11-03T23:59:59", estimated_mins=25,
        )
        t_high = _make_task(
            task_id="high", priority_weight=5.0,
            deadline="2026-11-03T23:59:59", estimated_mins=25,
        )
        req = _make_request(tasks=[t_low, t_high])
        res = optimize_schedule(req)
        assert res.scheduled_blocks[0].task_id == "high"


# ===================================================================
# G. Optimizer – Step 4: Cognitive Window Matching & Fatigue
# ===================================================================

class TestCognitiveWindowAndFatigue:
    def test_high_load_prefers_peak_window(self):
        """A high-cognitive-load task should land in the peak window."""
        task = _make_task(
            task_id="hard", cognitive_load="high", estimated_mins=25,
        )
        req = _make_request(
            tasks=[task],
            adaptive_overrides={"peak_productivity_window": "09:00-11:00"},
        )
        res = optimize_schedule(req)
        block = res.scheduled_blocks[0]
        start = _hhmm_to_mins(block.start)
        assert 540 <= start < 660  # within 09:00-11:00

    def test_low_load_prefers_secondary_window(self):
        """A low-cognitive-load task should prefer the secondary window."""
        task = _make_task(
            task_id="easy", cognitive_load="low", estimated_mins=25,
        )
        req = _make_request(
            tasks=[task],
            adaptive_overrides={
                "peak_productivity_window": "09:00-11:00",
                "secondary_productivity_window": "19:00-22:00",
            },
        )
        res = optimize_schedule(req)
        block = res.scheduled_blocks[0]
        start = _hhmm_to_mins(block.start)
        assert 1140 <= start < 1320  # within 19:00-22:00

    def test_max_daily_study_cap(self):
        """Tasks exceeding max_daily_study_mins should be unscheduled."""
        tasks = [
            _make_task(task_id=f"t{i}", estimated_mins=25)
            for i in range(20)  # 20 * 25 = 500 mins
        ]
        req = _make_request(
            tasks=tasks,
            adaptive_overrides={"max_daily_study_mins": 100},
        )
        res = optimize_schedule(req)
        total_scheduled = sum(
            _hhmm_to_mins(b.end) - _hhmm_to_mins(b.start)
            for b in res.scheduled_blocks
        )
        assert total_scheduled <= 100
        assert len(res.unscheduled_tasks) > 0


# ===================================================================
# H. Optimizer – Step 5: Feasibility Verification
# ===================================================================

class TestFeasibility:
    def test_feasible_when_all_fit(self):
        task = _make_task(estimated_mins=25)
        req = _make_request(tasks=[task])
        res = optimize_schedule(req)
        assert res.is_feasible is True
        assert res.feasibility_flags is None

    def test_infeasible_when_overloaded(self):
        tasks = [
            _make_task(task_id=f"t{i}", estimated_mins=25)
            for i in range(20)
        ]
        req = _make_request(
            tasks=tasks,
            adaptive_overrides={"max_daily_study_mins": 50},
        )
        res = optimize_schedule(req)
        assert res.is_feasible is False
        assert res.feasibility_flags is not None
        assert len(res.unscheduled_tasks) > 0

    def test_deadline_violation_flagged(self):
        task = _make_task(
            task_id="urgent", estimated_mins=25,
            deadline="2026-11-02T10:00:00",
        )
        # Fill the whole day so nothing fits
        req = _make_request(
            tasks=[task],
            adaptive_overrides={"max_daily_study_mins": 0},
        )
        res = optimize_schedule(req)
        assert res.is_feasible is False
        violations = res.feasibility_flags["deadline_violations"]
        assert any(v["task_id"] == "urgent" for v in violations)

    def test_non_pending_tasks_are_ignored(self):
        completed = _make_task(task_id="done", status="completed", estimated_mins=25)
        pending = _make_task(task_id="todo", status="pending", estimated_mins=25)
        req = _make_request(tasks=[completed, pending])
        res = optimize_schedule(req)
        ids = [b.task_id for b in res.scheduled_blocks]
        assert "done" not in ids
        assert "todo" in ids


# ===================================================================
# I. Integration-style test – realistic scenario
# ===================================================================

class TestIntegrationScenario:
    def test_full_day_scenario(self):
        """Simulate a realistic day with lectures, mixed tasks, and cognitive windows."""
        lectures = [
            FixedLecture(
                id="l1", day="Monday", date="2026-11-02",
                start="08:00", end="09:30", title="Calculus",
            ),
            FixedLecture(
                id="l2", day="Monday", date="2026-11-02",
                start="14:00", end="15:30", title="English",
            ),
        ]
        tasks = [
            _make_task(
                task_id="remed1",
                source="assessment_remediation",
                title="Fix probability gaps",
                estimated_mins=45,
                priority_weight=4.5,
                cognitive_load="high",
                deadline="2026-11-03T23:59:59",
            ),
            _make_task(
                task_id="study1",
                source="life_coach",
                title="Read history chapter",
                estimated_mins=30,
                priority_weight=2.0,
                cognitive_load="low",
            ),
            _make_task(
                task_id="hw1",
                source="user",
                title="CS homework",
                estimated_mins=40,
                priority_weight=3.5,
                cognitive_load="medium",
                deadline="2026-11-04T12:00:00",
            ),
        ]
        req = _make_request(
            tasks=tasks,
            lectures=lectures,
            adaptive_overrides={
                "effective_focus_duration_mins": 25,
                "peak_productivity_window": "09:30-12:00",
                "secondary_productivity_window": "19:00-22:00",
                "max_daily_study_mins": 240,
            },
        )
        res = optimize_schedule(req)

        # All tasks should be scheduled (total ~115 mins << 240 cap)
        assert res.is_feasible is True
        assert len(res.unscheduled_tasks) == 0

        # Remediation task should appear first (highest priority)
        assert res.scheduled_blocks[0].task_id == "remed1"

        # No block should overlap with lectures
        for block in res.scheduled_blocks:
            bs = _hhmm_to_mins(block.start)
            be = _hhmm_to_mins(block.end)
            for lec in lectures:
                ls = _hhmm_to_mins(lec.start)
                le = _hhmm_to_mins(lec.end)
                assert be <= ls or bs >= le, (
                    f"Block {block.start}-{block.end} overlaps lecture "
                    f"{lec.start}-{lec.end}"
                )

        # No block exceeds focus duration
        for block in res.scheduled_blocks:
            duration = _hhmm_to_mins(block.end) - _hhmm_to_mins(block.start)
            assert duration <= 25


class TestPinnedConstraintsAndBreaks:
    def test_pinned_task_placed_first_with_breaks_and_chronological_sort(self):
        """Pinned task holds exact requested time, leaves 10-min break on BOTH left and right, and result is sorted."""
        pinned = _make_task(
            task_id="pin1",
            title="Evening Pinned Session",
            estimated_mins=25,
            pinned_start="10:00",
            pinned_end="10:25",
            cognitive_load="low",
        )
        flex1 = _make_task(
            task_id="flex1",
            title="Morning Slot Before Pinned",
            estimated_mins=70,  # wait: high -> peak 09:00-11:00; with 25 focus, 09:00-09:25 and 09:35-09:50!
            priority_weight=4.0,
            cognitive_load="high",
        )
        req = _make_request(
            tasks=[flex1, pinned],
            adaptive_overrides={
                "effective_focus_duration_mins": 25,
                "peak_productivity_window": "09:00-11:00",
                "secondary_productivity_window": "19:00-22:00",
            },
        )
        res = optimize_schedule(req)
        assert res.is_feasible is True

        # Check pinned block exact coordinates
        pin_blocks = [b for b in res.scheduled_blocks if b.task_id == "pin1"]
        assert len(pin_blocks) == 1
        assert pin_blocks[0].start == "10:00"
        assert pin_blocks[0].end == "10:25"

        # Check chronological ascending order and >= 10 min breaks between ALL adjacent blocks
        starts = [_hhmm_to_mins(b.start) for b in res.scheduled_blocks]
        assert starts == sorted(starts)
        for prev, nxt in zip(res.scheduled_blocks, res.scheduled_blocks[1:]):
            assert _hhmm_to_mins(nxt.start) - _hhmm_to_mins(prev.end) >= 10

    def test_left_piece_respects_break_when_block_shifted_to_peak(self):
        """When a block shifts to peak start (e.g. 10:00), earlier block must stop <= 09:50."""
        # Peak starts at 09:15; free slot starts at 08:50. Flex low fits if it can squeeze before peak.
        # Put lecture 08:00-09:00; peak=09:30-11:00; high task takes 09:30-09:55 -> left piece is [09:00, 09:20] (20m)
        lec = FixedLecture(
            id="l1", day="Mon", date="2026-11-02",
            start="08:00", end="09:00", title="Morning Lecture",
        )
        high_task = _make_task(task_id="high1", cognitive_load="high", estimated_mins=25, priority_weight=5.0)
        low_25 = _make_task(task_id="low25", cognitive_load="low", estimated_mins=25, priority_weight=1.0)
        req = _make_request(
            tasks=[high_task, low_25],
            lectures=[lec],
            adaptive_overrides={
                "peak_productivity_window": "09:30-10:00",
                "secondary_productivity_window": "09:00-09:30",  # Forces low_25 to try 09:00-09:30
                "effective_focus_duration_mins": 25,
            },
        )
        res = optimize_schedule(req)
        b_low = next(b for b in res.scheduled_blocks if b.task_id == "low25")
        b_high = next(b for b in res.scheduled_blocks if b.task_id == "high1")
        assert b_high.start == "09:30"
        # low25 CANNOT be placed at 09:00-09:25 because that violates 10-min break before 09:30!
        assert _hhmm_to_mins(b_low.start) >= 605 or _hhmm_to_mins(b_low.end) <= 560

    def test_pinned_conflict_with_lecture_is_infeasible(self):
        lec = FixedLecture(
            id="l1", day="Mon", date="2026-11-02",
            start="09:00", end="10:30", title="Physics",
        )
        pinned = _make_task(
            task_id="pin_bad",
            pinned_start="09:30",
            pinned_end="10:00",
            estimated_mins=30,
        )
        req = _make_request(tasks=[pinned], lectures=[lec])
        res = optimize_schedule(req)
        assert res.is_feasible is False
        assert "pin_bad" in res.unscheduled_tasks
        assert "pin_bad" in res.feasibility_flags["pinned_conflicts"]

    def test_legacy_gcal_event_id_property_backward_compatible(self):
        t = _make_task(gcal_event_id="ev_single")
        assert t.gcal_event_ids == ["ev_single"]
        assert t.gcal_event_id == "ev_single"
        t2 = _make_task(gcal_event_ids=["ev_1", "ev_2"])
        assert t2.gcal_event_id == "ev_1"
        t2.gcal_event_id = "ev_replaced"
        assert t2.gcal_event_ids == ["ev_replaced"]


class TestP1ScheduledDeadlineVerification:
    """P1: Verify scheduled blocks against deadlines normalized to Africa/Cairo."""

    def test_scheduled_task_missing_local_deadline_sets_infeasible(self):
        # Task pinned from 18:00 to 19:00 but deadline is 17:00 same day
        t_late = _make_task(
            task_id="t_late",
            title="Late Task",
            estimated_mins=60,
            deadline="2026-11-02T17:00:00",
            pinned_start="18:00",
            pinned_end="19:00",
        )
        req = _make_request(
            tasks=[t_late],
            target_date="2026-11-02",
            adaptive_overrides={"effective_focus_duration_mins": 60},
        )
        res = optimize_schedule(req)
        assert len(res.scheduled_blocks) == 1
        assert res.is_feasible is False
        assert res.feasibility_flags is not None
        violations = res.feasibility_flags["deadline_violations"]
        assert len(violations) == 1
        assert violations[0]["task_id"] == "t_late"
        assert violations[0]["scheduled_end"] == "19:00"

    def test_utc_z_deadline_normalized_to_cairo_timezone(self):
        # 16:30 UTC == 18:30 Africa/Cairo (UTC+2 in November)
        # Task runs 18:00-19:00 Cairo time -> ends 30 mins after 18:30 -> late!
        t_utc_late = _make_task(
            task_id="t_utc_late",
            title="UTC Late Task",
            estimated_mins=60,
            deadline="2026-11-02T16:30:00Z",
            pinned_start="18:00",
            pinned_end="19:00",
        )
        res_late = optimize_schedule(
            _make_request(
                tasks=[t_utc_late],
                target_date="2026-11-02",
                adaptive_overrides={"effective_focus_duration_mins": 60},
            )
        )
        assert res_late.is_feasible is False
        assert res_late.feasibility_flags["deadline_violations"][0]["task_id"] == "t_utc_late"

        # 17:00 UTC == 19:00 Africa/Cairo -> ending at 19:00 meets deadline cleanly
        t_utc_ok = _make_task(
            task_id="t_utc_ok",
            title="UTC On-Time Task",
            estimated_mins=60,
            deadline="2026-11-02T17:00:00Z",
            pinned_start="18:00",
            pinned_end="19:00",
        )
        res_ok = optimize_schedule(
            _make_request(
                tasks=[t_utc_ok],
                target_date="2026-11-02",
                adaptive_overrides={"effective_focus_duration_mins": 60},
            )
        )
        assert res_ok.is_feasible is True
        assert res_ok.feasibility_flags is None

    def test_date_only_deadline_treated_as_end_of_day(self):
        # Date-only deadline 2026-11-02 means 23:59:59 Africa/Cairo
        t_same_day = _make_task(
            task_id="t_same_day",
            title="Same Day Task",
            estimated_mins=60,
            deadline="2026-11-02",
            pinned_start="21:00",
            pinned_end="22:00",
        )
        res_ok = optimize_schedule(
            _make_request(
                tasks=[t_same_day],
                target_date="2026-11-02",
                adaptive_overrides={"effective_focus_duration_mins": 60},
            )
        )
        assert res_ok.is_feasible is True

        # Scheduled on 2026-11-03 after a 2026-11-02 date-only deadline -> infeasible
        t_past_date = _make_task(
            task_id="t_past_date",
            title="Past Date Deadline Task",
            estimated_mins=60,
            deadline="2026-11-02",
        )
        res_fail = optimize_schedule(
            _make_request(
                tasks=[t_past_date],
                target_date="2026-11-03",
                adaptive_overrides={"effective_focus_duration_mins": 60},
            )
        )
        assert len(res_fail.scheduled_blocks) == 1
        assert res_fail.is_feasible is False
        assert res_fail.feasibility_flags["deadline_violations"][0]["task_id"] == "t_past_date"
