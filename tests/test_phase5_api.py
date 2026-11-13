"""Phase 5 tests – FastAPI API Server and Endpoints.

Tests cover:
  A. Health check endpoint function
  B. Schedule optimization endpoint (POST /api/v1/planning/optimize)
  C. Telemetry endpoint (POST /api/v1/planning/telemetry)
  D. Chat endpoint (POST /api/v1/planning/chat) with control signal
  E. FastAPI app configuration & route registration
"""

from app.api.routes import (
    ChatRequest,
    ChatResponse,
    OptimizeResponse,
    TelemetryRequest,
    chat_endpoint,
    optimize_schedule_endpoint,
    router,
    telemetry_endpoint,
)
from app.schema import (
    AdaptiveModelContext,
    FixedLecture,
    OutboundSignals,
    ScheduleRequest,
    ScheduledBlock,
    StudyTask,
)
from main import app, health_check


class TestHealthCheck:
    def test_health_check_returns_online(self):
        data = health_check()
        assert data["status"] == "online"
        assert "Personal Planning Agent" in data["agent"]


class TestAppRegistration:
    def test_routes_registered_on_app(self):
        router_paths = [r.path for r in router.routes]
        assert "/api/v1/planning/optimize" in router_paths
        assert "/api/v1/planning/chat" in router_paths
        assert "/api/v1/planning/telemetry" in router_paths


class TestOptimizeEndpoint:
    def test_optimize_schedule_success(self):
        req = ScheduleRequest(
            target_date="2026-11-02",
            tasks=[
                StudyTask(
                    task_id="api_t1",
                    title="API Testing Study",
                    source="user",
                    estimated_mins=25,
                    priority_weight=3.0,
                    cognitive_load="medium",
                    status="pending",
                )
            ],
            university_schedule=[
                FixedLecture(
                    id="lec_1",
                    day="Monday",
                    date="2026-11-02",
                    start="09:00",
                    end="10:30",
                    title="Software Architecture",
                )
            ],
            gcal_busy_blocks=[],
            adaptive_model=AdaptiveModelContext(
                peak_productivity_window="10:30-12:00",
                secondary_productivity_window="18:00-21:00",
                effective_focus_duration_mins=25,
                max_daily_study_mins=240,
                current_system_mode="NORMAL",
            ),
        )

        resp = optimize_schedule_endpoint(req)
        assert isinstance(resp, OptimizeResponse)
        assert resp.is_feasible is True
        assert len(resp.scheduled_blocks) == 1
        assert len(resp.gcal_events) == 1
        assert resp.outbound_signals is not None
        assert "workload_density" in resp.outbound_signals.workload_density_metrics

    def test_optimize_schedule_with_nested_study_data(self):
        """Verify ScheduleRequest accepts nested study_data {tasks, university_schedule} per spec."""
        payload = {
            "target_date": "2026-11-02",
            "study_data": {
                "tasks": [
                    {
                        "task_id": "nested_t1",
                        "title": "Nested Study Task",
                        "source": "life_coach",
                        "estimated_mins": 25,
                        "priority_weight": 3.0,
                        "cognitive_load": "medium",
                        "status": "pending",
                    }
                ],
                "university_schedule": [
                    {
                        "id": "lec_nested",
                        "day": "Monday",
                        "date": "2026-11-02",
                        "start": "09:00",
                        "end": "10:00",
                        "title": "Machine Learning",
                    }
                ],
            },
            "gcal_busy_blocks": [],
        }
        req = ScheduleRequest.model_validate(payload)
        assert len(req.tasks) == 1
        assert req.tasks[0].task_id == "nested_t1"
        assert len(req.university_schedule) == 1
        assert req.university_schedule[0].title == "Machine Learning"

        resp = optimize_schedule_endpoint(req)
        assert isinstance(resp, OptimizeResponse)
        assert resp.is_feasible is True
        assert len(resp.scheduled_blocks) == 1


class TestTelemetryEndpoint:
    def test_telemetry_computation(self):
        req = TelemetryRequest(
            target_date="2026-11-02",
            scheduled_blocks=[
                ScheduledBlock(
                    task_id="t1",
                    title="Study Block 1",
                    date="2026-11-02",
                    start="11:00",
                    end="11:25",
                    part=1,
                )
            ],
            tasks=[
                StudyTask(
                    task_id="t1",
                    title="Study Block 1",
                    source="user",
                    estimated_mins=25,
                    priority_weight=3.0,
                    cognitive_load="medium",
                    status="completed",
                )
            ],
            university_schedule=[],
            gcal_busy_blocks=[],
        )

        resp = telemetry_endpoint(req)
        assert isinstance(resp, OutboundSignals)
        assert "workload_density" in resp.workload_density_metrics
        assert len(resp.to_ai_tutor) == 1


class TestChatEndpoint:
    def test_control_signal_chat(self):
        req = ChatRequest(
            message='{"signal_type": "PREEMPTIVE_SCHEDULE_RESTRUCTURE", "adaptive_model_updates": {"task_delay_factor": 1.2}}',
            entry_mode="control_signal",
            target_date="2026-11-02",
            tasks=[],
            university_schedule=[],
            gcal_busy_blocks=[],
        )

        resp = chat_endpoint(req)
        assert isinstance(resp, ChatResponse)
        assert resp.entry_mode == "control_signal"
        assert "PREEMPTIVE_SCHEDULE_RESTRUCTURE" in resp.response

    def test_chat_endpoint_response_time_under_5_seconds(self):
        import time

        req = ChatRequest(
            message="I want to study linear algebra chapter 3 for 50 minutes tomorrow.",
            entry_mode="chat",
            target_date="2026-11-02",
            tasks=[],
            university_schedule=[],
            gcal_busy_blocks=[],
        )

        start = time.time()
        resp = chat_endpoint(req)
        elapsed = time.time() - start

        assert isinstance(resp, ChatResponse)
        assert resp.response is not None
        assert elapsed < 5.0, f"Chat endpoint took {elapsed:.2f}s, expected < 5.0s"

    def test_chat_natural_language_task_creation_and_optimization(self):
        """Verify chat mode parses natural language request, creates/deconstructs tasks,
        optimizes the schedule, and syncs calendar events without returning empty or null."""
        req = ChatRequest(
            message="Add a high priority study task for Deep Learning review tomorrow for 45 minutes and optimize my schedule",
            entry_mode="chat",
            target_date="2026-11-02",
            tasks=[],
            university_schedule=[],
            gcal_busy_blocks=[],
        )

        resp = chat_endpoint(req)
        assert isinstance(resp, ChatResponse)
        assert len(resp.tasks) >= 1
        assert resp.schedule_result is not None
        assert len(resp.schedule_result.scheduled_blocks) >= 1
        assert len(resp.gcal_events) >= 1
        assert resp.outbound_signals is not None
        # Tasks should have been deconstructed because 45 mins > default 25 min focus limit
        assert any("Deep Learning" in t.title for t in resp.tasks)

    def test_chat_swagger_reschedule_example_no_false_velocity_slip(self, monkeypatch):
        """Swagger UI default example reschedules task_algo to 19:30 with no ToolMessage JSON leak."""
        monkeypatch.setattr("app.config.GEMINI_API_KEY", "")
        example = dict(ChatRequest.model_config["json_schema_extra"]["example"])
        example["target_date"] = "2099-11-02"  # future date relative to now
        req = ChatRequest.model_validate(example)
        resp = chat_endpoint(req)

        assert resp.schedule_result is not None
        assert len(resp.schedule_result.scheduled_blocks) == 1
        assert resp.schedule_result.scheduled_blocks[0].start == "19:30"
        assert resp.schedule_result.scheduled_blocks[0].end == "19:55"
        assert '"action": "reschedule_block"' not in resp.response
        assert (
            resp.outbound_signals.to_adaptive_manager is None
            or resp.outbound_signals.to_adaptive_manager.get("signal_type") != "VELOCITY_SLIP_ALERT"
        )


class TestP1UnifiedPipeline:
    """P1: Verify unified execution pipeline shared by /optimize and LangGraph nodes."""

    def test_pipeline_deletes_stale_events_before_busy_query_and_merges_gcal(self, tmp_path):
        from app.engine.pipeline import execute_planning_pipeline
        from app.integrations.gcal_service import GoogleCalendarService

        gcal = GoogleCalendarService(credentials_path=str(tmp_path / "no_creds.json"))
        # External calendar meeting + stale study block owned by task "t_resched"
        gcal.create_calendar_event("Doctor Appointment", "2026-11-02", "09:00", "10:00")
        stale_evt = gcal.create_calendar_event("Old Study Chunk", "2026-11-02", "10:00", "11:00")
        stale_id = stale_evt["id"]

        task = StudyTask(
            task_id="t_resched",
            title="New Session",
            source="user",
            estimated_mins=30,
            priority_weight=4.0,
            cognitive_load="high",
            status="pending",
            gcal_event_ids=[stale_id],
        )

        res = execute_planning_pipeline(
            target_date="2026-11-02",
            tasks=[task],
            university_schedule=[],
            gcal_busy_blocks=[],
            gcal_service=gcal,
            adaptive_model=AdaptiveModelContext(effective_focus_duration_mins=30),
        )

        # Stale event was deleted BEFORE busy lookup, so slot 10:00 became free!
        assert stale_id not in gcal._mock_events
        assert res.schedule_result.scheduled_blocks[0].start == "10:00"
        assert res.schedule_result.scheduled_blocks[0].end == "10:30"
        # External meeting (09:00-10:00) remained in merged_gcal_busy_blocks & telemetry density
        assert any(
            b["start"] == "09:00" and b["end"] == "10:00"
            for b in res.merged_gcal_busy_blocks
        )
        assert res.outbound_signals.workload_density_metrics["gcal_busy_mins"] == 60
        # Task got replaced gcal_event_ids
        assert res.updated_tasks[0].gcal_event_ids != [stale_id]
        assert len(res.updated_tasks[0].gcal_event_ids) == 1
        assert res.updated_tasks[0].status == "scheduled"

    def test_optimize_endpoint_and_graph_nodes_produce_consistent_output(self, tmp_path, monkeypatch):
        from app.agent.graph import PlanningState, optimizer_and_gcal_node, telemetry_and_signals_node
        from app.integrations.gcal_service import GoogleCalendarService
        import app.engine.pipeline as pipeline_mod

        fake_creds = str(tmp_path / "missing_creds.json")
        monkeypatch.setattr(
            pipeline_mod,
            "GoogleCalendarService",
            lambda: GoogleCalendarService(credentials_path=fake_creds),
        )

        def _make_inputs():
            return (
                "2026-11-02",
                [
                    StudyTask(
                        task_id="sync_t1",
                        title="Consistent Task",
                        source="user",
                        estimated_mins=25,
                        priority_weight=4.0,
                        cognitive_load="high",
                        status="pending",
                    )
                ],
                [
                    FixedLecture(
                        id="lec_sync",
                        day="Monday",
                        date="2026-11-02",
                        start="08:00",
                        end="09:00",
                        title="OS Lecture",
                    )
                ],
                [{"start": "12:00", "end": "13:00"}],
            )

        t_date, tasks1, lecs1, busy1 = _make_inputs()
        api_resp = optimize_schedule_endpoint(
            ScheduleRequest(
                target_date=t_date,
                tasks=tasks1,
                university_schedule=lecs1,
                gcal_busy_blocks=busy1,
            )
        )

        t_date2, tasks2, lecs2, busy2 = _make_inputs()
        state = PlanningState(
            target_date=t_date2,
            tasks=[t.model_dump() for t in tasks2],
            university_schedule=[l.model_dump() for l in lecs2],
            gcal_busy_blocks=busy2,
            gcal_events_created=[],
            adaptive_model=AdaptiveModelContext().model_dump(),
            entry_mode="chat",
            state_dirty=True,
        )
        after_opt = optimizer_and_gcal_node(state)
        state = state.model_copy(update=after_opt)
        after_tel = telemetry_and_signals_node(state)

        assert api_resp.is_feasible == after_opt["schedule_result"]["is_feasible"]
        assert len(api_resp.scheduled_blocks) == len(after_opt["schedule_result"]["scheduled_blocks"]) == 1
        assert api_resp.scheduled_blocks[0].start == after_opt["schedule_result"]["scheduled_blocks"][0]["start"]
        assert api_resp.scheduled_blocks[0].end == after_opt["schedule_result"]["scheduled_blocks"][0]["end"]
        assert (
            api_resp.outbound_signals.workload_density_metrics
            == after_tel["outbound_signals"]["workload_density_metrics"]
        )
        assert api_resp.tasks is not None
        assert api_resp.tasks[0].status == "scheduled"
        assert len(api_resp.tasks[0].gcal_event_ids) == 1


class TestP2HttpAndSecurity:
    """P2: Real HTTP tests via FastAPI TestClient, dynamic Cairo dates, validation & CORS."""

    def test_http_optimize_endpoint_and_dynamic_default_date(self, monkeypatch):
        from fastapi.testclient import TestClient
        from app.config import get_today_date_str

        monkeypatch.setattr("app.config.GEMINI_API_KEY", "fake-live-looking-key")
        client = TestClient(app)

        # Verify dynamic default target_date uses Africa/Cairo today string (not 2026-11-02)
        default_req = ScheduleRequest()
        assert default_req.target_date == get_today_date_str()
        assert default_req.target_date != "2026-11-02"

        payload = {
            "target_date": "2026-11-02",
            "tasks": [
                {
                    "task_id": "http_opt_1",
                    "title": "Compiler Design Study",
                    "source": "user",
                    "estimated_mins": 25,
                    "priority_weight": 4.0,
                    "cognitive_load": "high",
                    "status": "pending",
                }
            ],
            "university_schedule": [],
            "gcal_busy_blocks": [],
        }
        resp = client.post("/api/v1/planning/optimize", json=payload)
        assert resp.status_code == 200
        body = resp.json()
        assert body["is_feasible"] is True
        assert len(body["scheduled_blocks"]) == 1
        assert body["tasks"][0]["status"] == "scheduled"
        assert body["outbound_signals"] is not None

    def test_http_chat_new_task_and_reschedule_without_network(self, monkeypatch):
        from fastapi.testclient import TestClient

        # Even when GEMINI_API_KEY looks populated, PYTEST_CURRENT_TEST suppresses real network calls
        monkeypatch.setattr("app.config.GEMINI_API_KEY", "sk-non-empty-test-key")
        client = TestClient(app)

        # 1. New study task scheduling over HTTP
        create_resp = client.post(
            "/api/v1/planning/chat",
            json={
                "message": "Schedule a 25 minute study task for Operating Systems",
                "target_date": "2099-11-02",
                "tasks": [],
                "entry_mode": "chat",
            },
        )
        assert create_resp.status_code == 200
        data_create = create_resp.json()
        assert data_create["entry_mode"] == "chat"
        assert len(data_create["tasks"]) == 1
        assert len(data_create["schedule_result"]["scheduled_blocks"]) == 1

        # 2. Reschedule over HTTP using task returned above
        resched_resp = client.post(
            "/api/v1/planning/chat",
            json={
                "message": "Move my Operating Systems session to 20:00",
                "target_date": "2099-11-02",
                "tasks": data_create["tasks"],
                "entry_mode": "chat",
            },
        )
        assert resched_resp.status_code == 200
        data_resched = resched_resp.json()
        assert data_resched["schedule_result"]["scheduled_blocks"][0]["start"] == "20:00"
        assert data_resched["tasks"][0]["pinned_start"] == "20:00"

    def test_http_chat_mocked_tool_calling(self, monkeypatch):
        import sys
        import types
        from fastapi.testclient import TestClient
        from langchain_core.messages import AIMessage

        class FakeBoundLLM:
            def invoke(self, conv):
                return AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "create_task",
                            "args": {
                                "title": "Distributed Consensus",
                                "estimated_mins": 25,
                                "cognitive_load": "high",
                                "priority_weight": 4.5,
                            },
                            "id": "call_http_1",
                        }
                    ],
                )

        class FakeLLM:
            def __init__(self, **kwargs):
                pass

            def bind_tools(self, tools):
                return FakeBoundLLM()

            def invoke(self, conv):
                return AIMessage(content="Added Distributed Consensus (25 mins) and scheduled it.")

        fake_mod = types.ModuleType("langchain_google_genai")
        fake_mod.ChatGoogleGenerativeAI = FakeLLM
        monkeypatch.setitem(sys.modules, "langchain_google_genai", fake_mod)
        monkeypatch.setattr("app.config.GEMINI_API_KEY", "fake-tool-key")

        client = TestClient(app)
        resp = client.post(
            "/api/v1/planning/chat",
            json={
                "message": "Please add a Distributed Consensus session",
                "target_date": "2099-11-05",
                "tasks": [],
                "entry_mode": "chat",
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "Distributed Consensus" in body["response"]
        assert len(body["tasks"]) == 1
        assert body["tasks"][0]["title"] == "Distributed Consensus"
        assert len(body["schedule_result"]["scheduled_blocks"]) == 1

    def test_http_chat_invalid_entry_mode_returns_422_and_500_never_leaks_exception(self, monkeypatch):
        from fastapi.testclient import TestClient

        client = TestClient(app)

        # Invalid entry_mode must fail fast at Pydantic validation -> 422, never 500
        invalid_resp = client.post(
            "/api/v1/planning/chat",
            json={
                "message": "Hello",
                "entry_mode": "invalid_mode",
            },
        )
        assert invalid_resp.status_code == 422

        # Unexpected internal exception must return 500 with generic safe detail (no str(e) leak)
        def _boom():
            raise RuntimeError("SECRET_DB_CONNECTION_STRING_LEAK")

        monkeypatch.setattr("app.api.routes.build_graph", _boom)
        err_resp = client.post(
            "/api/v1/planning/chat",
            json={"message": "Hello", "entry_mode": "chat"},
        )
        assert err_resp.status_code == 500
        detail = err_resp.json()["detail"]
        assert "SECRET_DB_CONNECTION_STRING_LEAK" not in detail
        assert "internal error" in detail.lower()

    def test_cors_does_not_combine_wildcard_and_credentials(self):
        from fastapi.testclient import TestClient

        client = TestClient(app)
        resp = client.get("/", headers={"Origin": "http://localhost:3000"})
        assert resp.status_code == 200
        allow_origin = resp.headers.get("access-control-allow-origin")
        allow_creds = resp.headers.get("access-control-allow-credentials")
        assert not (allow_origin == "*" and allow_creds == "true")
        assert allow_origin == "http://localhost:3000"
