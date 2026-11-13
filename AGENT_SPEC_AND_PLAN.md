# MASTER SPECIFICATION & EXECUTION PLAN: PERSONAL PLANNING AGENT (AGENT 3 OF 5)

## 1. STRICT ANTI-HALLUCINATION RULES FOR AI IDE (ANTIGRAVITY)
You are building Agent 3 (Personal Planning Agent) of a 5-Agent Graduation Project (Generative AI Personal Life & Study Coach)[cite: 1].
Follow these strict rules:
1. NO HALLUCINATIONS: Do not invent unrequested fields or APIs. Stick strictly to the schemas and workflows defined below.
2. PHASED EXECUTION ONLY: Execute ONLY ONE PHASE AT A TIME as requested by the user. Wait for terminal test verification before moving to the next phase.
3. HYBRID ARCHITECTURE: Never use the LLM to calculate time math or resolve calendar conflicts directly. The LLM is ONLY used for:
   - Conversational intent understanding.
   - Task Deconstruction (splitting large tasks into micro-tasks matching the student's focus duration)[cite: 1].
   - Calling deterministic Python tools: create_task(), reschedule_block(), optimize_schedule()[cite: 1].
   All actual time-slot allocation and conflict resolution MUST happen inside the Python algorithmic solver (app/engine/optimizer.py).
4. MODEL CONFIGURATION: Load GEMINI_API_KEY and GEMINI_MODEL_NAME (defaulting to "gemini-3.5-flash" or whatever is set in .env) from app/config.py. Never hardcode model names.
5. TEST VERIFICATION: At the end of each phase, run pytest in the terminal and ensure 100% pass rate.

---

## 2. MULTI-AGENT SYSTEM CONTEXT & INTERACTIONS
- Agent 1 (AI Tutor Agent): Receives "Study Block Invitations & Context Payload" from our Planning Agent at scheduled block times[cite: 2].
- Agent 2 (Assessment Agent): Sends "Required Remediation Topics & Urgency Weights" to our Planning Agent after grading quizzes[cite: 8].
- Agent 3 (Personal Planning Agent - THIS PROJECT): Builds conflict-free schedules, deconstructs tasks, manages fatigue, and syncs with Google Calendar[cite: 1, 2].
- Agent 4 (Goal & Life Coach Agent): Sends "Decomposed Milestones & Micro-Tasks" to our Planning Agent[cite: 11]. Receives "Milestone Feasibility Reports" if the student's calendar lacks capacity[cite: 2].
- Agent 5 (Adaptive Manager Agent - Orchestrator): Owns the adaptive_model[cite: 4]. Sends "PREEMPTIVE_SCHEDULE_RESTRUCTURE" commands to our Planning Agent[cite: 5]. Receives "Completion Rates & Schedule Slip Metrics" (when real completion velocity deviates from plan by > 20%) and "Schedule Feasibility Flags" (when deadlines are unreachable)[cite: 1, 2].

---

## 3. UNIVERSAL DATA SCHEMA (app/schema.py)
The Planning Agent reads/writes "study_data" and reads "adaptive_model"[cite: 1]:

1. FixedLecture (inside study_data.university_schedule):
   - id: str
   - day: str (e.g., "Monday")
   - date: str (YYYY-MM-DD)
   - start: str (HH:MM, 24-hour format)
   - end: str (HH:MM, 24-hour format)
   - title: str

2. StudyTask (inside study_data.tasks):
   - task_id: str
   - parent_milestone_id: Optional[str]
   - title: str
   - source: Literal["user", "life_coach", "assessment_remediation"]
   - topic_ref: Optional[str]
   - estimated_mins: int
   - adjusted_mins: Optional[int]
   - deadline: Optional[str] (ISO format YYYY-MM-DDTHH:MM:SS)
   - priority_weight: float (1.0 to 5.0)
   - cognitive_load: Literal["high", "medium", "low"]
   - status: Literal["pending", "scheduled", "completed", "missed", "delayed"]
   - scheduled_date: Optional[str] (YYYY-MM-DD)
   - scheduled_start: Optional[str] (HH:MM)
   - scheduled_end: Optional[str] (HH:MM)
   - gcal_event_id: Optional[str]

3. AdaptiveModelContext (inside adaptive_model):
   - peak_productivity_window: str (e.g., "09:00-11:00")[cite: 1]
   - secondary_productivity_window: str (e.g., "19:00-22:00")
   - effective_focus_duration_mins: int (e.g., 25 or 45)[cite: 4]
   - avg_task_completion_ratio: float (e.g., 0.75)[cite: 1]
   - task_delay_factor: float (e.g., 1.2)[cite: 4]
   - max_daily_study_mins: int (default 240)
   - current_system_mode: Literal["NORMAL", "PREEMPTIVE_RESTRUCTURE"][cite: 4]

4. OutboundSignals (produced by Planning Agent):
   - to_adaptive_manager: Optional[dict] (Schedule Feasibility Flags OR Velocity Slip > 20% alert)[cite: 1, 2]
   - to_life_coach: Optional[dict] (Milestone Feasibility Report)[cite: 2]
   - to_ai_tutor: List[dict] (Study Block Invitations & Context Payload)[cite: 2]
   - workload_density_metrics: dict[cite: 1]

---

## 4. TARGET PROJECT STRUCTURE
- .env.example
- requirements.txt
- AGENT_SPEC_AND_PLAN.md
- app/
  - __init__.py
  - config.py
  - schema.py
  - engine/
    - __init__.py
    - optimizer.py
    - deconstructor.py
    - telemetry.py
  - integrations/
    - __init__.py
    - gcal_service.py
  - agent/
    - __init__.py
    - tools.py
    - prompts.py
    - graph.py
  - api/
    - __init__.py
    - routes.py
- tests/
  - test_phase1_optimizer.py
  - test_phase2_deconstructor_telemetry.py
  - test_phase3_gcal.py
  - test_phase4_agent_graph.py
- main.py
- demo_simulation.py

---

## 5. IMPLEMENTATION PHASES

### PHASE 1: Config, Schemas, and Deterministic Schedule Optimizer
Create: requirements.txt, .env.example, app/config.py, app/schema.py, app/engine/optimizer.py, tests/test_phase1_optimizer.py.
Details for app/engine/optimizer.py:
- Build a pure Python deterministic scheduler.
- Step 1 (Constraint Ingestion): For each target date, start with operating hours (08:00 to 22:00) and subtract fixed university_schedule blocks and external Google Calendar busy blocks[cite: 1].
- Step 2 (Duration & Chunking Check): Multiply estimated_mins by task_delay_factor if current_system_mode == "PREEMPTIVE_RESTRUCTURE". Ensure no single scheduled block exceeds effective_focus_duration_mins (split into sequential parts with a 10-minute break between consecutive study blocks)[cite: 1, 4].
- Step 3 (Prioritization): Sort pending tasks by priority_weight, giving highest priority to source == "assessment_remediation" and earliest deadlines[cite: 1, 8].
- Step 4 (Cognitive Window Matching & Fatigue Management): Fit "high" cognitive_load tasks into peak_productivity_window slots first[cite: 1]. Respect max_daily_study_mins per day to prevent burnout[cite: 2].
- Step 5 (Feasibility Verification): If any task cannot fit before its deadline, set is_feasible = False and populate the Feasibility Flags for Adaptive Manager and Life Coach[cite: 1, 2].

### PHASE 2: Telemetry Engine & LLM Task Deconstructor
Create: app/engine/telemetry.py, app/engine/deconstructor.py, tests/test_phase2_deconstructor_telemetry.py.
Details:
- telemetry.py:
  1. Compute workload_density_metrics (scheduled study mins / total free mins per day)[cite: 1].
  2. Compute velocity slip: compare completed tasks vs scheduled/missed tasks. If deviation > 0.20 (20%), emit the alert payload for Adaptive Manager[cite: 2].
  3. Generate Study Block Invitations & Context Payload for AI Tutor Agent[cite: 2].
- deconstructor.py:
  1. Use ChatGoogleGenerativeAI (reading GEMINI_MODEL_NAME from config.py) with structured outputs to break large milestones/tasks into micro-tasks matching effective_focus_duration_mins[cite: 1].
  2. Provide a clean deterministic fallback splitter if GEMINI_API_KEY is not configured during offline unit tests.

### PHASE 3: Google Calendar API Integration
Create: app/integrations/gcal_service.py, tests/test_phase3_gcal.py.
Details:
- Implement GoogleCalendarService using google-api-python-client and google-auth-oauthlib.
- Functions: get_busy_blocks(), create_calendar_event(), update_calendar_event(), delete_calendar_event().
- Implement automatic mock_mode = True if credentials.json is not present in the root directory, allowing full local testing without crashing, while seamlessly switching to real Google Calendar API calls when credentials.json is provided.

### PHASE 4: LangGraph Agent & Tool-Driven Execution
Create: app/agent/tools.py, app/agent/prompts.py, app/agent/graph.py, tests/test_phase4_agent_graph.py.
Details:
- Define LangChain tools: create_task(), reschedule_block(), optimize_schedule()[cite: 1].
- Build a LangGraph StateGraph supporting two entry modes:
  1. Background/Control Signal Mode (handling PREEMPTIVE_SCHEDULE_RESTRUCTURE from Adaptive Manager, milestones from Life Coach, or remediation tasks from Assessment Agent)[cite: 1, 2, 5, 8, 11].
  2. Conversational Chat Mode (handling direct user requests like "Move my morning study session to the evening").
- Connect the nodes: RouterNode -> DeconstructorOrToolLLMNode -> OptimizerAndGCalExecutionNode -> TelemetryAndOutboundSignalsNode.

### PHASE 5: FastAPI Server & End-to-End Simulation
Create: app/api/routes.py, main.py, demo_simulation.py.
Details:
- Build FastAPI endpoints (/api/v1/planning/optimize, /api/v1/planning/chat, /api/v1/planning/telemetry) ready for integration with the Frontend and Orchestrator.
- Build demo_simulation.py demonstrating:
  1. Initial schedule optimization from university timetable + Life Coach milestones[cite: 1].
  2. Handling a PREEMPTIVE_SCHEDULE_RESTRUCTURE signal from Adaptive Manager[cite: 2, 5].
  3. Detecting >20% velocity slip and alerting the Adaptive Manager[cite: 2].