"""LangGraph agent graph for the Personal Planning Agent.

Implements a StateGraph with two entry modes:
  1. Background / Control Signal Mode  (from Adaptive Manager, Life Coach,
     or Assessment Agent)
  2. Conversational Chat Mode  (direct student requests)

Node pipeline:
  RouterNode → DeconstructorOrToolLLMNode → OptimizerAndGCalExecutionNode
             → TelemetryAndOutboundSignalsNode
"""

from __future__ import annotations

from datetime import datetime, timedelta
import json
import re
import uuid
from typing import Annotated, Literal, Optional

from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, ConfigDict, Field

from app import config
from app.agent.prompts import SYSTEM_PROMPT
from app.agent.tools import ALL_TOOLS
from app.config import get_today_date_str
from app.engine.deconstructor import deconstruct_task
from app.engine.pipeline import run_schedule_and_gcal_sync
from app.engine.telemetry import compute_outbound_signals
from app.schema import (
    AdaptiveModelContext,
    FixedLecture,
    OutboundSignals,
    ScheduleResult,
    StudyTask,
)


# ---------------------------------------------------------------------------
# Graph State
# ---------------------------------------------------------------------------

class PlanningState(BaseModel):
    """Shared state flowing through the graph."""

    # --- Input fields ---
    messages: Annotated[list[BaseMessage], add_messages] = Field(default_factory=list)
    entry_mode: Literal["chat", "control_signal", ""] = ""

    # --- Study data (mutable across nodes) ---
    tasks: list[dict] = Field(default_factory=list)
    university_schedule: list[dict] = Field(default_factory=list)
    gcal_busy_blocks: list[dict] = Field(default_factory=list)
    adaptive_model: dict = Field(default_factory=dict)
    target_date: str = ""
    state_dirty: bool = False

    # --- Intermediate results ---
    schedule_result: Optional[dict] = None
    gcal_events_created: list[dict] = Field(default_factory=list)

    # --- Outputs ---
    outbound_signals: Optional[dict] = None
    response_text: str = ""

    model_config = ConfigDict(arbitrary_types_allowed=True)


# ---------------------------------------------------------------------------
# Node 1: Router
# ---------------------------------------------------------------------------

def router_node(state: PlanningState) -> dict:
    """Classify the incoming request as chat or control_signal.

    For programmatic invocations the caller can pre-set ``entry_mode``; if it
    is already set we skip classification.
    """
    if state.entry_mode:
        return {}  # already classified

    last_msg = state.messages[-1] if state.messages else None
    if last_msg is None:
        return {"entry_mode": "chat"}

    content = last_msg.content if isinstance(last_msg.content, str) else str(last_msg.content)

    # Fast heuristic: if the message looks like JSON with known signal keys,
    # classify as control_signal without an LLM call.
    try:
        payload = json.loads(content)
        if isinstance(payload, dict):
            signal_keys = {"signal_type", "command", "milestones", "remediation_topics"}
            if signal_keys & set(payload.keys()):
                return {"entry_mode": "control_signal"}
    except (json.JSONDecodeError, TypeError):
        pass

    return {"entry_mode": "chat"}


# ---------------------------------------------------------------------------
# Node 2: Deconstructor / Tool-LLM
# ---------------------------------------------------------------------------

def _deconstruct_to_task_dicts(task: StudyTask, focus_duration: int) -> list[dict]:
    """Split *task* into micro-task dicts via deconstruct_task when over *focus_duration*."""
    if task.estimated_mins <= focus_duration:
        return [task.model_dump()]

    dr = deconstruct_task(task, focus_duration)
    out: list[dict] = []
    for i, mt in enumerate(dr.micro_tasks):
        sub = StudyTask(
            task_id=f"{task.task_id}_p{i+1}",
            parent_milestone_id=task.parent_milestone_id,
            title=mt.title,
            source=task.source,
            estimated_mins=mt.estimated_mins,
            priority_weight=task.priority_weight,
            cognitive_load=mt.cognitive_load,
            deadline=task.deadline,
            topic_ref=mt.topic_ref or task.topic_ref,
            status=task.status,
        )
        out.append(sub.model_dump())
    return out


def _add_or_deconstruct_task(
    title: str,
    estimated_mins: int,
    priority_weight: float = 3.0,
    cognitive_load: str = "medium",
    topic_ref: Optional[str] = None,
    source: str = "user",
    parent_milestone_id: Optional[str] = None,
    deadline: Optional[str] = None,
    focus_duration: int = 25,
) -> list[dict]:
    """Helper to create a task or deconstruct it into microtasks if it exceeds focus duration."""
    task = StudyTask(
        task_id=f"task_{uuid.uuid4().hex[:8]}",
        title=title,
        source=source,
        estimated_mins=estimated_mins,
        priority_weight=priority_weight,
        cognitive_load=cognitive_load,
        topic_ref=topic_ref,
        parent_milestone_id=parent_milestone_id,
        deadline=deadline,
        status="pending",
    )
    return _deconstruct_to_task_dicts(task, focus_duration)


def _compute_end_time(start_hhmm: str, duration_mins: int) -> str:
    """Return 'HH:MM' equal to *start_hhmm* + *duration_mins* (capped cleanly in day if needed)."""
    start_dt = datetime.strptime(start_hhmm, "%H:%M")
    end_dt = start_dt + timedelta(minutes=max(1, duration_mins))
    return end_dt.strftime("%H:%M")


def _normalize_hhmm(raw: str) -> Optional[str]:
    """Validate and zero-pad an 'HH:MM' time string."""
    match = re.fullmatch(r"\s*(\d{1,2}):(\d{2})\s*", str(raw))
    if not match:
        return None
    hour, minute = int(match.group(1)), int(match.group(2))
    if 0 <= hour <= 23 and 0 <= minute <= 59:
        return f"{hour:02d}:{minute:02d}"
    return None


def _extract_time_from_groups(
    hour_str: str,
    minute_str: Optional[str],
    meridiem: Optional[str],
) -> Optional[str]:
    """Convert parsed hour/minute/meridiem tokens into 'HH:MM'."""
    hour = int(hour_str)
    minute = int(minute_str) if minute_str is not None else 0
    if minute < 0 or minute > 59:
        return None

    if meridiem:
        mer = meridiem.lower().replace(".", "")
        is_pm = mer in ("pm", "مساء", "مساءً")
        is_am = mer in ("am", "صباحا", "صباحاً")
        if hour < 1 or hour > 12:
            return None
        if is_pm and hour < 12:
            hour += 12
        elif is_am and hour == 12:
            hour = 0
    elif hour < 0 or hour > 23:
        return None

    return f"{hour:02d}:{minute:02d}"


def _find_task_for_reschedule(
    tasks: list[dict],
    task_id: Optional[str] = None,
    message: str = "",
) -> Optional[dict]:
    """Locate the target task dict by task_id, title overlap, or single active fallback."""
    if not tasks:
        return None

    if task_id:
        for t in tasks:
            if t.get("task_id") == task_id:
                return t
        tid_lower = task_id.strip().lower()
        for t in tasks:
            if (t.get("title") or "").strip().lower() == tid_lower:
                return t

    msg_lower = (message or "").lower()
    if msg_lower:
        # Check explicit task_id mention in text
        for t in tasks:
            tid = (t.get("task_id") or "").lower()
            if tid and re.search(r"\b" + re.escape(tid) + r"\b", msg_lower):
                return t

        # Score by meaningful word overlap with task title
        best_task: Optional[dict] = None
        best_score = 0
        stop_words = {"study", "task", "session", "block", "move", "reschedule", "my", "the", "to", "at", "for"}
        for t in tasks:
            if t.get("status") == "completed":
                continue
            title_lower = (t.get("title") or "").lower()
            if title_lower and title_lower in msg_lower:
                return t
            tokens = [
                w for w in re.findall(r"[a-zA-Z\u0600-\u06FF0-9]+", title_lower)
                if len(w) >= 3 and w not in stop_words
            ]
            score = sum(1 for tok in tokens if tok in msg_lower)
            if score > best_score:
                best_score = score
                best_task = t
        if best_task is not None and best_score > 0:
            return best_task

        # Disambiguate time-of-day hints against scheduled tasks
        active = [t for t in tasks if t.get("status") != "completed"]
        if any(hint in msg_lower for hint in ("evening", "مساء")):
            evening = [t for t in active if (t.get("scheduled_start") or "") >= "17:00"]
            if len(evening) == 1:
                return evening[0]
        if any(hint in msg_lower for hint in ("morning", "صباح")):
            morning = [t for t in active if "00:00" < (t.get("scheduled_start") or "") < "12:00"]
            if len(morning) == 1:
                return morning[0]

    active = [t for t in tasks if t.get("status") != "completed"]
    if len(active) == 1:
        return active[0]
    return None


def _apply_reschedule_to_task(
    task_dict: dict,
    new_start: str,
    new_end: Optional[str],
    new_date: Optional[str],
    default_mins: int,
) -> Optional[tuple[str, str]]:
    """Pin *task_dict* to the requested slot and reset its status to pending."""
    norm_start = _normalize_hhmm(new_start)
    if norm_start is None:
        return None
    norm_end = _normalize_hhmm(new_end) if new_end else None
    if norm_end is None:
        dur = int(task_dict.get("estimated_mins") or default_mins or 25)
        norm_end = _compute_end_time(norm_start, dur)

    task_dict["pinned_start"] = norm_start
    task_dict["pinned_end"] = norm_end
    if new_date:
        task_dict["scheduled_date"] = new_date
    task_dict["status"] = "pending"
    return norm_start, norm_end


def _build_llm_context_prompt(
    tasks: list[dict],
    university_schedule: list[dict],
    gcal_busy_blocks: list[dict],
    target_date: str,
    adaptive: AdaptiveModelContext,
) -> str:
    """Serialize current planning state concisely for LLM tool decisions."""
    context_payload = {
        "target_date": target_date,
        "tasks": [
            {
                "task_id": t.get("task_id"),
                "title": t.get("title"),
                "status": t.get("status"),
                "estimated_mins": t.get("estimated_mins"),
                "priority_weight": t.get("priority_weight"),
                "cognitive_load": t.get("cognitive_load"),
                "scheduled_date": t.get("scheduled_date"),
                "scheduled_start": t.get("scheduled_start"),
                "scheduled_end": t.get("scheduled_end"),
                "pinned_start": t.get("pinned_start"),
                "pinned_end": t.get("pinned_end"),
            }
            for t in tasks
        ],
        "university_schedule": university_schedule,
        "gcal_busy_blocks": gcal_busy_blocks,
        "adaptive_model": adaptive.model_dump(),
    }
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"## Current Planning Context\n"
        f"Use the exact `task_id` values below when calling `reschedule_block`.\n"
        f"```json\n{json.dumps(context_payload, ensure_ascii=False)}\n```"
    )


def _parse_natural_language_intent(
    message: str,
    focus_duration: int = 25,
    existing_tasks: Optional[list[dict]] = None,
) -> Optional[dict]:
    """Extract reschedule, optimize, or task creation parameters from natural language."""
    if not message or not isinstance(message, str):
        return None

    # 1. Reschedule intent ("move", "reschedule", "انقل")
    if re.search(r"(?:\b(?:move|reschedule)\b|انقل)", message, re.IGNORECASE):
        date_match = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", message)
        new_date = date_match.group(1) if date_match else None
        time_text = re.sub(r"\b\d{4}-\d{2}-\d{2}\b", " ", message)

        new_start: Optional[str] = None
        new_end: Optional[str] = None

        range_match = re.search(
            r"(\d{1,2}:\d{2})\s*(?:-|to|until|حتى|إلى|الى)\s*(\d{1,2}:\d{2})",
            time_text,
            re.IGNORECASE,
        )
        if range_match:
            new_start = _normalize_hhmm(range_match.group(1))
            new_end = _normalize_hhmm(range_match.group(2))
        else:
            # Prefer explicit HH:MM
            hhmm_match = re.search(
                r"\b(\d{1,2}):(\d{2})\s*(am|pm|a\.m\.|p\.m\.|صباحا|صباحاً|مساء|مساءً)?\b",
                time_text,
                re.IGNORECASE,
            )
            if hhmm_match:
                new_start = _extract_time_from_groups(
                    hhmm_match.group(1), hhmm_match.group(2), hhmm_match.group(3)
                )
            else:
                hour_match = re.search(
                    r"(?:to|at|إلى|الى|لـ|للساعة|الساعة)\s*(\d{1,2})\s*(am|pm|a\.m\.|p\.m\.|صباحا|صباحاً|مساء|مساءً)?(?!\s*(?:mins?|minutes?|hrs?|hours?|دقيقة|دقائق))(?=\s|$|[.,!?;])",
                    time_text,
                    re.IGNORECASE,
                )
                if hour_match:
                    new_start = _extract_time_from_groups(
                        hour_match.group(1), None, hour_match.group(2)
                    )

        target_task = _find_task_for_reschedule(existing_tasks or [], message=message)
        return {
            "action": "reschedule",
            "task_id": target_task.get("task_id") if target_task else None,
            "new_start": new_start,
            "new_end": new_end,
            "new_date": new_date,
        }

    # 2. Pure optimize intent without task details
    if re.search(
        r"^\s*(?:please\s+)?(?:optimize|re-?optimize)\s+(?:my\s+)?schedule[.!]?\s*$",
        message,
        re.IGNORECASE,
    ):
        return {"action": "optimize"}

    # 3. Task creation intent
    keywords = ["study", "task", "review", "read", "practice", "schedule", "add", "create", "plan", "session", "block"]
    if not any(re.search(r'\b' + re.escape(kw) + r'\b', message, re.IGNORECASE) for kw in keywords):
        return None

    # Duration extraction
    dur_match = re.search(r'(\d+)\s*(?:mins?|minutes?|m\b)', message, re.IGNORECASE)
    if dur_match:
        dur = int(dur_match.group(1))
    else:
        dur_h = re.search(r'(\d+)\s*(?:hours?|hrs?|h\b)', message, re.IGNORECASE)
        dur = int(dur_h.group(1)) * 60 if dur_h else focus_duration

    # Priority extraction
    prio_num = re.search(r'priority\s*[:=]?\s*([1-5](?:\.\d+)?)', message, re.IGNORECASE)
    if prio_num:
        prio = float(prio_num.group(1))
    elif re.search(r'\b(high priority|urgent|critical|important)\b', message, re.IGNORECASE):
        prio = 4.5
    elif re.search(r'\b(low priority|casual|light|easy)\b', message, re.IGNORECASE):
        prio = 2.0
    else:
        prio = 3.0

    load = "high" if prio >= 4.0 else ("low" if prio <= 2.0 else "medium")

    # Clean Title extraction
    t = message
    t = re.sub(r'^(?:please\s+)?(?:add|create|schedule|plan|insert|study)\s+', '', t, flags=re.IGNORECASE)
    t = re.sub(r'^(?:a\s+|an\s+)?(?:(?:high|medium|low)\s+priority\s+)?(?:study\s+)?(?:task|session|block)?(?:\s+for)?\s*', '', t, flags=re.IGNORECASE)
    t = re.sub(r'^(?:task\s*[:=]|\d+[- ](?:minute|min)\s+(?:study\s+)?(?:session|block|task)?(?:\s+for)?)\s*', '', t, flags=re.IGNORECASE)
    t = re.sub(r'^[:\s]+', '', t)
    t = re.sub(r'\s*(?:tomorrow|today|tonight|for\s+\d+\s*(?:mins?|minutes?|hours?|hrs?|m\b|h\b)|\b\d+\s*(?:mins?|minutes?|hours?|hrs?)\b|at\s+\d{1,2}(?::\d{2})?|priority\s*[:=]?\s*[1-5](?:\.\d+)?|and\s+optimize.*|optimize.*).*$', '', t, flags=re.IGNORECASE)
    title = t.strip(' ,.-:') or "Study Session"

    topic_ref = re.sub(r'[^\w\s]', '', title).replace(' ', '_')

    return {
        "action": "create",
        "title": title,
        "estimated_mins": dur,
        "priority_weight": prio,
        "cognitive_load": load,
        "topic_ref": topic_ref,
    }


def deconstructor_or_tool_llm_node(state: PlanningState) -> dict:
    """Handle control signals directly OR run the tool-calling LLM / fallback parser.

    - Control signals: parse the JSON, create tasks / apply restructure, mark state_dirty=True.
    - Chat: invoke the LLM with state context and tools bound, close any ToolMessage loop
      with a follow-up summary turn, and track state_dirty cleanly.
    """
    new_tasks = [dict(t) for t in state.tasks]
    initial_tasks_count = len(new_tasks)
    adaptive = AdaptiveModelContext(**(state.adaptive_model or {}))
    focus_duration = adaptive.effective_focus_duration_mins or 25
    target_date = state.target_date or get_today_date_str()
    messages_out: list[BaseMessage] = []
    state_dirty = False

    if state.entry_mode == "control_signal":
        last_content = state.messages[-1].content if state.messages else "{}"
        try:
            payload = json.loads(last_content)
        except (json.JSONDecodeError, TypeError):
            payload = {}

        signal_type = payload.get("signal_type") or payload.get("command", "")

        # --- PREEMPTIVE_SCHEDULE_RESTRUCTURE from Adaptive Manager ---
        if signal_type == "PREEMPTIVE_SCHEDULE_RESTRUCTURE":
            adaptive_updates = payload.get("adaptive_model_updates", {})
            adaptive_dict = adaptive.model_dump()
            adaptive_dict.update(adaptive_updates)
            adaptive_dict["current_system_mode"] = "PREEMPTIVE_RESTRUCTURE"

            # Reset non-completed tasks back to pending for re-optimization
            for t in new_tasks:
                if t.get("status") != "completed":
                    t["status"] = "pending"

            messages_out.append(
                AIMessage(content="Received PREEMPTIVE_SCHEDULE_RESTRUCTURE signal. "
                          "Switching to restructure mode and re-optimising.")
            )
            return {
                "tasks": new_tasks,
                "target_date": target_date,
                "adaptive_model": adaptive_dict,
                "messages": messages_out,
                "state_dirty": True,
            }

        # --- Milestones from Life Coach ---
        if signal_type == "MILESTONE_TASKS" or "milestones" in payload:
            milestones = payload.get("milestones", payload.get("tasks", []))
            focus = adaptive.effective_focus_duration_mins
            for ms in milestones:
                task = StudyTask(
                    task_id=ms.get("task_id", f"lc_{len(new_tasks)}"),
                    title=ms.get("title", "Milestone task"),
                    source="life_coach",
                    estimated_mins=ms.get("estimated_mins", focus),
                    priority_weight=ms.get("priority_weight", 3.0),
                    cognitive_load=ms.get("cognitive_load", "medium"),
                    deadline=ms.get("deadline"),
                    topic_ref=ms.get("topic_ref"),
                    parent_milestone_id=ms.get("parent_milestone_id"),
                )
                # Deconstruct if large (shared helper unifies LLM invocation and topic_ref fallback)
                new_tasks.extend(_deconstruct_to_task_dicts(task, focus))

            messages_out.append(
                AIMessage(content=f"Ingested {len(milestones)} milestone(s) from Life Coach.")
            )
            return {
                "tasks": new_tasks,
                "target_date": target_date,
                "messages": messages_out,
                "state_dirty": True,
            }

        # --- Remediation from Assessment Agent ---
        if signal_type == "REMEDIATION_TASKS" or "remediation_topics" in payload:
            topics = payload.get("remediation_topics", payload.get("tasks", []))
            for rt in topics:
                task = StudyTask(
                    task_id=rt.get("task_id", f"rem_{len(new_tasks)}"),
                    title=rt.get("title", "Remediation"),
                    source="assessment_remediation",
                    estimated_mins=rt.get("estimated_mins", 30),
                    priority_weight=rt.get("priority_weight", 4.5),
                    cognitive_load=rt.get("cognitive_load", "high"),
                    deadline=rt.get("deadline"),
                    topic_ref=rt.get("topic_ref"),
                )
                new_tasks.append(task.model_dump())
            messages_out.append(
                AIMessage(content=f"Ingested {len(topics)} remediation task(s) from Assessment Agent.")
            )
            return {
                "tasks": new_tasks,
                "target_date": target_date,
                "messages": messages_out,
                "state_dirty": True,
            }

        # Unknown signal – pass through
        messages_out.append(AIMessage(content=f"Received unknown signal: {signal_type}"))
        return {"messages": messages_out, "target_date": target_date, "state_dirty": False}

    # ---------- Chat mode: invoke LLM with context + tools ----------
    user_msg = state.messages[-1].content if state.messages else ""
    user_msg_str = (
        user_msg
        if isinstance(user_msg, str)
        else (
            " ".join(
                [c.get("text", "") if isinstance(c, dict) else str(c) for c in user_msg]
            )
            if isinstance(user_msg, list)
            else str(user_msg)
        )
    )

    ai_content = ""
    llm_responded_cleanly = False
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI

        if not config.can_invoke_llm(ChatGoogleGenerativeAI):
            raise RuntimeError("LLM invocation disabled (missing API key or unmocked test run)")

        api_key = getattr(config, "GEMINI_API_KEY", "")
        system_prompt_with_ctx = _build_llm_context_prompt(
            tasks=new_tasks,
            university_schedule=state.university_schedule,
            gcal_busy_blocks=state.gcal_busy_blocks,
            target_date=target_date,
            adaptive=adaptive,
        )
        conversation: list[BaseMessage] = [SystemMessage(content=system_prompt_with_ctx)] + list(state.messages)

        active_llm = None
        for candidate_model in config.get_gemini_candidate_models():
            try:
                active_llm = ChatGoogleGenerativeAI(
                    model=candidate_model,
                    google_api_key=api_key,
                    timeout=15,
                    max_retries=1,
                )
                ai_msg: AIMessage = active_llm.bind_tools(ALL_TOOLS).invoke(conversation)
                break
            except Exception:
                active_llm = None
                continue

        if active_llm is None:
            raise RuntimeError("All configured Gemini models failed")

        llm_responded_cleanly = True
        messages_out.append(ai_msg)
        conversation.append(ai_msg)
        if ai_msg.content:
            ai_content = ai_msg.content if isinstance(ai_msg.content, str) else str(ai_msg.content)

        # Execute any tool calls the LLM made and feed ToolMessages back for clean summary
        if ai_msg.tool_calls:
            tool_map = {t.name: t for t in ALL_TOOLS}
            tool_summaries: list[str] = []

            for tc in ai_msg.tool_calls:
                tool_fn = tool_map.get(tc["name"])
                if not tool_fn:
                    continue

                result_str = tool_fn.invoke(tc["args"])
                tool_msg_content = result_str

                if tc["name"] == "create_task":
                    raw_task = json.loads(result_str)
                    created_blocks = _add_or_deconstruct_task(
                        title=raw_task.get("title", "Study Task"),
                        estimated_mins=raw_task.get("estimated_mins", focus_duration),
                        priority_weight=raw_task.get("priority_weight", 3.0),
                        cognitive_load=raw_task.get("cognitive_load", "medium"),
                        topic_ref=raw_task.get("topic_ref"),
                        source=raw_task.get("source", "user"),
                        parent_milestone_id=raw_task.get("parent_milestone_id"),
                        deadline=raw_task.get("deadline"),
                        focus_duration=focus_duration,
                    )
                    new_tasks.extend(created_blocks)
                    state_dirty = True
                    tool_summaries.append(
                        f"Created task '{raw_task.get('title', 'Study Task')}' ({raw_task.get('estimated_mins', focus_duration)} mins)."
                    )
                elif tc["name"] == "reschedule_block":
                    resched = json.loads(result_str)
                    tid = resched.get("task_id")
                    matched = _find_task_for_reschedule(new_tasks, task_id=tid, message=user_msg_str)
                    if matched is not None:
                        applied = _apply_reschedule_to_task(
                            matched,
                            new_start=resched.get("new_start", ""),
                            new_end=resched.get("new_end"),
                            new_date=resched.get("new_date"),
                            default_mins=focus_duration,
                        )
                        if applied is not None:
                            norm_start, norm_end = applied
                            if resched.get("new_date"):
                                target_date = resched["new_date"]
                            state_dirty = True
                            tool_msg_content = json.dumps({
                                **resched,
                                "status": "pending",
                                "pinned_start": norm_start,
                                "pinned_end": norm_end,
                                "applied": True,
                            })
                            tool_summaries.append(
                                f"Pinned '{matched.get('title', matched.get('task_id'))}' to {norm_start}-{norm_end} for rescheduling."
                            )
                        else:
                            tool_msg_content = json.dumps({
                                "error": f"Invalid time format for reschedule_block: {resched.get('new_start')}"
                            })
                            tool_summaries.append(f"Could not parse target time '{resched.get('new_start')}'.")
                    else:
                        tool_msg_content = json.dumps({
                            "error": f"Task '{tid}' not found in current schedule."
                        })
                        tool_summaries.append(f"No matching task '{tid}' found to reschedule.")
                elif tc["name"] == "optimize_schedule":
                    state_dirty = True
                    tool_summaries.append("Triggered schedule optimization.")

                t_msg = ToolMessage(content=tool_msg_content, tool_call_id=tc["id"])
                messages_out.append(t_msg)
                conversation.append(t_msg)

            # Close ToolMessage loop with a final conversational turn (no tools bound)
            final_text = ""
            try:
                followup_msg: AIMessage = active_llm.invoke(conversation)
                if followup_msg and followup_msg.content:
                    final_text = (
                        followup_msg.content
                        if isinstance(followup_msg.content, str)
                        else str(followup_msg.content)
                    )
                    messages_out.append(followup_msg)
            except Exception:
                final_text = ""

            if not final_text:
                fallback_summary = " ".join(tool_summaries) or "Completed requested schedule action."
                messages_out.append(AIMessage(content=fallback_summary))

    except Exception:
        # LLM unavailable, timed out, or threw an error
        llm_responded_cleanly = False

    # Fallback Parser: run when LLM failed/unavailable and no mutation happened yet
    if not llm_responded_cleanly and len(new_tasks) == initial_tasks_count and not state_dirty and user_msg_str:
        parsed = _parse_natural_language_intent(
            user_msg_str,
            focus_duration=focus_duration,
            existing_tasks=new_tasks,
        )
        if parsed:
            action = parsed.get("action", "create")
            if action == "reschedule":
                matched = _find_task_for_reschedule(
                    new_tasks, task_id=parsed.get("task_id"), message=user_msg_str
                )
                if matched is not None and parsed.get("new_start"):
                    applied = _apply_reschedule_to_task(
                        matched,
                        new_start=parsed["new_start"],
                        new_end=parsed.get("new_end"),
                        new_date=parsed.get("new_date"),
                        default_mins=focus_duration,
                    )
                    if applied is not None:
                        norm_start, norm_end = applied
                        if parsed.get("new_date"):
                            target_date = parsed["new_date"]
                        state_dirty = True
                        messages_out.append(
                            AIMessage(
                                content=(
                                    f"Rescheduled '{matched.get('title', matched.get('task_id'))}' "
                                    f"to {norm_start}-{norm_end} and triggered schedule optimization."
                                )
                            )
                        )
                    else:
                        messages_out.append(
                            AIMessage(content="Could not parse the requested reschedule time window.")
                        )
                elif matched is None:
                    messages_out.append(
                        AIMessage(content="Could not find an active study task matching your reschedule request.")
                    )
                else:
                    messages_out.append(
                        AIMessage(content="Please specify the target time (e.g. 19:30) to move your study session.")
                    )
            elif action == "optimize":
                state_dirty = bool(new_tasks)
                messages_out.append(
                    AIMessage(content="Optimizing your current study schedule.")
                )
            else:
                created_blocks = _add_or_deconstruct_task(
                    title=parsed["title"],
                    estimated_mins=parsed["estimated_mins"],
                    priority_weight=parsed["priority_weight"],
                    cognitive_load=parsed["cognitive_load"],
                    topic_ref=parsed["topic_ref"],
                    source="user",
                    focus_duration=focus_duration,
                )
                new_tasks.extend(created_blocks)
                state_dirty = True
                if not ai_content:
                    messages_out.append(
                        AIMessage(
                            content=f"Created task '{parsed['title']}' ({parsed['estimated_mins']} mins) and triggered schedule optimization."
                        )
                    )

    if not messages_out and user_msg_str:
        messages_out.append(
            AIMessage(
                content=f"Received: '{user_msg_str}'. No schedule changes were needed."
            )
        )

    return {
        "tasks": new_tasks,
        "target_date": target_date,
        "messages": messages_out,
        "state_dirty": state_dirty,
    }


# ---------------------------------------------------------------------------
# Node 3: Optimizer & GCal Execution
# ---------------------------------------------------------------------------

def optimizer_and_gcal_node(state: PlanningState) -> dict:
    """Execute unified optimizer + GCal sync pipeline on current graph state."""
    if not state.target_date:
        return {}

    tasks = [StudyTask(**t) for t in state.tasks]
    lectures = [FixedLecture(**l) for l in state.university_schedule]
    adaptive = AdaptiveModelContext(**(state.adaptive_model or {}))

    sync_res = run_schedule_and_gcal_sync(
        target_date=state.target_date,
        tasks=tasks,
        university_schedule=lectures,
        gcal_busy_blocks=state.gcal_busy_blocks,
        adaptive_model=adaptive,
        prior_gcal_events=state.gcal_events_created,
    )
    result = sync_res.schedule_result

    messages_out: list[BaseMessage] = []
    if result.is_feasible:
        messages_out.append(AIMessage(
            content=f"Schedule optimised: {len(result.scheduled_blocks)} block(s) "
                    f"placed for {state.target_date}."
        ))
    else:
        messages_out.append(AIMessage(
            content=f"Schedule partially feasible: "
                    f"{len(result.unscheduled_tasks)} task(s) could not be placed."
        ))

    return {
        "tasks": [t.model_dump() for t in sync_res.updated_tasks],
        "schedule_result": result.model_dump(),
        "gcal_events_created": sync_res.gcal_events,
        "gcal_busy_blocks": sync_res.merged_gcal_busy_blocks,
        "messages": messages_out,
        "state_dirty": False,
    }


# ---------------------------------------------------------------------------
# Node 4: Telemetry & Outbound Signals
# ---------------------------------------------------------------------------

def telemetry_and_signals_node(state: PlanningState) -> dict:
    """Compute telemetry metrics and produce outbound signals."""
    if not state.schedule_result:
        return {}

    sr = ScheduleResult(**state.schedule_result)
    tasks = [StudyTask(**t) for t in state.tasks]
    lectures = [FixedLecture(**l) for l in state.university_schedule]

    target_date = state.target_date or get_today_date_str()
    signals: OutboundSignals = compute_outbound_signals(
        scheduled_blocks=sr.scheduled_blocks,
        tasks=tasks,
        university_schedule=lectures,
        gcal_busy_blocks=state.gcal_busy_blocks,
        target_date=target_date,
        feasibility_flags=sr.feasibility_flags,
    )

    response_parts: list[str] = []
    density = signals.workload_density_metrics.get("workload_density", 0)
    response_parts.append(
        f"Workload density: {density:.0%}"
    )
    if signals.to_adaptive_manager:
        active = signals.to_adaptive_manager.get("active_signals")
        if active:
            response_parts.append(
                f"[Alert] Sent to Adaptive Manager: {', '.join(active)}"
            )
        else:
            response_parts.append(
                f"[Alert] Sent to Adaptive Manager: "
                f"{signals.to_adaptive_manager.get('signal_type', signals.to_adaptive_manager.get('message', ''))}"
            )
    if signals.to_life_coach:
        response_parts.append("[Report] Feasibility report sent to Life Coach.")
    if signals.to_ai_tutor:
        response_parts.append(
            f"[Tutor] {len(signals.to_ai_tutor)} study block invitation(s) sent to AI Tutor."
        )

    return {
        "outbound_signals": signals.model_dump(),
        "response_text": " | ".join(response_parts),
    }


# ---------------------------------------------------------------------------
# Conditional edge: should we run the optimizer?
# ---------------------------------------------------------------------------

def _should_optimize(state: PlanningState) -> str:
    """Run optimizer only when target_date, tasks, and state_dirty are present."""
    if state.target_date and state.tasks and state.state_dirty:
        return "optimizer_and_gcal"
    return "telemetry_and_signals"


# ---------------------------------------------------------------------------
# Build the graph
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    """Construct and compile the Planning Agent StateGraph."""

    graph = StateGraph(PlanningState)

    # Add nodes
    graph.add_node("router", router_node)
    graph.add_node("deconstructor_or_tool_llm", deconstructor_or_tool_llm_node)
    graph.add_node("optimizer_and_gcal", optimizer_and_gcal_node)
    graph.add_node("telemetry_and_signals", telemetry_and_signals_node)

    # Edges
    graph.set_entry_point("router")
    graph.add_edge("router", "deconstructor_or_tool_llm")
    graph.add_conditional_edges(
        "deconstructor_or_tool_llm",
        _should_optimize,
        {
            "optimizer_and_gcal": "optimizer_and_gcal",
            "telemetry_and_signals": "telemetry_and_signals",
        },
    )
    graph.add_edge("optimizer_and_gcal", "telemetry_and_signals")
    graph.add_edge("telemetry_and_signals", END)

    return graph.compile()
