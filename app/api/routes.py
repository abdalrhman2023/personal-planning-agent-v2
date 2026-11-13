"""FastAPI HTTP Routes for the Personal Planning Agent (Agent 3).

Exposes the following endpoints:
  - POST /api/v1/planning/optimize   – Run deterministic optimizer + calendar sync
  - POST /api/v1/planning/chat       – Interactive LangGraph agent conversation / control signal
  - POST /api/v1/planning/telemetry  – Compute workload density, velocity slip, and tutor signals
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Literal, Optional

from fastapi import APIRouter, HTTPException
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel, ConfigDict, Field

from app.agent.graph import build_graph
from app.config import get_today_date_str
from app.engine.pipeline import execute_planning_pipeline
from app.engine.telemetry import compute_outbound_signals
from app.schema import (
    AdaptiveModelContext,
    FixedLecture,
    OutboundSignals,
    ScheduleRequest,
    ScheduleResult,
    ScheduledBlock,
    StudyTask,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/planning", tags=["planning"])


# ---------------------------------------------------------------------------
# Request & Response Models
# ---------------------------------------------------------------------------

class OptimizeResponse(BaseModel):
    is_feasible: bool
    scheduled_blocks: List[ScheduledBlock] = Field(default_factory=list)
    unscheduled_tasks: List[str] = Field(default_factory=list)
    feasibility_flags: Optional[dict] = None
    tasks: List[StudyTask] = Field(default_factory=list)
    schedule_result: Optional[ScheduleResult] = None
    outbound_signals: Optional[OutboundSignals] = None
    gcal_events: List[Dict[str, Any]] = Field(default_factory=list)


class ChatRequest(BaseModel):
    message: str = "Please schedule a 25-minute study block for Algorithms today"
    target_date: Optional[str] = Field(default_factory=get_today_date_str)
    tasks: List[StudyTask] = Field(default_factory=list)
    university_schedule: List[FixedLecture] = Field(default_factory=list)
    gcal_busy_blocks: List[Dict[str, str]] = Field(default_factory=list)
    adaptive_model: AdaptiveModelContext = Field(default_factory=AdaptiveModelContext)
    entry_mode: Optional[Literal["chat", "control_signal", ""]] = None

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "message": "Move my evening study session to 19:30",
                "target_date": "2026-11-02",
                "tasks": [
                    {
                        "task_id": "task_algo",
                        "title": "Algorithms Practice",
                        "source": "user",
                        "estimated_mins": 25,
                        "priority_weight": 3.5,
                        "cognitive_load": "medium",
                        "status": "pending"
                    }
                ],
                "university_schedule": [],
                "gcal_busy_blocks": [],
                "adaptive_model": {
                    "peak_productivity_window": "09:00-11:00",
                    "secondary_productivity_window": "19:00-22:00",
                    "effective_focus_duration_mins": 25,
                    "max_daily_study_mins": 240,
                    "current_system_mode": "NORMAL"
                },
                "entry_mode": "chat"
            }
        }
    )


class ChatResponse(BaseModel):
    response: str
    entry_mode: str
    tasks: List[StudyTask] = Field(default_factory=list)
    schedule_result: Optional[ScheduleResult] = None
    outbound_signals: Optional[OutboundSignals] = None
    gcal_events: List[Dict[str, Any]] = Field(default_factory=list)


class TelemetryRequest(BaseModel):
    target_date: str = Field(default_factory=get_today_date_str)
    scheduled_blocks: List[ScheduledBlock] = Field(default_factory=list)
    tasks: List[StudyTask] = Field(default_factory=list)
    university_schedule: List[FixedLecture] = Field(default_factory=list)
    gcal_busy_blocks: List[Dict[str, str]] = Field(default_factory=list)
    feasibility_flags: Optional[dict] = None

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "target_date": "2026-11-02",
                "scheduled_blocks": [
                    {
                        "task_id": "task_1",
                        "title": "Calculus Study Block",
                        "date": "2026-11-02",
                        "start": "11:00",
                        "end": "11:25",
                        "part": 1
                    }
                ],
                "tasks": [
                    {
                        "task_id": "task_1",
                        "title": "Calculus Study Block",
                        "source": "user",
                        "estimated_mins": 25,
                        "priority_weight": 4.0,
                        "cognitive_load": "high",
                        "status": "completed"
                    }
                ],
                "university_schedule": [],
                "gcal_busy_blocks": []
            }
        }
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/optimize", response_model=OptimizeResponse)
def optimize_schedule_endpoint(request: ScheduleRequest) -> OptimizeResponse:
    """Run the unified planning pipeline for target_date and sync to Google Calendar."""
    try:
        pipeline_res = execute_planning_pipeline(
            target_date=request.target_date,
            tasks=request.tasks,
            university_schedule=request.university_schedule,
            gcal_busy_blocks=request.gcal_busy_blocks,
            adaptive_model=request.adaptive_model,
        )
        sr = pipeline_res.schedule_result

        return OptimizeResponse(
            is_feasible=sr.is_feasible,
            scheduled_blocks=sr.scheduled_blocks,
            unscheduled_tasks=sr.unscheduled_tasks,
            feasibility_flags=sr.feasibility_flags,
            tasks=pipeline_res.updated_tasks,
            schedule_result=sr,
            outbound_signals=pipeline_res.outbound_signals,
            gcal_events=pipeline_res.gcal_events,
        )
    except HTTPException:
        raise
    except Exception:
        logger.exception("Unhandled error in /api/v1/planning/optimize")
        raise HTTPException(
            status_code=500,
            detail="An internal error occurred while optimizing the schedule.",
        )


@router.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest) -> ChatResponse:
    """Handle conversational user requests or multi-agent control signals via LangGraph."""
    try:
        graph = build_graph()
        initial_state = {
            "messages": [HumanMessage(content=request.message)],
            "entry_mode": request.entry_mode or "",
            "target_date": request.target_date or get_today_date_str(),
            "tasks": [t.model_dump() for t in request.tasks],
            "university_schedule": [l.model_dump() for l in request.university_schedule],
            "gcal_busy_blocks": request.gcal_busy_blocks,
            "adaptive_model": request.adaptive_model.model_dump(),
        }

        final_state = graph.invoke(initial_state)

        # Extract clean AI reply messages (never leak raw ToolMessage JSON)
        response_text = final_state.get("response_text", "")
        messages = final_state.get("messages", [])
        ai_contents = [
            str(m.content).strip()
            for m in messages
            if isinstance(m, AIMessage) and getattr(m, "content", None) and str(m.content).strip()
        ]
        if ai_contents:
            msg_content = "\n".join(dict.fromkeys(ai_contents))
            if response_text:
                response_text = f"{msg_content}\n\n[Telemetry] {response_text}"
            else:
                response_text = msg_content

        updated_tasks = [StudyTask(**t) for t in final_state.get("tasks", [])]
        raw_sr = final_state.get("schedule_result")
        schedule_result = ScheduleResult(**raw_sr) if raw_sr else None

        raw_signals = final_state.get("outbound_signals")
        outbound_signals = OutboundSignals(**raw_signals) if raw_signals else None

        return ChatResponse(
            response=response_text or "Processed successfully.",
            entry_mode=final_state.get("entry_mode", "chat"),
            tasks=updated_tasks,
            schedule_result=schedule_result,
            outbound_signals=outbound_signals,
            gcal_events=final_state.get("gcal_events_created", []),
        )
    except HTTPException:
        raise
    except Exception:
        logger.exception("Unhandled error in /api/v1/planning/chat")
        raise HTTPException(
            status_code=500,
            detail="An internal error occurred while processing the chat request.",
        )


@router.post("/telemetry", response_model=OutboundSignals)
def telemetry_endpoint(request: TelemetryRequest) -> OutboundSignals:
    """Generate telemetry metrics, velocity slip analysis, and study block invitations."""
    try:
        signals = compute_outbound_signals(
            scheduled_blocks=request.scheduled_blocks,
            tasks=request.tasks,
            university_schedule=request.university_schedule,
            gcal_busy_blocks=request.gcal_busy_blocks,
            target_date=request.target_date,
            feasibility_flags=request.feasibility_flags,
        )
        return signals
    except HTTPException:
        raise
    except Exception:
        logger.exception("Unhandled error in /api/v1/planning/telemetry")
        raise HTTPException(
            status_code=500,
            detail="An internal error occurred while computing telemetry signals.",
        )
