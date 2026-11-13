"""Universal data schemas for the Personal Planning Agent."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.config import get_today_date_str


# ---------------------------------------------------------------------------
# 1. FixedLecture (inside study_data.university_schedule)
# ---------------------------------------------------------------------------
class FixedLecture(BaseModel):
    id: str
    day: str  # e.g. "Monday"
    date: str  # YYYY-MM-DD
    start: str  # HH:MM, 24-hour
    end: str  # HH:MM, 24-hour
    title: str


# ---------------------------------------------------------------------------
# 2. StudyTask (inside study_data.tasks)
# ---------------------------------------------------------------------------
class StudyTask(BaseModel):
    task_id: str
    parent_milestone_id: Optional[str] = None
    title: str
    source: Literal["user", "life_coach", "assessment_remediation"]
    topic_ref: Optional[str] = None
    estimated_mins: int
    adjusted_mins: Optional[int] = None
    deadline: Optional[str] = None  # ISO YYYY-MM-DDTHH:MM:SS
    priority_weight: float = Field(ge=1.0, le=5.0)
    cognitive_load: Literal["high", "medium", "low"]
    status: Literal["pending", "scheduled", "completed", "missed", "delayed"] = "pending"
    scheduled_date: Optional[str] = None  # YYYY-MM-DD
    scheduled_start: Optional[str] = None  # HH:MM
    scheduled_end: Optional[str] = None  # HH:MM
    pinned_start: Optional[str] = None  # HH:MM hard-constraint start
    pinned_end: Optional[str] = None  # HH:MM hard-constraint end
    gcal_event_ids: List[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _coerce_legacy_gcal_event_id(cls, data: Any) -> Any:
        if isinstance(data, dict):
            data = dict(data)
            legacy_id = data.pop("gcal_event_id", None)
            if legacy_id and not data.get("gcal_event_ids"):
                data["gcal_event_ids"] = [legacy_id]
        return data

    @property
    def gcal_event_id(self) -> Optional[str]:
        """Backward-compatible single event ID accessor."""
        return self.gcal_event_ids[0] if self.gcal_event_ids else None

    @gcal_event_id.setter
    def gcal_event_id(self, value: Optional[str]) -> None:
        self.gcal_event_ids = [value] if value else []


# ---------------------------------------------------------------------------
# 3. AdaptiveModelContext (inside adaptive_model – owned by Agent 5)
# ---------------------------------------------------------------------------
class AdaptiveModelContext(BaseModel):
    peak_productivity_window: str = "09:00-11:00"
    secondary_productivity_window: str = "19:00-22:00"
    effective_focus_duration_mins: int = 25
    avg_task_completion_ratio: float = 0.75
    task_delay_factor: float = 1.2
    max_daily_study_mins: int = 240
    current_system_mode: Literal["NORMAL", "PREEMPTIVE_RESTRUCTURE"] = "NORMAL"


# ---------------------------------------------------------------------------
# 4. OutboundSignals (produced by Planning Agent)
# ---------------------------------------------------------------------------
class OutboundSignals(BaseModel):
    to_adaptive_manager: Optional[dict] = None
    to_life_coach: Optional[dict] = None
    to_ai_tutor: List[dict] = Field(default_factory=list)
    workload_density_metrics: dict = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Convenience container for a scheduling request / result
# ---------------------------------------------------------------------------
class ScheduleRequest(BaseModel):
    """Input bundle handed to the optimizer.

    Supports either flat fields (tasks, university_schedule) or the nested
    study_data dictionary per the specification.
    """
    target_date: str = Field(default_factory=get_today_date_str)
    tasks: List[StudyTask] = Field(default_factory=list)
    university_schedule: List[FixedLecture] = Field(default_factory=list)
    gcal_busy_blocks: List[Dict[str, str]] = Field(default_factory=list)
    adaptive_model: AdaptiveModelContext = Field(default_factory=AdaptiveModelContext)
    study_data: Optional[Dict[str, Any]] = None

    @model_validator(mode="before")
    @classmethod
    def extract_nested_study_data(cls, data: Any) -> Any:
        if isinstance(data, dict):
            sd = data.get("study_data")
            if isinstance(sd, dict):
                if "tasks" in sd and not data.get("tasks"):
                    data["tasks"] = sd["tasks"]
                if "university_schedule" in sd and not data.get("university_schedule"):
                    data["university_schedule"] = sd["university_schedule"]
        return data

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        json_schema_extra={
            "example": {
                "target_date": "2026-11-02",
                "tasks": [
                    {
                        "task_id": "t101",
                        "title": "Distributed Systems Practice",
                        "source": "user",
                        "estimated_mins": 50,
                        "priority_weight": 4.5,
                        "cognitive_load": "high",
                        "status": "pending",
                        "deadline": "2026-11-03T23:59:59",
                        "topic_ref": "CS301_Distributed"
                    },
                    {
                        "task_id": "t102",
                        "title": "Review Linear Algebra Chapter 3",
                        "source": "life_coach",
                        "estimated_mins": 30,
                        "priority_weight": 3.0,
                        "cognitive_load": "medium",
                        "status": "pending",
                        "topic_ref": "MATH205_LinearAlgebra"
                    }
                ],
                "university_schedule": [
                    {
                        "id": "lec_1",
                        "day": "Monday",
                        "date": "2026-11-02",
                        "start": "09:00",
                        "end": "10:30",
                        "title": "CS301: Distributed Systems"
                    }
                ],
                "gcal_busy_blocks": [
                    {
                        "start": "16:00",
                        "end": "16:45"
                    }
                ],
                "adaptive_model": {
                    "peak_productivity_window": "10:30-12:30",
                    "secondary_productivity_window": "18:00-21:00",
                    "effective_focus_duration_mins": 25,
                    "avg_task_completion_ratio": 0.85,
                    "task_delay_factor": 1.2,
                    "max_daily_study_mins": 240,
                    "current_system_mode": "NORMAL"
                }
            }
        }
    )


class ScheduledBlock(BaseModel):
    """One contiguous study block produced by the optimizer."""
    task_id: str
    title: str
    date: str  # YYYY-MM-DD
    start: str  # HH:MM
    end: str  # HH:MM
    part: int = 1  # chunk sequence number when a task is split


class ScheduleResult(BaseModel):
    """Output of optimize_schedule()."""
    is_feasible: bool = True
    scheduled_blocks: List[ScheduledBlock] = Field(default_factory=list)
    unscheduled_tasks: List[str] = Field(default_factory=list)  # task_ids
    feasibility_flags: Optional[dict] = None
