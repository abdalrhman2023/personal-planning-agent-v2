"""LangChain tool definitions for the Personal Planning Agent.

Defines the three tools the LLM can call:
  - create_task       – add a new StudyTask into the state
  - reschedule_block  – move a scheduled block to a new time
  - optimize_schedule – trigger the deterministic optimizer
"""

from __future__ import annotations

import json
import uuid
from typing import Optional

from langchain_core.tools import tool

from app.schema import (
    AdaptiveModelContext,
    FixedLecture,
    ScheduleRequest,
    ScheduleResult,
    StudyTask,
)
from app.engine.optimizer import optimize_schedule as run_optimizer


# ---------------------------------------------------------------------------
# 1. create_task
# ---------------------------------------------------------------------------

@tool
def create_task(
    title: str,
    estimated_mins: int,
    cognitive_load: str = "medium",
    priority_weight: float = 3.0,
    source: str = "user",
    deadline: Optional[str] = None,
    topic_ref: Optional[str] = None,
    parent_milestone_id: Optional[str] = None,
) -> str:
    """Create a new study task and return it as JSON.

    Parameters
    ----------
    title : str
        Short name for the task.
    estimated_mins : int
        Estimated duration in minutes.
    cognitive_load : str
        One of "high", "medium", "low".
    priority_weight : float
        From 1.0 (low) to 5.0 (urgent).
    source : str
        One of "user", "life_coach", "assessment_remediation".
    deadline : str, optional
        ISO deadline  YYYY-MM-DDTHH:MM:SS.
    topic_ref : str, optional
        Reference to curriculum topic.
    parent_milestone_id : str, optional
        ID of the parent milestone from Life Coach.
    """
    task = StudyTask(
        task_id=f"task_{uuid.uuid4().hex[:8]}",
        title=title,
        source=source,
        estimated_mins=estimated_mins,
        priority_weight=priority_weight,
        cognitive_load=cognitive_load,
        deadline=deadline,
        topic_ref=topic_ref,
        parent_milestone_id=parent_milestone_id,
    )
    return task.model_dump_json()


# ---------------------------------------------------------------------------
# 2. reschedule_block
# ---------------------------------------------------------------------------

@tool
def reschedule_block(
    task_id: str,
    new_start: str,
    new_end: str,
    new_date: Optional[str] = None,
) -> str:
    """Reschedule a study block to a hard-pinned time slot.

    Returns a JSON payload instructing the execution node to set
    ``pinned_start``/``pinned_end`` and reset the task status to ``"pending"``
    so the deterministic optimizer re-places it and updates Google Calendar.

    Parameters
    ----------
    task_id : str
        ID of the task whose block should be moved.
    new_start : str
        New start time in HH:MM format.
    new_end : str
        New end time in HH:MM format.
    new_date : str, optional
        New date in YYYY-MM-DD format.  Omit to keep the same date.
    """
    return json.dumps({
        "action": "reschedule_block",
        "task_id": task_id,
        "new_start": new_start,
        "new_end": new_end,
        "new_date": new_date,
        "pinned_start": new_start,
        "pinned_end": new_end,
        "status": "pending",
    })


# ---------------------------------------------------------------------------
# 3. optimize_schedule
# ---------------------------------------------------------------------------

@tool
def optimize_schedule(
    target_date: str,
    tasks_json: str,
    university_schedule_json: str = "[]",
    gcal_busy_blocks_json: str = "[]",
    adaptive_model_json: str = "{}",
) -> str:
    """Run the deterministic schedule optimizer and return the result as JSON.

    All list/object parameters are passed as JSON strings so the LLM can
    construct them easily.

    Parameters
    ----------
    target_date : str
        The date to optimise for (YYYY-MM-DD).
    tasks_json : str
        JSON array of StudyTask objects.
    university_schedule_json : str
        JSON array of FixedLecture objects.
    gcal_busy_blocks_json : str
        JSON array of {start, end} busy-block dicts.
    adaptive_model_json : str
        JSON object with AdaptiveModelContext fields.
    """
    tasks = [StudyTask(**t) for t in json.loads(tasks_json)]
    lectures = [FixedLecture(**l) for l in json.loads(university_schedule_json)]
    gcal_busy = json.loads(gcal_busy_blocks_json)
    adaptive_kw = json.loads(adaptive_model_json)
    adaptive = AdaptiveModelContext(**adaptive_kw) if adaptive_kw else AdaptiveModelContext()

    request = ScheduleRequest(
        target_date=target_date,
        tasks=tasks,
        university_schedule=lectures,
        gcal_busy_blocks=gcal_busy,
        adaptive_model=adaptive,
    )
    result: ScheduleResult = run_optimizer(request)
    return result.model_dump_json()


# ---------------------------------------------------------------------------
# Convenience: list of all tools for the agent graph
# ---------------------------------------------------------------------------

ALL_TOOLS = [create_task, reschedule_block, optimize_schedule]
