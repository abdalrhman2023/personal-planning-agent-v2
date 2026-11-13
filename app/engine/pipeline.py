"""Unified schedule execution pipeline shared by HTTP routes and the LangGraph agent.

Centralizes:
  1. Cleaning up stale Google Calendar events for pending/re-scheduled tasks.
  2. Auto-fetching external Google Calendar busy blocks and merging with request busy blocks.
  3. Running the deterministic optimizer.
  4. Creating new Google Calendar events and syncing ``gcal_event_ids`` onto tasks.
  5. Computing outbound telemetry signals.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from app.engine.optimizer import optimize_schedule
from app.engine.telemetry import compute_outbound_signals
from app.integrations.gcal_service import GoogleCalendarService
from app.schema import (
    AdaptiveModelContext,
    FixedLecture,
    OutboundSignals,
    ScheduleRequest,
    ScheduleResult,
    ScheduledBlock,
    StudyTask,
)


@dataclass
class ScheduleSyncResult:
    """Output of schedule optimization + Google Calendar synchronization."""

    schedule_result: ScheduleResult
    updated_tasks: List[StudyTask]
    gcal_events: List[Dict[str, Any]]
    merged_gcal_busy_blocks: List[Dict[str, str]]


@dataclass
class PlanningPipelineResult:
    """Complete pipeline output including optimization, GCal sync, and telemetry."""

    schedule_result: ScheduleResult
    updated_tasks: List[StudyTask]
    gcal_events: List[Dict[str, Any]]
    merged_gcal_busy_blocks: List[Dict[str, str]]
    outbound_signals: OutboundSignals


def _merge_busy_blocks(
    fetched: List[Dict[str, str]],
    provided: List[Dict[str, str]],
    target_date: str,
) -> List[Dict[str, str]]:
    """Combine auto-fetched GCal busy blocks with caller-provided blocks without duplicates."""
    merged: List[Dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()

    for blk in list(provided) + list(fetched):
        start = blk.get("start", "")
        end = blk.get("end", "")
        if not start or not end:
            continue
        blk_date = blk.get("date") or target_date
        key = (blk_date, start, end)
        if key not in seen:
            seen.add(key)
            merged.append(dict(blk))

    return merged


def _delete_stale_gcal_events(
    gcal: GoogleCalendarService,
    tasks: List[StudyTask],
    pending_task_ids: set[str],
    prior_gcal_events: Optional[List[Dict[str, Any]]] = None,
) -> None:
    """Delete previous GCal events belonging to tasks being re-scheduled."""
    kept_event_ids = {
        eid
        for t in tasks
        if t.task_id not in pending_task_ids
        for eid in t.gcal_event_ids
        if eid
    }

    ids_to_delete: List[str] = []
    seen_del: set[str] = set()
    for t in tasks:
        if t.task_id in pending_task_ids:
            for eid in t.gcal_event_ids:
                if eid and eid not in seen_del:
                    seen_del.add(eid)
                    ids_to_delete.append(eid)

    for old_ev in prior_gcal_events or []:
        old_id = old_ev.get("id")
        if old_id and old_id not in kept_event_ids and old_id not in seen_del:
            seen_del.add(old_id)
            ids_to_delete.append(old_id)

    for old_id in ids_to_delete:
        try:
            gcal.delete_calendar_event(old_id)
        except Exception:
            pass


def run_schedule_and_gcal_sync(
    target_date: str,
    tasks: List[StudyTask],
    university_schedule: Optional[List[FixedLecture]] = None,
    gcal_busy_blocks: Optional[List[Dict[str, str]]] = None,
    adaptive_model: Optional[AdaptiveModelContext] = None,
    prior_gcal_events: Optional[List[Dict[str, Any]]] = None,
    gcal_service: Optional[GoogleCalendarService] = None,
) -> ScheduleSyncResult:
    """Clean stale events, fetch busy blocks, run optimizer, and push new GCal events."""
    gcal = gcal_service or GoogleCalendarService()
    lectures = list(university_schedule or [])
    adaptive = adaptive_model or AdaptiveModelContext()
    pending_task_ids = {t.task_id for t in tasks if t.status == "pending"}

    # 1. Remove stale events for pending/re-scheduled tasks BEFORE reading busy blocks
    _delete_stale_gcal_events(gcal, tasks, pending_task_ids, prior_gcal_events)

    # 2. Auto-fetch external GCal busy blocks and merge with explicit request blocks
    try:
        fetched_busy = gcal.get_busy_blocks(target_date)
    except Exception:
        fetched_busy = []
    merged_busy = _merge_busy_blocks(fetched_busy, list(gcal_busy_blocks or []), target_date)

    # 3. Deterministic optimization
    request = ScheduleRequest(
        target_date=target_date,
        tasks=tasks,
        university_schedule=lectures,
        gcal_busy_blocks=merged_busy,
        adaptive_model=adaptive,
    )
    result: ScheduleResult = optimize_schedule(request)

    # 4. Push newly scheduled blocks to Google Calendar
    events_created: List[Dict[str, Any]] = []
    task_events_map: Dict[str, List[str]] = defaultdict(list)
    for block in result.scheduled_blocks:
        try:
            evt = gcal.create_calendar_event(
                summary=f"Study: {block.title}",
                date=block.date,
                start_time=block.start,
                end_time=block.end,
                description=f"Task {block.task_id} – Part {block.part}",
            )
            events_created.append(evt)
            if evt.get("id"):
                task_events_map[block.task_id].append(evt["id"])
        except Exception:
            pass  # best-effort calendar sync

    # 5. Update task statuses, spans, and complete gcal_event_ids lists
    blocks_by_task: Dict[str, List[ScheduledBlock]] = defaultdict(list)
    for b in result.scheduled_blocks:
        blocks_by_task[b.task_id].append(b)

    updated_tasks: List[StudyTask] = []
    for task in tasks:
        t_copy = task.model_copy(deep=True)
        if t_copy.task_id in blocks_by_task:
            t_blocks = blocks_by_task[t_copy.task_id]
            t_copy.status = "scheduled"
            t_copy.gcal_event_ids = list(task_events_map.get(t_copy.task_id, []))
            t_copy.scheduled_date = t_blocks[0].date
            t_copy.scheduled_start = t_blocks[0].start
            t_copy.scheduled_end = t_blocks[-1].end
        elif t_copy.task_id in pending_task_ids:
            t_copy.gcal_event_ids = []
        updated_tasks.append(t_copy)

    return ScheduleSyncResult(
        schedule_result=result,
        updated_tasks=updated_tasks,
        gcal_events=events_created,
        merged_gcal_busy_blocks=merged_busy,
    )


def execute_planning_pipeline(
    target_date: str,
    tasks: List[StudyTask],
    university_schedule: Optional[List[FixedLecture]] = None,
    gcal_busy_blocks: Optional[List[Dict[str, str]]] = None,
    adaptive_model: Optional[AdaptiveModelContext] = None,
    prior_gcal_events: Optional[List[Dict[str, Any]]] = None,
    gcal_service: Optional[GoogleCalendarService] = None,
    now: Optional[datetime] = None,
) -> PlanningPipelineResult:
    """Run schedule optimization, Google Calendar sync, and outbound telemetry together."""
    sync_res = run_schedule_and_gcal_sync(
        target_date=target_date,
        tasks=tasks,
        university_schedule=university_schedule,
        gcal_busy_blocks=gcal_busy_blocks,
        adaptive_model=adaptive_model,
        prior_gcal_events=prior_gcal_events,
        gcal_service=gcal_service,
    )

    signals = compute_outbound_signals(
        scheduled_blocks=sync_res.schedule_result.scheduled_blocks,
        tasks=sync_res.updated_tasks,
        university_schedule=list(university_schedule or []),
        gcal_busy_blocks=sync_res.merged_gcal_busy_blocks,
        target_date=target_date,
        feasibility_flags=sync_res.schedule_result.feasibility_flags,
        now=now,
    )

    return PlanningPipelineResult(
        schedule_result=sync_res.schedule_result,
        updated_tasks=sync_res.updated_tasks,
        gcal_events=sync_res.gcal_events,
        merged_gcal_busy_blocks=sync_res.merged_gcal_busy_blocks,
        outbound_signals=signals,
    )
