"""Deterministic Schedule Optimizer – pure Python, no LLM calls.

Implements the 5-step scheduling algorithm specified in the master plan:
  1. Constraint Ingestion  (build free slots from operating hours minus blocks)
  2. Duration & Chunking   (apply delay factor, split by focus duration)
  3. Prioritization         (sort tasks by remediation source, deadline, weight)
  4. Cognitive Window Matching & Fatigue Management
  5. Feasibility Verification
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import re
from typing import Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

from app import config
from app.schema import (
    AdaptiveModelContext,
    ScheduleRequest,
    ScheduleResult,
    ScheduledBlock,
    StudyTask,
)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# Minutes since midnight
_DAY_START = 8 * 60   # 08:00
_DAY_END = 22 * 60    # 22:00
_BREAK_MINS = 10       # break between consecutive study blocks


def _hhmm_to_mins(t: str) -> int:
    """Convert 'HH:MM' to minutes since midnight."""
    h, m = t.split(":")
    return int(h) * 60 + int(m)


def _mins_to_hhmm(m: int) -> str:
    """Convert minutes since midnight to 'HH:MM'."""
    return f"{m // 60:02d}:{m % 60:02d}"


@dataclass(order=True)
class _Interval:
    """Half-open interval [start, end) in minutes since midnight."""
    start: int
    end: int

    @property
    def length(self) -> int:
        return max(0, self.end - self.start)


def _subtract_intervals(
    free: List[_Interval], busy: List[_Interval]
) -> List[_Interval]:
    """Remove *busy* intervals from *free* intervals, returning what remains."""
    result: List[_Interval] = []
    for f in free:
        remaining = [_Interval(f.start, f.end)]
        for b in busy:
            new_remaining: List[_Interval] = []
            for r in remaining:
                # No overlap
                if b.end <= r.start or b.start >= r.end:
                    new_remaining.append(r)
                else:
                    # Left piece
                    if b.start > r.start:
                        new_remaining.append(_Interval(r.start, b.start))
                    # Right piece
                    if b.end < r.end:
                        new_remaining.append(_Interval(b.end, r.end))
            remaining = new_remaining
        result.extend(remaining)
    # Sort and filter out zero-length
    return sorted([s for s in result if s.length > 0], key=lambda s: s.start)


def _parse_window(window_str: str) -> _Interval:
    """Parse 'HH:MM-HH:MM' into an _Interval."""
    parts = window_str.split("-")
    return _Interval(_hhmm_to_mins(parts[0]), _hhmm_to_mins(parts[1]))


def _overlaps(slot: _Interval, window: _Interval) -> int:
    """Return minutes of overlap between *slot* and *window*."""
    overlap_start = max(slot.start, window.start)
    overlap_end = min(slot.end, window.end)
    return max(0, overlap_end - overlap_start)


# ---------------------------------------------------------------------------
# Task preparation helpers
# ---------------------------------------------------------------------------

@dataclass
class _TaskChunk:
    """A chunk of a task that fits within one focus session."""
    task_id: str
    title: str
    duration_mins: int
    cognitive_load: str  # "high", "medium", "low"
    part: int
    priority_sort_key: Tuple  # for sorting


def _prepare_tasks(
    tasks: List[StudyTask],
    adaptive: AdaptiveModelContext,
) -> List[_TaskChunk]:
    """Step 2 & 3 – compute effective duration, chunk, and sort."""
    chunks: List[_TaskChunk] = []

    for task in tasks:
        if task.status != "pending":
            continue

        # --- Step 2: Duration & Chunking ---
        effective_mins = task.estimated_mins
        if adaptive.current_system_mode == "PREEMPTIVE_RESTRUCTURE":
            effective_mins = int(effective_mins * adaptive.task_delay_factor)

        focus = adaptive.effective_focus_duration_mins
        part = 1
        remaining = effective_mins
        while remaining > 0:
            chunk_len = min(remaining, focus)
            # Sort key (Step 3): assessment_remediation first (0), then
            # by deadline ascending (earliest first), then by
            # priority_weight descending (highest first).
            source_rank = 0 if task.source == "assessment_remediation" else 1
            # Use a very late date for tasks with no deadline so they sort last
            deadline_key = task.deadline if task.deadline else "9999-12-31T23:59:59"
            sort_key = (source_rank, deadline_key, -task.priority_weight)

            chunks.append(
                _TaskChunk(
                    task_id=task.task_id,
                    title=task.title,
                    duration_mins=chunk_len,
                    cognitive_load=task.cognitive_load,
                    part=part,
                    priority_sort_key=sort_key,
                )
            )
            remaining -= chunk_len
            part += 1

    # --- Step 3: Sort ---
    chunks.sort(key=lambda c: c.priority_sort_key)
    return chunks


# ---------------------------------------------------------------------------
# Slot allocation (Step 4)
# ---------------------------------------------------------------------------

def _consume_slot(
    available: List[_Interval],
    idx: int,
    block_start: int,
    block_end: int,
) -> None:
    """Remove [block_start, block_end) plus surrounding breaks from available[idx]."""
    slot = available[idx]
    left_end = block_start - _BREAK_MINS
    right_start = block_end + _BREAK_MINS
    left_piece = _Interval(slot.start, left_end) if left_end > slot.start else None
    right_piece = _Interval(right_start, slot.end) if right_start < slot.end else None

    available.pop(idx)
    # Insert right first so indices stay correct when inserting left
    if right_piece and right_piece.length > 0:
        available.insert(idx, right_piece)
    if left_piece and left_piece.length > 0:
        available.insert(idx, left_piece)


def _allocate_chunks(
    chunks: List[_TaskChunk],
    free_slots: List[_Interval],
    adaptive: AdaptiveModelContext,
    target_date: str,
    pinned_tasks: Optional[List[StudyTask]] = None,
) -> Tuple[List[ScheduledBlock], List[str], List[str]]:
    """Reserve hard-pinned tasks first, then place flexible chunks cleanly.

    Returns (scheduled_blocks, unscheduled_task_ids, pinned_conflict_ids).
    """
    peak = _parse_window(adaptive.peak_productivity_window)
    secondary = _parse_window(adaptive.secondary_productivity_window)

    # Mutable copy of free slots (will be consumed as we allocate)
    available = [_Interval(s.start, s.end) for s in free_slots]
    scheduled: List[ScheduledBlock] = []
    unscheduled_ids: List[str] = []
    pinned_conflict_ids: List[str] = []
    daily_study_mins = 0
    max_daily = adaptive.max_daily_study_mins

    # --- Reserve pinned tasks as hard time constraints first ---
    for task in pinned_tasks or []:
        if not task.pinned_start:
            continue
        try:
            p_start = _hhmm_to_mins(task.pinned_start)
            if task.pinned_end:
                p_end = _hhmm_to_mins(task.pinned_end)
            else:
                dur = task.estimated_mins
                if adaptive.current_system_mode == "PREEMPTIVE_RESTRUCTURE":
                    dur = int(dur * adaptive.task_delay_factor)
                p_end = p_start + dur
        except ValueError:
            unscheduled_ids.append(task.task_id)
            pinned_conflict_ids.append(task.task_id)
            continue

        p_dur = p_end - p_start
        if p_dur <= 0 or daily_study_mins + p_dur > max_daily:
            unscheduled_ids.append(task.task_id)
            pinned_conflict_ids.append(task.task_id)
            continue

        placed_idx: Optional[int] = None
        for i, slot in enumerate(available):
            if slot.start <= p_start and p_end <= slot.end:
                placed_idx = i
                break

        if placed_idx is None:
            unscheduled_ids.append(task.task_id)
            pinned_conflict_ids.append(task.task_id)
            continue

        scheduled.append(
            ScheduledBlock(
                task_id=task.task_id,
                title=task.title,
                date=target_date,
                start=_mins_to_hhmm(p_start),
                end=_mins_to_hhmm(p_end),
                part=1,
            )
        )
        daily_study_mins += p_dur
        _consume_slot(available, placed_idx, p_start, p_end)

    def _find_best_slot(chunk: _TaskChunk) -> Optional[Tuple[int, int]]:
        """Return (index_of_slot, start_minute) for the best placement, or None.

        The algorithm tries to place the block's start inside the preferred
        cognitive window.  If the preferred window is contained within a
        larger free slot the block is shifted forward so it starts at the
        window boundary rather than at the slot boundary.
        """
        preferred_window = peak if chunk.cognitive_load == "high" else secondary
        best: Optional[Tuple[int, int]] = None  # (slot_idx, block_start)
        best_overlap = -1
        best_start = float("inf")

        for i, slot in enumerate(available):
            if slot.length < chunk.duration_mins:
                continue

            # Candidate start: beginning of the slot
            candidate_start = slot.start

            # Try to shift the block start into the preferred window if the
            # slot is large enough.
            window_start_in_slot = max(slot.start, preferred_window.start)
            window_candidate_end = window_start_in_slot + chunk.duration_mins
            if window_candidate_end <= slot.end:
                candidate_start = window_start_in_slot

            candidate_end = candidate_start + chunk.duration_mins
            candidate_interval = _Interval(candidate_start, candidate_end)
            overlap = _overlaps(candidate_interval, preferred_window)

            if (overlap > best_overlap) or (
                overlap == best_overlap and candidate_start < best_start
            ):
                best = (i, candidate_start)
                best_overlap = overlap
                best_start = candidate_start
        return best

    for chunk in chunks:
        if daily_study_mins + chunk.duration_mins > max_daily:
            unscheduled_ids.append(chunk.task_id)
            continue

        result = _find_best_slot(chunk)
        if result is None:
            unscheduled_ids.append(chunk.task_id)
            continue

        idx, block_start = result
        block_end = block_start + chunk.duration_mins

        scheduled.append(
            ScheduledBlock(
                task_id=chunk.task_id,
                title=chunk.title,
                date=target_date,
                start=_mins_to_hhmm(block_start),
                end=_mins_to_hhmm(block_end),
                part=chunk.part,
            )
        )

        daily_study_mins += chunk.duration_mins
        _consume_slot(available, idx, block_start, block_end)

    # Sort chronologically by start time ascending
    scheduled.sort(key=lambda b: (_hhmm_to_mins(b.start), _hhmm_to_mins(b.end), b.task_id, b.part))

    # De-duplicate unscheduled (a multi-chunk task may appear multiple times)
    seen = set()
    unique_unscheduled: List[str] = []
    for tid in unscheduled_ids:
        if tid not in seen:
            seen.add(tid)
            unique_unscheduled.append(tid)

    return scheduled, unique_unscheduled, pinned_conflict_ids


# ---------------------------------------------------------------------------
# Timezone & Deadline helpers
# ---------------------------------------------------------------------------

def _get_configured_tz() -> ZoneInfo | timezone:
    """Return configured ZoneInfo (default Africa/Cairo) with UTC fallback."""
    tz_name = getattr(config, "TIMEZONE", "Africa/Cairo")
    try:
        return ZoneInfo(tz_name)
    except Exception:
        return timezone.utc


def _parse_deadline_dt(deadline_str: str, tz: ZoneInfo | timezone) -> Optional[datetime]:
    """Parse date-only ('YYYY-MM-DD') or ISO deadline string into timezone-aware datetime.

    Date-only deadlines map to 23:59:59 on that date in *tz* to avoid naive vs aware
    comparison errors and treat the full day as valid.
    """
    raw = (deadline_str or "").strip()
    if not raw:
        return None
    try:
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", raw):
            return datetime.fromisoformat(f"{raw}T23:59:59").replace(tzinfo=tz)
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            return dt.replace(tzinfo=tz)
        return dt.astimezone(tz)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def optimize_schedule(request: ScheduleRequest) -> ScheduleResult:
    """Build a conflict-free daily study schedule.

    This is the single entry point for the deterministic optimizer.
    """
    adaptive = request.adaptive_model
    target_date = request.target_date

    # --- Step 1: Constraint Ingestion ---
    # Start with full operating window
    free_slots = [_Interval(_DAY_START, _DAY_END)]

    # Subtract university lectures on the target date
    lecture_blocks: List[_Interval] = []
    for lec in request.university_schedule:
        if lec.date == target_date:
            lecture_blocks.append(
                _Interval(_hhmm_to_mins(lec.start), _hhmm_to_mins(lec.end))
            )

    # Subtract external Google Calendar busy blocks
    gcal_blocks: List[_Interval] = []
    for blk in request.gcal_busy_blocks:
        gcal_blocks.append(
            _Interval(_hhmm_to_mins(blk["start"]), _hhmm_to_mins(blk["end"]))
        )

    # Subtract already-scheduled study tasks on target_date (with breaks)
    scheduled_task_blocks: List[_Interval] = []
    for t in request.tasks:
        if t.status == "scheduled" and t.scheduled_start and t.scheduled_end:
            if t.scheduled_date is None or t.scheduled_date == target_date:
                try:
                    s_min = _hhmm_to_mins(t.scheduled_start)
                    e_min = _hhmm_to_mins(t.scheduled_end)
                    if e_min > s_min:
                        scheduled_task_blocks.append(
                            _Interval(
                                max(_DAY_START, s_min - _BREAK_MINS),
                                min(_DAY_END, e_min + _BREAK_MINS),
                            )
                        )
                except ValueError:
                    pass

    all_busy = sorted(
        lecture_blocks + gcal_blocks + scheduled_task_blocks,
        key=lambda b: b.start,
    )
    free_slots = _subtract_intervals(free_slots, all_busy)

    # --- Steps 2 & 3: Prepare task chunks (flexible) & hard-pinned tasks ---
    pending_tasks = [t for t in request.tasks if t.status == "pending"]
    pinned_tasks = [t for t in pending_tasks if t.pinned_start]
    flex_tasks = [t for t in pending_tasks if not t.pinned_start]
    chunks = _prepare_tasks(flex_tasks, adaptive)

    # --- Step 4: Allocate ---
    scheduled_blocks, unscheduled_ids, pinned_conflict_ids = _allocate_chunks(
        chunks, free_slots, adaptive, target_date, pinned_tasks=pinned_tasks
    )

    # --- Step 5: Feasibility Verification ---
    tz = _get_configured_tz()
    blocks_by_task: Dict[str, List[ScheduledBlock]] = {}
    for blk in scheduled_blocks:
        blocks_by_task.setdefault(blk.task_id, []).append(blk)

    # Check deadline violations for BOTH unscheduled AND scheduled tasks
    deadline_violations: List[dict] = []
    for task in pending_tasks:
        if not task.deadline:
            continue
        if task.task_id in unscheduled_ids:
            deadline_violations.append(
                {
                    "task_id": task.task_id,
                    "title": task.title,
                    "deadline": task.deadline,
                    "reason": "Could not schedule before deadline – insufficient free slots or daily study cap reached.",
                }
            )
            continue

        deadline_dt = _parse_deadline_dt(task.deadline, tz)
        if deadline_dt is None:
            continue

        late_blocks = []
        for blk in blocks_by_task.get(task.task_id, []):
            end_dt = datetime.fromisoformat(f"{blk.date}T{blk.end}:00").replace(tzinfo=tz)
            if end_dt > deadline_dt:
                late_blocks.append(blk)

        if late_blocks:
            latest_blk = max(late_blocks, key=lambda b: (b.date, b.end))
            deadline_violations.append(
                {
                    "task_id": task.task_id,
                    "title": task.title,
                    "deadline": task.deadline,
                    "scheduled_date": latest_blk.date,
                    "scheduled_end": latest_blk.end,
                    "reason": (
                        f"Scheduled block ends at {latest_blk.date} {latest_blk.end}, "
                        f"exceeding task deadline {task.deadline}."
                    ),
                }
            )

    is_feasible = (len(unscheduled_ids) == 0) and (len(deadline_violations) == 0)

    feasibility_flags = None
    if not is_feasible:
        feasibility_flags = {
            "is_feasible": False,
            "unscheduled_task_ids": unscheduled_ids,
            "deadline_violations": deadline_violations,
            "pinned_conflicts": pinned_conflict_ids,
            "message": (
                "Schedule is infeasible: some tasks could not be placed or finish after their deadline. "
                "Consider extending deadlines, resolving pinned-slot conflicts, "
                "reducing workload, or increasing max_daily_study_mins."
            ),
        }

    return ScheduleResult(
        is_feasible=is_feasible,
        scheduled_blocks=scheduled_blocks,
        unscheduled_tasks=unscheduled_ids,
        feasibility_flags=feasibility_flags,
    )
