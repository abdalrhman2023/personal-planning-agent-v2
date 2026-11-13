"""Telemetry Engine – computes workload metrics, velocity slip, and study-block invitations.

All computations are pure Python / deterministic.  No LLM calls.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

from app import config
from app.schema import (
    FixedLecture,
    OutboundSignals,
    ScheduledBlock,
    StudyTask,
)
from app.engine.optimizer import _hhmm_to_mins, _DAY_START, _DAY_END


# ---------------------------------------------------------------------------
# 1. Workload Density Metrics
# ---------------------------------------------------------------------------

def _merge_intervals(intervals: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Merge overlapping or adjacent [start, end) minute intervals."""
    if not intervals:
        return []
    sorted_iv = sorted((s, e) for s, e in intervals if e > s)
    if not sorted_iv:
        return []
    merged: List[Tuple[int, int]] = [sorted_iv[0]]
    for s, e in sorted_iv[1:]:
        prev_s, prev_e = merged[-1]
        if s <= prev_e:
            merged[-1] = (prev_s, max(prev_e, e))
        else:
            merged.append((s, e))
    return merged


def _clip_to_operating_window(start_m: int, end_m: int) -> Optional[Tuple[int, int]]:
    """Clip a minute interval to [_DAY_START, _DAY_END] (08:00-22:00)."""
    s = max(_DAY_START, start_m)
    e = min(_DAY_END, end_m)
    return (s, e) if e > s else None


def _extract_gcal_interval_for_date(
    blk: Dict[str, str],
    target_date: str,
    tz: ZoneInfo | timezone,
) -> Optional[Tuple[int, int]]:
    """Return clipped [start, end) minutes on target_date for a GCal busy block, or None."""
    start_str = (blk.get("start") or "").strip()
    end_str = (blk.get("end") or "").strip()
    if not start_str or not end_str:
        return None

    # Full ISO datetime strings (may span dates or timezones)
    if "T" in start_str or "T" in end_str:
        try:
            s_dt = datetime.fromisoformat(start_str.replace("Z", "+00:00"))
            e_dt = datetime.fromisoformat(end_str.replace("Z", "+00:00"))
            if s_dt.tzinfo is None:
                s_dt = s_dt.replace(tzinfo=tz)
            else:
                s_dt = s_dt.astimezone(tz)
            if e_dt.tzinfo is None:
                e_dt = e_dt.replace(tzinfo=tz)
            else:
                e_dt = e_dt.astimezone(tz)

            op_start = datetime.fromisoformat(f"{target_date}T08:00:00").replace(tzinfo=tz)
            op_end = datetime.fromisoformat(f"{target_date}T22:00:00").replace(tzinfo=tz)
            clipped_s = max(s_dt, op_start)
            clipped_e = min(e_dt, op_end)
            if clipped_e <= clipped_s:
                return None
            day_midnight = datetime.fromisoformat(f"{target_date}T00:00:00").replace(tzinfo=tz)
            return (
                int((clipped_s - day_midnight).total_seconds() // 60),
                int((clipped_e - day_midnight).total_seconds() // 60),
            )
        except ValueError:
            return None

    # HH:MM format with optional date field
    blk_date = blk.get("date")
    if blk_date and blk_date != target_date:
        return None
    try:
        return _clip_to_operating_window(_hhmm_to_mins(start_str), _hhmm_to_mins(end_str))
    except ValueError:
        return None


def compute_workload_density(
    scheduled_blocks: List[ScheduledBlock],
    university_schedule: List[FixedLecture],
    gcal_busy_blocks: List[Dict[str, str]],
    target_date: str,
) -> Dict[str, float]:
    """Compute workload density = scheduled_study_mins / total_free_mins.

    Filters fixed lectures and GCal busy blocks to *target_date*, merges all
    overlapping intervals (preventing double-counting), and clips intervals to
    operating hours (08:00-22:00).
    """
    total_operating_mins = _DAY_END - _DAY_START  # 840
    tz_name = getattr(config, "TIMEZONE", "Africa/Cairo")
    try:
        tz: ZoneInfo | timezone = ZoneInfo(tz_name)
    except Exception:
        tz = timezone.utc

    lecture_intervals: List[Tuple[int, int]] = []
    for lec in university_schedule:
        if lec.date != target_date:
            continue
        try:
            clipped = _clip_to_operating_window(_hhmm_to_mins(lec.start), _hhmm_to_mins(lec.end))
            if clipped:
                lecture_intervals.append(clipped)
        except ValueError:
            continue

    gcal_intervals: List[Tuple[int, int]] = []
    for blk in gcal_busy_blocks:
        iv = _extract_gcal_interval_for_date(blk, target_date, tz)
        if iv:
            gcal_intervals.append(iv)

    merged_lectures = _merge_intervals(lecture_intervals)
    merged_gcal = _merge_intervals(gcal_intervals)
    all_merged_busy = _merge_intervals(merged_lectures + merged_gcal)

    lecture_mins = sum(e - s for s, e in merged_lectures)
    gcal_mins = sum(e - s for s, e in merged_gcal)
    total_busy_mins = sum(e - s for s, e in all_merged_busy)

    total_free_mins = max(1, total_operating_mins - total_busy_mins)

    scheduled_study_mins = sum(
        _hhmm_to_mins(b.end) - _hhmm_to_mins(b.start)
        for b in scheduled_blocks
        if b.date == target_date
    )

    density = round(scheduled_study_mins / total_free_mins, 4)

    return {
        "target_date": target_date,
        "total_operating_mins": total_operating_mins,
        "lecture_mins": lecture_mins,
        "gcal_busy_mins": gcal_mins,
        "total_free_mins": total_free_mins,
        "scheduled_study_mins": scheduled_study_mins,
        "workload_density": density,
    }


# ---------------------------------------------------------------------------
# 2. Velocity Slip Detection
# ---------------------------------------------------------------------------

_VELOCITY_SLIP_THRESHOLD = 0.20  # 20%


def _resolve_now(now: Optional[datetime]) -> datetime:
    """Return a timezone-aware datetime normalized to the configured TIMEZONE."""
    tz_name = getattr(config, "TIMEZONE", "Africa/Cairo")
    try:
        tz = ZoneInfo(tz_name)
    except Exception:
        tz = timezone.utc

    if now is None:
        return datetime.now(tz)
    if now.tzinfo is None:
        return now.replace(tzinfo=tz)
    return now.astimezone(tz)


def _has_scheduled_task_ended(task: StudyTask, ref_now: datetime) -> bool:
    """Return True only when a scheduled task's end date/time has passed."""
    if not task.scheduled_date:
        return False

    today_str = ref_now.strftime("%Y-%m-%d")
    if task.scheduled_date < today_str:
        return True
    if task.scheduled_date > today_str:
        return False

    if not task.scheduled_end:
        return False

    try:
        end_dt = datetime.fromisoformat(f"{task.scheduled_date}T{task.scheduled_end}:00")
        if ref_now.tzinfo is not None:
            end_dt = end_dt.replace(tzinfo=ref_now.tzinfo)
        return end_dt <= ref_now
    except ValueError:
        return False


def compute_velocity_slip(
    tasks: List[StudyTask],
    now: Optional[datetime] = None,
) -> Dict:
    """Compare completed tasks vs missed or overdue-scheduled tasks.

    Returns a dict with ``slip_ratio``, ``deviation_exceeded`` flag, and,
    when the threshold is crossed, an ``alert_payload`` for the Adaptive
    Manager.

    Scheduled tasks are only counted as uncompleted/attempted once their
    scheduled end time has actually passed (``end <= now`` or on a past date).
    Future scheduled, pending, and delayed tasks are excluded.
    """
    ref_now = _resolve_now(now)
    completed = [t for t in tasks if t.status == "completed"]
    scheduled_or_missed = [
        t
        for t in tasks
        if t.status in ("completed", "missed")
        or (t.status == "scheduled" and _has_scheduled_task_ended(t, ref_now))
    ]

    total_attempted = len(scheduled_or_missed)
    if total_attempted == 0:
        return {
            "completed_count": 0,
            "attempted_count": 0,
            "completion_rate": 1.0,
            "expected_rate": 1.0,
            "slip_ratio": 0.0,
            "deviation_exceeded": False,
            "alert_payload": None,
        }

    completion_rate = len(completed) / total_attempted
    expected_rate = 1.0  # we expect all scheduled tasks to be completed
    slip_ratio = round(abs(expected_rate - completion_rate), 4)
    deviation_exceeded = slip_ratio > _VELOCITY_SLIP_THRESHOLD

    alert_payload = None
    if deviation_exceeded:
        alert_payload = {
            "signal_type": "VELOCITY_SLIP_ALERT",
            "completed_count": len(completed),
            "attempted_count": total_attempted,
            "completion_rate": round(completion_rate, 4),
            "slip_ratio": slip_ratio,
            "message": (
                f"Velocity slip detected: completion rate "
                f"{completion_rate:.0%} deviates from expected by "
                f"{slip_ratio:.0%} (threshold {_VELOCITY_SLIP_THRESHOLD:.0%})."
            ),
        }

    return {
        "completed_count": len(completed),
        "attempted_count": total_attempted,
        "completion_rate": round(completion_rate, 4),
        "expected_rate": expected_rate,
        "slip_ratio": slip_ratio,
        "deviation_exceeded": deviation_exceeded,
        "alert_payload": alert_payload,
    }


# ---------------------------------------------------------------------------
# 3. Study Block Invitations for AI Tutor Agent
# ---------------------------------------------------------------------------

def generate_study_block_invitations(
    scheduled_blocks: List[ScheduledBlock],
    tasks: List[StudyTask],
) -> List[dict]:
    """Generate Study Block Invitations & Context Payload for AI Tutor Agent.

    Each invitation carries enough context for the Tutor to prepare the
    right material when the study session starts.
    """
    task_lookup: Dict[str, StudyTask] = {t.task_id: t for t in tasks}
    invitations: List[dict] = []

    for block in scheduled_blocks:
        task = task_lookup.get(block.task_id)
        if task is None:
            continue

        invitations.append({
            "block_date": block.date,
            "block_start": block.start,
            "block_end": block.end,
            "block_part": block.part,
            "task_id": task.task_id,
            "task_title": task.title,
            "topic_ref": task.topic_ref,
            "source": task.source,
            "cognitive_load": task.cognitive_load,
            "parent_milestone_id": task.parent_milestone_id,
        })

    return invitations


# ---------------------------------------------------------------------------
# 4. Aggregate – produce full OutboundSignals
# ---------------------------------------------------------------------------

def compute_outbound_signals(
    scheduled_blocks: List[ScheduledBlock],
    tasks: List[StudyTask],
    university_schedule: List[FixedLecture],
    gcal_busy_blocks: List[Dict[str, str]],
    target_date: str,
    feasibility_flags: Optional[dict] = None,
    now: Optional[datetime] = None,
) -> OutboundSignals:
    """One-call convenience that assembles all outbound signals."""

    # Workload density
    density = compute_workload_density(
        scheduled_blocks, university_schedule, gcal_busy_blocks, target_date,
    )

    # Velocity slip
    slip = compute_velocity_slip(tasks, now=now)

    # Adaptive Manager signals: evaluate feasibility and velocity slip independently
    to_adaptive: Optional[dict] = None
    if feasibility_flags:
        to_adaptive = dict(feasibility_flags)
    if slip["deviation_exceeded"]:
        if to_adaptive is None:
            to_adaptive = slip["alert_payload"]
        else:
            to_adaptive["velocity_slip_alert"] = slip["alert_payload"]
            to_adaptive["active_signals"] = [
                to_adaptive.get("signal_type", "FEASIBILITY_INFEASIBLE"),
                "VELOCITY_SLIP_ALERT",
            ]

    # Life Coach: send feasibility report if schedule is infeasible
    to_life_coach: Optional[dict] = None
    if feasibility_flags:
        to_life_coach = {
            "signal_type": "MILESTONE_FEASIBILITY_REPORT",
            "details": feasibility_flags,
        }

    # AI Tutor invitations
    to_tutor = generate_study_block_invitations(scheduled_blocks, tasks)

    return OutboundSignals(
        to_adaptive_manager=to_adaptive,
        to_life_coach=to_life_coach,
        to_ai_tutor=to_tutor,
        workload_density_metrics=density,
    )
