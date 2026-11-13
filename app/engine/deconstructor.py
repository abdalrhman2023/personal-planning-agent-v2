"""LLM Task Deconstructor – breaks large milestones into micro-tasks.

Uses ChatGoogleGenerativeAI with structured outputs when GEMINI_API_KEY is
available.  Falls back to a deterministic splitter for offline / CI testing.
"""

from __future__ import annotations

import logging
import math
from typing import List, Optional

from pydantic import BaseModel, Field

from app import config
from app.schema import StudyTask

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Structured output model for LLM response
# ---------------------------------------------------------------------------

class MicroTask(BaseModel):
    """A single micro-task produced by deconstruction."""
    title: str
    estimated_mins: int
    cognitive_load: str = "medium"  # "high", "medium", "low"
    topic_ref: Optional[str] = None


class DeconstructionResult(BaseModel):
    """Result of deconstructing a parent task / milestone."""
    parent_task_id: str
    parent_title: str
    micro_tasks: List[MicroTask] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Deterministic fallback splitter
# ---------------------------------------------------------------------------

def _deterministic_split(
    task: StudyTask,
    focus_duration_mins: int,
) -> DeconstructionResult:
    """Split a task into equal-ish chunks of *focus_duration_mins* each.

    This is the offline / CI fallback when no Gemini API key is configured.
    """
    total_mins = task.estimated_mins
    num_chunks = max(1, math.ceil(total_mins / focus_duration_mins))
    chunk_mins = total_mins // num_chunks
    remainder = total_mins % num_chunks

    micro_tasks: List[MicroTask] = []
    for i in range(num_chunks):
        # Spread the remainder across the first chunks
        extra = 1 if i < remainder else 0
        micro_tasks.append(
            MicroTask(
                title=f"{task.title} – Part {i + 1}/{num_chunks}",
                estimated_mins=chunk_mins + extra,
                cognitive_load=task.cognitive_load,
                topic_ref=task.topic_ref,
            )
        )

    return DeconstructionResult(
        parent_task_id=task.task_id,
        parent_title=task.title,
        micro_tasks=micro_tasks,
    )


# ---------------------------------------------------------------------------
# LLM-powered deconstructor
# ---------------------------------------------------------------------------

def _llm_deconstruct(
    task: StudyTask,
    focus_duration_mins: int,
) -> DeconstructionResult:
    """Use ChatGoogleGenerativeAI to break a task into meaningful micro-tasks."""
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.messages import HumanMessage, SystemMessage

    if not config.can_invoke_llm(ChatGoogleGenerativeAI):
        raise RuntimeError("LLM invocation disabled (missing API key or unmocked test run)")

    system_prompt = (
        "You are a study planning assistant. Your job is to break a large "
        "study task into smaller, actionable micro-tasks. Each micro-task "
        "must fit within a single focus session.\n\n"
        "Rules:\n"
        f"- Each micro-task must be at most {focus_duration_mins} minutes.\n"
        f"- The total estimated minutes across all micro-tasks must equal "
        f"the original task's estimated_mins ({task.estimated_mins} mins).\n"
        "- Assign a cognitive_load ('high', 'medium', or 'low') to each.\n"
        "- Preserve the topic_ref from the original task.\n"
        f"- Set parent_task_id to '{task.task_id}'.\n"
        f"- Set parent_title to '{task.title}'.\n"
    )

    human_prompt = (
        f"Deconstruct this study task into micro-tasks:\n"
        f"  Title: {task.title}\n"
        f"  Estimated minutes: {task.estimated_mins}\n"
        f"  Cognitive load: {task.cognitive_load}\n"
        f"  Topic ref: {task.topic_ref or 'N/A'}\n"
        f"  Focus session limit: {focus_duration_mins} minutes\n"
    )

    candidate_models = config.get_gemini_candidate_models()
    last_exc: Optional[Exception] = None
    for candidate_model in candidate_models:
        try:
            llm = ChatGoogleGenerativeAI(
                model=candidate_model,
                google_api_key=config.GEMINI_API_KEY,
                timeout=15,
                max_retries=1,
            )
            structured_llm = llm.with_structured_output(DeconstructionResult)
            result: DeconstructionResult = structured_llm.invoke(
                [SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)]
            )
            # Ensure parent fields are set correctly regardless of LLM output
            result.parent_task_id = task.task_id
            result.parent_title = task.title
            return result
        except Exception as exc:
            last_exc = exc
            logger.warning("LLM deconstruction failed for model %s: %s", candidate_model, exc)
            continue

    raise RuntimeError(f"All Gemini models failed during deconstruction: {last_exc}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def deconstruct_task(
    task: StudyTask,
    focus_duration_mins: int,
    force_fallback: bool = False,
) -> DeconstructionResult:
    """Deconstruct *task* into micro-tasks of at most *focus_duration_mins*.

    Uses the LLM when ``GEMINI_API_KEY`` is configured.  Falls back to the
    deterministic splitter when the key is missing or *force_fallback* is True.
    """
    use_llm = bool(config.GEMINI_API_KEY) and not force_fallback

    if use_llm:
        try:
            return _llm_deconstruct(task, focus_duration_mins)
        except Exception:
            # Graceful degradation – fall back to deterministic splitter
            return _deterministic_split(task, focus_duration_mins)

    return _deterministic_split(task, focus_duration_mins)


def deconstruct_tasks(
    tasks: List[StudyTask],
    focus_duration_mins: int,
    force_fallback: bool = False,
) -> List[DeconstructionResult]:
    """Batch-deconstruct multiple tasks."""
    return [
        deconstruct_task(t, focus_duration_mins, force_fallback=force_fallback)
        for t in tasks
    ]
