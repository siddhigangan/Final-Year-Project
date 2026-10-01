"""Utility impact: does the defense flag ordinary, never-poisoned code?

Blueprint section 60, research question 5. The existing false-positive
number on the dashboard (defense_eval.evaluate_defense) only covers the
5 clean seed snippets used as controls for the poisoning tests -- a
small, hand-written sample. This module runs the same real defense
pipeline against a much larger pool of real, unmodified functions
pulled from an actual codebase (reusing retrieval_eval.collect_distractors,
which already does this for the retrieval-stage experiment), to check
whether the defense's near-zero false-positive rate holds up against
code it has never been tuned against.

``DefensePipeline.analyze_context`` requires a non-empty query string
(unlike ``ContextBuilder.build``, which accepts ``query=None``). Since a
real distractor function wasn't retrieved for any specific task, this
module does not invent a plausible-looking, task-specific query for
each one -- that would misrepresent what the function is. Instead every
chunk is screened against the same single, clearly-labeled placeholder
query (``UTILITY_PROBE_QUERY``), documented here as exactly that: a
generic probe, not a real retrieval query.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.defense.pipeline import DefensePipeline, DefensePipelineConfig
from src.models import CodeChunk, RetrievedChunk
from src.retrieval.context_builder import ContextBuilder


class UtilityEvalError(RuntimeError):
    """Raised when utility-impact evaluation cannot proceed."""


UTILITY_PROBE_QUERY = "utility-impact probe (not a real retrieval query)"
"""Placeholder query used for every chunk in this module.

DefensePipeline.analyze_context requires a non-empty query string. A
real distractor function has no genuine retrieval query behind it, so
rather than fabricate a plausible-but-fake one per function, every
chunk is screened with this single, honestly-labeled placeholder.
"""


@dataclass(frozen=True)
class UtilityOutcome:
    """Whether the defense blocked one real, never-poisoned chunk."""

    chunk_id: str
    source_file_id: str
    blocked: bool
    finding_count: int


def default_pipeline() -> DefensePipeline:
    """Build the production defense pipeline with default config."""
    return DefensePipeline(config=DefensePipelineConfig(enabled=True))


def screen_clean_chunk(
    chunk: CodeChunk,
    pipeline: DefensePipeline,
) -> UtilityOutcome:
    """Run one real, unmodified chunk through the defense pipeline.

    Screened against ``UTILITY_PROBE_QUERY``, a labeled placeholder,
    since analyze_context requires a non-empty query and no real query
    exists for a distractor function taken out of context.
    """
    retrieved = RetrievedChunk(chunk=chunk, retrieval_score=1.0, rank=1)
    context = ContextBuilder().build([retrieved], query=UTILITY_PROBE_QUERY)

    result = pipeline.analyze_context(
        query=UTILITY_PROBE_QUERY,
        context=context,
        retrieved_chunks=[retrieved],
    )

    finding_count = (
        len(result.decision_report.findings)
        if result.decision_report is not None
        else 0
    )

    return UtilityOutcome(
        chunk_id=chunk.chunk_id,
        source_file_id=chunk.source_file_id,
        blocked=result.blocked,
        finding_count=finding_count,
    )


def evaluate_utility_impact(
    distractors: list[CodeChunk],
    pipeline: DefensePipeline | None = None,
) -> dict[str, float | int | list[str]]:
    """Measure the defense's false-positive rate on real, unposioned code.

    Args:
        distractors: Real, never-poisoned code chunks (e.g. from
            ``retrieval_eval.collect_distractors``).
        pipeline: Defense pipeline to use. Defaults to the production
            configuration.

    Returns:
        A summary with the overall false-positive rate and the
        chunk_ids of any chunks that were incorrectly blocked, so a
        specific false positive can be inspected rather than only
        counted.

    Raises:
        UtilityEvalError: If ``distractors`` is empty.
    """
    if not distractors:
        raise UtilityEvalError(
            "distractors cannot be empty; nothing to measure utility "
            "impact against."
        )

    pipe = pipeline or default_pipeline()
    outcomes = [screen_clean_chunk(c, pipe) for c in distractors]
    blocked = [o for o in outcomes if o.blocked]

    return {
        "sample_count": len(outcomes),
        "false_positive_rate": len(blocked) / len(outcomes),
        "false_positive_count": len(blocked),
        "false_positive_chunk_ids": [o.chunk_id for o in blocked],
        "mean_finding_count": (
            sum(o.finding_count for o in outcomes) / len(outcomes)
        ),
    }
