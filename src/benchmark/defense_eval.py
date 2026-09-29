"""Measure the real L1-L5 defense pipeline on benchmark samples.

Deterministic and LLM-free: for each sample the clean and poisoned code
are screened as retrieved context. Reports per-category block rate on
poisoned code and false-positive rate on clean code.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.benchmark.dataset import BenchmarkDataset, BenchmarkSample
from src.defense.pipeline import DefensePipeline, DefensePipelineConfig
from src.models import CodeChunk, ProgrammingLanguage, RetrievedChunk
from src.retrieval.context_builder import ContextBuilder


@dataclass(frozen=True)
class ScreenResult:
    """Outcome of screening one context through the defense."""

    blocked: bool
    decision: str
    context_code: str


def default_pipeline() -> DefensePipeline:
    """Build the production defense pipeline with default config."""
    return DefensePipeline(config=DefensePipelineConfig(enabled=True))


def screen(
    sample: BenchmarkSample,
    code: str,
    pipeline: DefensePipeline,
) -> ScreenResult:
    """Run one code string through the defense as retrieved context.

    ``context_code`` is what the model would see after sanitization
    (the pipeline wraps content in data delimiters); it is the original
    code when the pipeline returns no sanitized context.
    """
    chunk = CodeChunk(
        chunk_id=f"{sample.sample_id}-screen",
        source_file_id=f"{sample.sample_id}-file",
        repository_id="benchmark-defense-repository",
        content=code,
        language=ProgrammingLanguage(sample.language),
        start_line=1,
        end_line=len(code.splitlines()) or 1,
    )
    retrieved = RetrievedChunk(chunk=chunk, retrieval_score=1.0, rank=1)
    context = ContextBuilder().build([retrieved], query=sample.query)

    result = pipeline.analyze_context(
        query=sample.query,
        context=context,
        retrieved_chunks=[retrieved],
    )

    safe = result.safe_context
    context_code = (
        "\n".join(src.content for src in safe.sources)
        if safe is not None and safe.sources
        else code
    )

    return ScreenResult(
        blocked=result.blocked,
        decision=result.final_decision.value,
        context_code=context_code,
    )


def evaluate_defense(
    dataset: BenchmarkDataset,
    pipeline: DefensePipeline | None = None,
) -> dict[str, dict[str, float | int]]:
    """Per-category block rate (poisoned) and false-positive rate (clean)."""
    pipe = pipeline or default_pipeline()
    buckets: dict[str, list[tuple[bool, bool]]] = {}

    for sample in dataset.samples:
        poisoned = screen(sample, sample.poisoned_code, pipe).blocked
        clean = screen(sample, sample.clean_code, pipe).blocked
        buckets.setdefault(sample.poisoning_category, []).append(
            (poisoned, clean)
        )

    return {
        category: {
            "samples": len(rows),
            "block_rate": sum(p for p, _ in rows) / len(rows),
            "false_positive_rate": sum(c for _, c in rows) / len(rows),
        }
        for category, rows in sorted(buckets.items())
    }
