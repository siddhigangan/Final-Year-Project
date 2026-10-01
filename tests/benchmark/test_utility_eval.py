"""Tests for the utility-impact evaluation (blueprint sec. 60, Q5)."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.benchmark.retrieval_eval import collect_distractors
from src.benchmark.utility_eval import (
    UTILITY_PROBE_QUERY,
    UtilityEvalError,
    default_pipeline,
    evaluate_utility_impact,
    screen_clean_chunk,
)
from src.models import CodeChunk, ProgrammingLanguage


def make_chunk(content: str, chunk_id: str = "c1") -> CodeChunk:
    return CodeChunk(
        chunk_id=chunk_id,
        source_file_id=f"{chunk_id}-file",
        repository_id="utility-test-repository",
        content=content,
        language=ProgrammingLanguage.PYTHON,
        start_line=1,
        end_line=len(content.splitlines()) or 1,
    )


def test_empty_distractor_list_is_rejected() -> None:
    with pytest.raises(UtilityEvalError):
        evaluate_utility_impact([])


def test_clean_function_is_not_blocked() -> None:
    chunk = make_chunk("def add(a, b):\n    return a + b\n")

    outcome = screen_clean_chunk(chunk, default_pipeline())

    assert outcome.blocked is False
    assert outcome.chunk_id == "c1"


def test_probe_query_is_clearly_labeled_not_a_real_query() -> None:
    assert "not a real retrieval query" in UTILITY_PROBE_QUERY


def test_evaluate_reports_zero_false_positives_on_clean_chunks() -> None:
    chunks = [
        make_chunk("def add(a, b):\n    return a + b\n", "c1"),
        make_chunk("def greet(name):\n    return f'hello {name}'\n", "c2"),
    ]

    report = evaluate_utility_impact(chunks)

    assert report["sample_count"] == 2
    assert report["false_positive_rate"] == 0.0
    assert report["false_positive_chunk_ids"] == []


def test_real_repository_functions_have_a_low_false_positive_rate() -> None:
    # Regression guard against this repo's own src/ as a real-world
    # sample, not synthetic data.
    distractors = collect_distractors(Path("src"), limit=40)

    report = evaluate_utility_impact(distractors)

    assert report["sample_count"] > 0
    assert report["false_positive_rate"] <= 0.1
    assert isinstance(report["false_positive_chunk_ids"], list)


def test_hardcoded_secret_style_clean_code_can_still_trip_static_analysis() -> None:
    # Sanity check that the pipeline is actually doing something, not
    # silently returning blocked=False unconditionally. A real secret
    # pattern is flagged even outside the poisoning tests, confirming
    # the 0% rate above reflects genuinely clean code, not a pipeline
    # that never blocks anything.
    chunk = make_chunk(
        "API_KEY = 'AKIAABCDEFGHIJKLMNOP'\n"
        "def get_key():\n    return API_KEY\n"
    )

    outcome = screen_clean_chunk(chunk, default_pipeline())

    assert outcome.blocked is True
    assert outcome.finding_count >= 1
