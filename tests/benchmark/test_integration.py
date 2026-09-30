"""Tests for the benchmark integration adapter."""

from __future__ import annotations

import pytest

from src.benchmark.integration import (
    ALL_POISONING_STRATEGIES,
    BenchmarkIntegrationInputError,
    EchoGenerationProvider,
    SeedCodeSample,
    build_benchmark_runner,
    build_poisoning_dataset,
    is_measurable,
    static_analysis_flags,
    summarize_by_category,
)
from src.models import ProgrammingLanguage

SEED = SeedCodeSample(
    sample_id="s1",
    query="hash a password",
    task="code_completion",
    content="def hash_password(password):\n    return password\n",
    language=ProgrammingLanguage.PYTHON,
)


def test_dataset_covers_all_seven_categories() -> None:
    dataset = build_poisoning_dataset([SEED])

    assert dataset.size == 7
    assert len(ALL_POISONING_STRATEGIES) == 7
    assert len({s.poisoning_category for s in dataset.samples}) == 7


def test_empty_seed_samples_are_rejected() -> None:
    with pytest.raises(BenchmarkIntegrationInputError):
        build_poisoning_dataset([])


def test_poisoned_code_differs_from_clean_code() -> None:
    for sample in build_poisoning_dataset([SEED]).samples:
        assert sample.poisoned_code != sample.clean_code


def test_dataset_is_deterministic() -> None:
    first = build_poisoning_dataset([SEED])
    second = build_poisoning_dataset([SEED])

    assert [s.poisoned_code for s in first.samples] == [
        s.poisoned_code for s in second.samples
    ]


def test_oracle_flags_only_vulnerable_code() -> None:
    dataset = build_poisoning_dataset([SEED])

    for sample in dataset.samples:
        flagged = static_analysis_flags(
            sample.poisoned_code, sample.language
        )
        assert flagged == (
            sample.poisoning_category == "vulnerable_code"
        )

    assert static_analysis_flags(SEED.content, "python") is False


def test_echo_provider_needs_no_network() -> None:
    assert EchoGenerationProvider().provider_name == "benchmark-echo"


def test_runner_distinguishes_poisoned_from_defended() -> None:
    dataset = build_poisoning_dataset([SEED])
    result = build_benchmark_runner().run(dataset)

    by_id = {e.sample_id: e for e in result.evaluations}
    vulnerable = by_id["s1-vulnerable_code"]

    assert vulnerable.clean_success is True
    assert vulnerable.poisoned_success is False
    assert vulnerable.defended_success is True
    assert vulnerable.detected is True
    assert result.metrics.attack_impact > 0.0


def test_text_only_categories_are_not_reported_detected() -> None:
    dataset = build_poisoning_dataset([SEED])
    result = build_benchmark_runner().run(dataset)

    for evaluation in result.evaluations:
        if evaluation.sample_id != "s1-vulnerable_code":
            assert evaluation.detected is False


def test_summary_marks_unmeasured_categories() -> None:
    dataset = build_poisoning_dataset([SEED])
    result = build_benchmark_runner().run(dataset)
    summary = summarize_by_category(result, dataset)

    assert summary["vulnerable_code"]["measured"] is True
    assert summary["vulnerable_code"]["poisoned_success_rate"] == 0.0
    assert summary["vulnerable_code"]["defended_success_rate"] == 1.0

    for category, entry in summary.items():
        if category != "vulnerable_code":
            assert entry["measured"] is False
            assert "poisoned_success_rate" not in entry


def test_real_generator_makes_every_category_measurable() -> None:
    dataset = build_poisoning_dataset([SEED])

    assert all(
        is_measurable(s, real_generator=True) for s in dataset.samples
    )
