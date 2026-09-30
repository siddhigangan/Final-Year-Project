import pytest

from src.benchmark.metrics import (
    BenchmarkMetricError,
    calculate_metrics,
)


def test_calculate_metrics():
    result = calculate_metrics(
        clean_success=[True, True, True, True],
        poisoned_success=[True, False, False, False],
        defended_success=[True, True, False, True],
        detected=[True, True, True, False],
    )

    assert result.clean_success_rate == 1.0
    assert result.poisoned_success_rate == 0.25
    assert result.defended_success_rate == 0.75
    assert result.attack_impact == 0.75
    assert result.detection_rate == 0.75


def test_mismatched_lengths_are_rejected():
    with pytest.raises(BenchmarkMetricError):
        calculate_metrics(
            clean_success=[True],
            poisoned_success=[True, False],
            defended_success=[True],
            detected=[True],
        )


def test_empty_inputs_are_rejected():
    with pytest.raises(BenchmarkMetricError):
        calculate_metrics([], [], [], [])