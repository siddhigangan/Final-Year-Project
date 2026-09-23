from src.benchmark.dataset import (
    BenchmarkDataset,
    BenchmarkSample,
)
from src.benchmark.runner import BenchmarkRunner


def make_dataset():
    return BenchmarkDataset(
        [
            BenchmarkSample(
                sample_id="sample-1",
                query="Find the secure implementation.",
                clean_code="clean implementation",
                poisoned_code="poisoned implementation",
                language="python",
                poisoning_category="misleading_code",
            ),
            BenchmarkSample(
                sample_id="sample-2",
                query="Find the secure implementation.",
                clean_code="clean implementation",
                poisoned_code="poisoned implementation",
                language="python",
                poisoning_category="vulnerable_code",
            ),
        ]
    )


def test_runner_executes_all_samples():
    runner = BenchmarkRunner(
        evaluator=lambda sample, code, poisoned: not poisoned,
        detector=lambda sample: True,
    )

    result = runner.run(make_dataset())

    assert result.sample_count == 2
    assert result.metrics.clean_success_rate == 1.0
    assert result.metrics.poisoned_success_rate == 0.0
    assert result.metrics.defended_success_rate == 0.0
    assert result.metrics.detection_rate == 1.0