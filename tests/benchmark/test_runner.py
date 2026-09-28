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


def test_defended_evaluator_can_differ_from_poisoned_evaluator():
    # The undefended evaluator always reports failure once poisoned.
    # The defended evaluator simulates a defense that neutralizes the
    # poisoning, always reporting success. Both are called with
    # identical (sample, sample.poisoned_code, True) arguments, so
    # this can only be distinguished by using two separate evaluator
    # callables, not by branching on the arguments alone.
    runner = BenchmarkRunner(
        evaluator=lambda sample, code, poisoned: not poisoned,
        defended_evaluator=lambda sample, code, poisoned: True,
        detector=lambda sample: True,
    )

    result = runner.run(make_dataset())

    assert result.metrics.poisoned_success_rate == 0.0
    assert result.metrics.defended_success_rate == 1.0


def test_defended_evaluator_defaults_to_evaluator_when_omitted():
    calls = []

    def evaluator(sample, code, poisoned):
        calls.append(poisoned)
        return not poisoned

    runner = BenchmarkRunner(
        evaluator=evaluator,
        detector=lambda sample: True,
    )

    runner.run(make_dataset())

    # Two samples, each evaluated for clean, poisoned, and defended:
    # 2 * 3 = 6 total evaluator calls, all through the same callable.
    assert len(calls) == 6

