"""Unit tests for the CodeRAG benchmark orchestration layer."""

from __future__ import annotations

from typing import Any

import pytest

from src.experiments.benchmark import (
    Benchmark,
    BenchmarkCase,
    BenchmarkConfigurationError,
    BenchmarkInputError,
    BenchmarkRun,
)
from src.experiments.experiment_runner import (
    ExperimentRunInput,
    ExperimentRunner,
)
from src.models import (
    ExperimentCondition,
    ExperimentConfig,
    ExperimentResult,
    ProgrammingLanguage,
)


def make_config(
    condition: ExperimentCondition = (
        ExperimentCondition.CLEAN_NO_DEFENSE
    ),
    *,
    experiment_id: str = "experiment-1",
    top_k: int = 5,
    seed: int = 42,
) -> ExperimentConfig:
    """Create a valid experiment configuration for tests."""

    return ExperimentConfig(
        experiment_id=experiment_id,
        condition=condition,
        repository_id="test-repository",
        task_type="code_generation",
        language=ProgrammingLanguage.PYTHON,
        model_name="test-model",
        retriever_name="test-retriever",
        top_k=top_k,
        defense_layers=[],
        seed=seed,
        metadata={"source": "unit-test"},
    )


def make_input(
    condition: ExperimentCondition = (
        ExperimentCondition.CLEAN_NO_DEFENSE
    ),
    *,
    experiment_id: str = "experiment-1",
    query: str = "Generate a hello function.",
    task: str = "code_generation",
    request_id: str = "request-1",
) -> ExperimentRunInput:
    """Create a valid experiment input for tests."""

    return ExperimentRunInput(
        query=query,
        task=task,
        experiment_config=make_config(
            condition,
            experiment_id=experiment_id,
        ),
        request_id=request_id,
        baseline_generated_code=(
            "def hello():\n"
            "    return 'hello'\n"
        ),
        metadata={"source": "unit-test"},
    )


def make_result(
    experiment_id: str = "experiment-1",
    condition: ExperimentCondition = (
        ExperimentCondition.CLEAN_NO_DEFENSE
    ),
) -> ExperimentResult:
    """Create a valid experiment result for tests."""

    return ExperimentResult(
        experiment_id=experiment_id,
        condition=condition,
        request_id=f"request-{experiment_id}",
        poison_present=condition
        in {
            ExperimentCondition.POISONED_NO_DEFENSE,
            ExperimentCondition.POISONED_DEFENSE,
        },
        poison_retrieved=False,
        poison_in_context=False,
        generation_changed=False,
        vulnerability_introduced=False,
        poison_retrieval_rate=0.0,
        attack_success_rate=0.0,
        vulnerability_introduction_rate=0.0,
        utility_score=1.0,
        latency_ms=10.0,
        findings=[],
        metadata={"source": "unit-test"},
    )


def make_case(
    case_id: str = "case-1",
    *,
    experiment_id: str = "experiment-1",
    condition: ExperimentCondition = (
        ExperimentCondition.CLEAN_NO_DEFENSE
    ),
) -> BenchmarkCase:
    """Create a valid benchmark case."""

    return BenchmarkCase(
        case_id=case_id,
        experiment=make_input(
            condition,
            experiment_id=experiment_id,
            request_id=f"request-{experiment_id}",
        ),
        metadata={"dataset_split": "test"},
    )


class StubExperimentRunner(ExperimentRunner):
    """Test double satisfying the ExperimentRunner type contract."""

    def __init__(
        self,
        *,
        results: dict[str, ExperimentResult] | None = None,
        failures: dict[str, Exception] | None = None,
    ) -> None:
        self.results = results or {}
        self.failures = failures or {}
        self.calls: list[str] = []

    def run(
        self,
        experiment: ExperimentRunInput,
    ) -> ExperimentResult:
        """Return a configured result or raise a configured failure."""

        experiment_id = experiment.experiment_config.experiment_id
        self.calls.append(experiment_id)

        if experiment_id in self.failures:
            raise self.failures[experiment_id]

        result = self.results.get(experiment_id)

        if result is None:
            return make_result(
                experiment_id=experiment_id,
                condition=experiment.experiment_config.condition,
            )

        return result


class TestBenchmarkCase:
    """Tests for BenchmarkCase."""

    def test_valid_case(self) -> None:
        case = make_case()

        assert case.case_id == "case-1"
        assert case.experiment_id == "experiment-1"
        assert (
            case.condition
            == ExperimentCondition.CLEAN_NO_DEFENSE
        )
        assert case.seed == 42

    def test_empty_case_id_is_rejected(self) -> None:
        with pytest.raises(BenchmarkInputError):
            BenchmarkCase(
                case_id="",
                experiment=make_input(),
            )

    def test_invalid_experiment_is_rejected(self) -> None:
        with pytest.raises(BenchmarkInputError):
            BenchmarkCase(
                case_id="case-1",
                experiment="invalid",  # type: ignore[arg-type]
            )

    def test_invalid_metadata_is_rejected(self) -> None:
        with pytest.raises(BenchmarkInputError):
            BenchmarkCase(
                case_id="case-1",
                experiment=make_input(),
                metadata=[],  # type: ignore[arg-type]
            )


class TestBenchmarkRun:
    """Tests for BenchmarkRun."""

    def test_valid_empty_run(self) -> None:
        run = BenchmarkRun(
            benchmark_id="benchmark-1",
            benchmark_name="Test Benchmark",
            total_cases=0,
            completed_cases=0,
            failed_cases=0,
            results=(),
        )

        assert run.success_rate == 0.0
        assert run.has_failures is False
        assert run.is_complete is True

    def test_success_rate(self) -> None:
        result = make_result()

        run = BenchmarkRun(
            benchmark_id="benchmark-1",
            benchmark_name="Test Benchmark",
            total_cases=2,
            completed_cases=2,
            failed_cases=0,
            results=(result, result),
        )

        assert run.success_rate == 1.0
        assert run.has_failures is False
        assert run.is_complete is True

    def test_failed_run(self) -> None:
        result = make_result()

        run = BenchmarkRun(
            benchmark_id="benchmark-1",
            benchmark_name="Test Benchmark",
            total_cases=2,
            completed_cases=1,
            failed_cases=1,
            results=(result,),
            failures=(
                {
                    "case_id": "case-2",
                    "error_type": "RuntimeError",
                    "error": "test failure",
                },
            ),
        )

        assert run.success_rate == 0.5
        assert run.has_failures is True
        assert run.is_complete is True

    def test_invalid_case_counts_are_rejected(self) -> None:
        with pytest.raises(BenchmarkInputError):
            BenchmarkRun(
                benchmark_id="benchmark-1",
                benchmark_name="Test Benchmark",
                total_cases=1,
                completed_cases=1,
                failed_cases=1,
                results=(),
            )

    def test_invalid_result_type_is_rejected(self) -> None:
        with pytest.raises(BenchmarkInputError):
            BenchmarkRun(
                benchmark_id="benchmark-1",
                benchmark_name="Test Benchmark",
                total_cases=1,
                completed_cases=1,
                failed_cases=0,
                results=("invalid",),  # type: ignore[arg-type]
            )


class TestBenchmarkConfiguration:
    """Tests for benchmark construction and configuration."""

    def test_default_configuration(self) -> None:
        benchmark = Benchmark(
            benchmark_id="coderag-test",
        )

        assert benchmark.benchmark_id == "coderag-test"
        assert benchmark.name == "CodeRAG-PoisonBench"
        assert benchmark.version == "0.1.0"
        assert benchmark.case_count == 0
        assert benchmark.cases == ()

    def test_custom_configuration(self) -> None:
        benchmark = Benchmark(
            benchmark_id="benchmark-1",
            name="Research Benchmark",
            version="1.0.0",
            description="Controlled benchmark.",
            metadata={"owner": "research"},
        )

        assert benchmark.name == "Research Benchmark"
        assert benchmark.version == "1.0.0"
        assert benchmark.description == "Controlled benchmark."
        assert benchmark.metadata == {"owner": "research"}

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"benchmark_id": ""},
            {"benchmark_id": "   "},
        ],
    )
    def test_invalid_benchmark_id_is_rejected(
        self,
        kwargs: dict[str, Any],
    ) -> None:
        with pytest.raises(BenchmarkConfigurationError):
            Benchmark(**kwargs)

    def test_empty_name_is_rejected(self) -> None:
        with pytest.raises(BenchmarkConfigurationError):
            Benchmark(
                benchmark_id="benchmark-1",
                name="",
            )

    def test_empty_version_is_rejected(self) -> None:
        with pytest.raises(BenchmarkConfigurationError):
            Benchmark(
                benchmark_id="benchmark-1",
                version="",
            )

    def test_invalid_metadata_is_rejected(self) -> None:
        with pytest.raises(BenchmarkConfigurationError):
            Benchmark(
                benchmark_id="benchmark-1",
                metadata=[],  # type: ignore[arg-type]
            )


class TestBenchmarkCases:
    """Tests for benchmark case management."""

    def test_add_case(self) -> None:
        benchmark = Benchmark(
            benchmark_id="benchmark-1",
        )

        benchmark.add_case(make_case())

        assert benchmark.case_count == 1
        assert benchmark.get_case("case-1") == make_case()

    def test_add_multiple_cases(self) -> None:
        benchmark = Benchmark(
            benchmark_id="benchmark-1",
        )

        benchmark.add_cases(
            [
                make_case(
                    "case-1",
                    experiment_id="experiment-1",
                ),
                make_case(
                    "case-2",
                    experiment_id="experiment-2",
                ),
            ]
        )

        assert benchmark.case_count == 2

    def test_cases_are_sorted_by_case_id(self) -> None:
        benchmark = Benchmark(
            benchmark_id="benchmark-1",
        )

        benchmark.add_case(
            make_case(
                "case-z",
                experiment_id="experiment-z",
            )
        )
        benchmark.add_case(
            make_case(
                "case-a",
                experiment_id="experiment-a",
            )
        )
        benchmark.add_case(
            make_case(
                "case-m",
                experiment_id="experiment-m",
            )
        )

        assert [
            case.case_id
            for case in benchmark.cases
        ] == [
            "case-a",
            "case-m",
            "case-z",
        ]

    def test_duplicate_case_id_is_rejected(self) -> None:
        benchmark = Benchmark(
            benchmark_id="benchmark-1",
        )

        benchmark.add_case(make_case())

        with pytest.raises(BenchmarkInputError):
            benchmark.add_case(make_case())

    def test_duplicate_experiment_id_is_rejected(self) -> None:
        benchmark = Benchmark(
            benchmark_id="benchmark-1",
        )

        benchmark.add_case(
            make_case(
                "case-1",
                experiment_id="experiment-1",
            )
        )

        with pytest.raises(BenchmarkInputError):
            benchmark.add_case(
                make_case(
                    "case-2",
                    experiment_id="experiment-1",
                )
            )

    def test_invalid_case_type_is_rejected(self) -> None:
        benchmark = Benchmark(
            benchmark_id="benchmark-1",
        )

        with pytest.raises(BenchmarkInputError):
            benchmark.add_case("invalid")  # type: ignore[arg-type]

    def test_get_unknown_case_is_rejected(self) -> None:
        benchmark = Benchmark(
            benchmark_id="benchmark-1",
        )

        with pytest.raises(BenchmarkInputError):
            benchmark.get_case("missing")

    def test_remove_case(self) -> None:
        benchmark = Benchmark(
            benchmark_id="benchmark-1",
        )

        benchmark.add_case(make_case())

        removed = benchmark.remove_case("case-1")

        assert removed.case_id == "case-1"
        assert benchmark.case_count == 0

    def test_remove_unknown_case_is_rejected(self) -> None:
        benchmark = Benchmark(
            benchmark_id="benchmark-1",
        )

        with pytest.raises(BenchmarkInputError):
            benchmark.remove_case("missing")

    def test_clear_removes_all_cases(self) -> None:
        benchmark = Benchmark(
            benchmark_id="benchmark-1",
            cases=[
                make_case(
                    "case-1",
                    experiment_id="experiment-1",
                ),
                make_case(
                    "case-2",
                    experiment_id="experiment-2",
                ),
            ],
        )

        benchmark.clear()

        assert benchmark.case_count == 0
        assert benchmark.cases == ()

    def test_len_and_iteration(self) -> None:
        benchmark = Benchmark(
            benchmark_id="benchmark-1",
            cases=[
                make_case(
                    "case-2",
                    experiment_id="experiment-2",
                ),
                make_case(
                    "case-1",
                    experiment_id="experiment-1",
                ),
            ],
        )

        assert len(benchmark) == 2
        assert [
            case.case_id
            for case in benchmark
        ] == [
            "case-1",
            "case-2",
        ]


class TestBenchmarkValidation:
    """Tests for benchmark validation."""

    def test_valid_benchmark_passes_validation(self) -> None:
        benchmark = Benchmark(
            benchmark_id="benchmark-1",
            cases=[
                make_case(
                    "case-1",
                    experiment_id="experiment-1",
                ),
                make_case(
                    "case-2",
                    experiment_id="experiment-2",
                    condition=(
                        ExperimentCondition.POISONED_NO_DEFENSE
                    ),
                ),
            ],
        )

        benchmark.validate()

    def test_invalid_top_k_is_rejected_by_experiment_config(
        self,
    ) -> None:
        with pytest.raises(Exception, match="top_k must be >= 1"):
            make_config(top_k=0)

    def test_invalid_seed_is_rejected(self) -> None:
        experiment = ExperimentRunInput(
            query="Generate code.",
            task="code_generation",
            experiment_config=make_config(
                seed=-1,
            ),
            request_id="request-invalid-seed",
        )

        case = BenchmarkCase(
            case_id="case-invalid-seed",
            experiment=experiment,
        )

        with pytest.raises(BenchmarkInputError):
            Benchmark(
                benchmark_id="benchmark-1",
                cases=[case],
            )


class TestBenchmarkExecution:
    """Tests for benchmark execution."""

    def test_empty_benchmark_runs_successfully(self) -> None:
        benchmark = Benchmark(
            benchmark_id="benchmark-empty",
        )

        runner = StubExperimentRunner()

        run = benchmark.run(runner)

        assert run.total_cases == 0
        assert run.completed_cases == 0
        assert run.failed_cases == 0
        assert run.results == ()
        assert run.failures == ()
        assert run.is_complete is True
        assert runner.calls == []

    def test_successful_cases_are_executed(self) -> None:
        benchmark = Benchmark(
            benchmark_id="benchmark-success",
            cases=[
                make_case(
                    "case-b",
                    experiment_id="experiment-b",
                ),
                make_case(
                    "case-a",
                    experiment_id="experiment-a",
                ),
            ],
        )

        runner = StubExperimentRunner()

        run = benchmark.run(runner)

        assert run.total_cases == 2
        assert run.completed_cases == 2
        assert run.failed_cases == 0
        assert len(run.results) == 2
        assert run.failures == ()

        assert runner.calls == [
            "experiment-a",
            "experiment-b",
        ]

    def test_custom_results_are_preserved(self) -> None:
        benchmark = Benchmark(
            benchmark_id="benchmark-results",
            cases=[
                make_case(
                    "case-1",
                    experiment_id="experiment-1",
                ),
            ],
        )

        expected = make_result(
            experiment_id="experiment-1",
        )

        runner = StubExperimentRunner(
            results={
                "experiment-1": expected,
            }
        )

        run = benchmark.run(runner)

        assert run.results == (expected,)

    def test_failed_case_is_recorded(self) -> None:
        benchmark = Benchmark(
            benchmark_id="benchmark-failure",
            cases=[
                make_case(
                    "case-1",
                    experiment_id="experiment-1",
                ),
            ],
        )

        runner = StubExperimentRunner(
            failures={
                "experiment-1": RuntimeError(
                    "synthetic experiment failure"
                ),
            }
        )

        run = benchmark.run(runner)

        assert run.total_cases == 1
        assert run.completed_cases == 0
        assert run.failed_cases == 1
        assert len(run.failures) == 1

        failure = run.failures[0]

        assert failure["case_id"] == "case-1"
        assert failure["experiment_id"] == "experiment-1"
        assert failure["error_type"] == "RuntimeError"
        assert failure["error"] == (
            "synthetic experiment failure"
        )

    def test_stop_on_error_stops_execution(self) -> None:
        benchmark = Benchmark(
            benchmark_id="benchmark-stop",
            cases=[
                make_case(
                    "case-a",
                    experiment_id="experiment-a",
                ),
                make_case(
                    "case-b",
                    experiment_id="experiment-b",
                ),
                make_case(
                    "case-c",
                    experiment_id="experiment-c",
                ),
            ],
        )

        runner = StubExperimentRunner(
            failures={
                "experiment-a": RuntimeError(
                    "first failure"
                ),
            }
        )

        run = benchmark.run(
            runner,
            stop_on_error=True,
        )

        assert run.total_cases == 3
        assert run.completed_cases == 0
        assert run.failed_cases == 1
        assert len(run.failures) == 1
        assert runner.calls == ["experiment-a"]

    def test_execution_continues_after_failure_by_default(
        self,
    ) -> None:
        benchmark = Benchmark(
            benchmark_id="benchmark-continue",
            cases=[
                make_case(
                    "case-a",
                    experiment_id="experiment-a",
                ),
                make_case(
                    "case-b",
                    experiment_id="experiment-b",
                ),
            ],
        )

        runner = StubExperimentRunner(
            failures={
                "experiment-a": RuntimeError(
                    "first failure"
                ),
            }
        )

        run = benchmark.run(runner)

        assert run.total_cases == 2
        assert run.completed_cases == 1
        assert run.failed_cases == 1

        assert runner.calls == [
            "experiment-a",
            "experiment-b",
        ]

    def test_execution_metadata_is_preserved(self) -> None:
        benchmark = Benchmark(
            benchmark_id="benchmark-metadata",
            name="Research Benchmark",
            version="2.0.0",
            metadata={
                "dataset": "CodeRAG-PoisonBench",
            },
            cases=[
                make_case(),
            ],
        )

        runner = StubExperimentRunner()

        run = benchmark.run(
            runner,
            metadata={
                "run_id": "run-001",
            },
        )

        assert run.metadata["dataset"] == (
            "CodeRAG-PoisonBench"
        )
        assert run.metadata["run_id"] == "run-001"
        assert run.metadata["benchmark_id"] == (
            "benchmark-metadata"
        )
        assert run.metadata["benchmark_name"] == (
            "Research Benchmark"
        )
        assert run.metadata["benchmark_version"] == "2.0.0"
        assert run.metadata["case_order"] == ["case-1"]
        assert run.metadata["stop_on_error"] is False

    def test_run_rejects_invalid_runner(self) -> None:
        benchmark = Benchmark(
            benchmark_id="benchmark-1",
        )

        with pytest.raises(BenchmarkInputError):
            benchmark.run("invalid")  # type: ignore[arg-type]

    def test_run_rejects_invalid_stop_on_error(self) -> None:
        benchmark = Benchmark(
            benchmark_id="benchmark-1",
        )

        runner = StubExperimentRunner()

        with pytest.raises(BenchmarkInputError):
            benchmark.run(
                runner,
                stop_on_error="yes",  # type: ignore[arg-type]
            )

    def test_run_rejects_invalid_metadata(self) -> None:
        benchmark = Benchmark(
            benchmark_id="benchmark-1",
        )

        runner = StubExperimentRunner()

        with pytest.raises(BenchmarkInputError):
            benchmark.run(
                runner,
                metadata=[],  # type: ignore[arg-type]
            )


class TestBenchmarkSummary:
    """Tests for benchmark summaries."""

    def test_summary_contains_condition_counts(self) -> None:
        benchmark = Benchmark(
            benchmark_id="coderag-poisonbench",
            name="CodeRAG-PoisonBench",
            version="0.1.0",
            description="Controlled RAG security benchmark.",
            cases=[
                make_case(
                    "case-a",
                    experiment_id="experiment-a",
                    condition=(
                        ExperimentCondition.CLEAN_NO_DEFENSE
                    ),
                ),
                make_case(
                    "case-b",
                    experiment_id="experiment-b",
                    condition=(
                        ExperimentCondition.POISONED_NO_DEFENSE
                    ),
                ),
                make_case(
                    "case-c",
                    experiment_id="experiment-c",
                    condition=(
                        ExperimentCondition.POISONED_DEFENSE
                    ),
                ),
                make_case(
                    "case-d",
                    experiment_id="experiment-d",
                    condition=(
                        ExperimentCondition.CLEAN_DEFENSE
                    ),
                ),
            ],
        )

        summary = benchmark.summary()

        assert summary["benchmark_id"] == (
            "coderag-poisonbench"
        )
        assert summary["name"] == "CodeRAG-PoisonBench"
        assert summary["version"] == "0.1.0"
        assert summary["case_count"] == 4

        assert summary["conditions"] == {
            "A_clean_no_defense": 1,
            "B_poisoned_no_defense": 1,
            "C_poisoned_defense": 1,
            "D_clean_defense": 1,
        }

        assert summary["experiment_ids"] == [
            "experiment-a",
            "experiment-b",
            "experiment-c",
            "experiment-d",
        ]

    def test_repr_contains_basic_information(self) -> None:
        benchmark = Benchmark(
            benchmark_id="benchmark-1",
            cases=[
                make_case(),
            ],
        )

        representation = repr(benchmark)

        assert "benchmark-1" in representation
        assert "CodeRAG-PoisonBench" in representation
        assert "cases=1" in representation


class TestBenchmarkConditionCoverage:
    """Tests for controlled A/B/C/D benchmark conditions."""

    def test_all_four_conditions_can_be_registered(self) -> None:
        benchmark = Benchmark(
            benchmark_id="benchmark-abcd",
            cases=[
                make_case(
                    "case-a",
                    experiment_id="experiment-a",
                    condition=(
                        ExperimentCondition.CLEAN_NO_DEFENSE
                    ),
                ),
                make_case(
                    "case-b",
                    experiment_id="experiment-b",
                    condition=(
                        ExperimentCondition.POISONED_NO_DEFENSE
                    ),
                ),
                make_case(
                    "case-c",
                    experiment_id="experiment-c",
                    condition=(
                        ExperimentCondition.POISONED_DEFENSE
                    ),
                ),
                make_case(
                    "case-d",
                    experiment_id="experiment-d",
                    condition=(
                        ExperimentCondition.CLEAN_DEFENSE
                    ),
                ),
            ],
        )

        assert benchmark.case_count == 4

        assert {
            case.condition
            for case in benchmark.cases
        } == {
            ExperimentCondition.CLEAN_NO_DEFENSE,
            ExperimentCondition.POISONED_NO_DEFENSE,
            ExperimentCondition.POISONED_DEFENSE,
            ExperimentCondition.CLEAN_DEFENSE,
        }