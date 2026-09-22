"""Unit tests for the SecureCodeRAG result store."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.experiments.benchmark import BenchmarkRun
from src.experiments.experiment_runner import ExperimentRunInput
from src.experiments.result_store import (
    ResultStore,
    ResultStoreError,
    ResultStoreInputError,
)
from src.models import (
    ExperimentCondition,
    ExperimentConfig,
    ExperimentResult,
    ProgrammingLanguage,
)


def make_result(
    *,
    experiment_id: str = "exp-001",
    condition: ExperimentCondition = (
        ExperimentCondition.CLEAN_NO_DEFENSE
    ),
    repository_id: str = "repo-001",
    task_type: str = "code_generation",
    model_name: str = "test-model",
    retriever_name: str = "test-retriever",
    defense_enabled: bool = False,
    utility_score: float = 0.8,
    created_at: datetime | None = None,
) -> ExperimentResult:
    """Create a valid ExperimentResult for testing."""
    if created_at is None:
        created_at = datetime.now(timezone.utc)

    return ExperimentResult(
        experiment_id=experiment_id,
        condition=condition,
        request_id=f"request-{experiment_id}",
        poison_present=condition
        in {
            ExperimentCondition.POISONED_NO_DEFENSE,
            ExperimentCondition.POISONED_DEFENSE,
        },
        poison_retrieved=condition
        in {
            ExperimentCondition.POISONED_NO_DEFENSE,
            ExperimentCondition.POISONED_DEFENSE,
        },
        poison_in_context=condition
        in {
            ExperimentCondition.POISONED_NO_DEFENSE,
            ExperimentCondition.POISONED_DEFENSE,
        },
        generation_changed=False,
        vulnerability_introduced=False,
        poison_retrieval_rate=0.0,
        attack_success_rate=0.0,
        vulnerability_introduction_rate=0.0,
        utility_score=utility_score,
        latency_ms=100.0,
        findings=[],
        metadata={
            "repository_id": repository_id,
            "task_type": task_type,
            "model_name": model_name,
            "retriever_name": retriever_name,
            "defense_enabled": defense_enabled,
        },
        created_at=created_at,
    )


def make_experiment_input(
    *,
    experiment_id: str = "exp-001",
    condition: ExperimentCondition = (
        ExperimentCondition.CLEAN_NO_DEFENSE
    ),
) -> ExperimentRunInput:
    """Create a minimal ExperimentRunInput for benchmark tests."""
    config = ExperimentConfig(
        experiment_id=experiment_id,
        condition=condition,
        repository_id="repo-001",
        task_type="code_generation",
        language=ProgrammingLanguage.PYTHON,
        model_name="test-model",
        retriever_name="test-retriever",
        top_k=5,
        defense_layers=[],
        seed=42,
    )

    return ExperimentRunInput(
        experiment_config=config,
        task="Generate a Python function.",
        query="Create a function that returns one.",
    )


def make_benchmark_run(
    *,
    benchmark_id: str = "benchmark-001",
    results: tuple[ExperimentResult, ...] | None = None,
) -> BenchmarkRun:
    """Create a valid BenchmarkRun for testing."""
    if results is None:
        results = (
            make_result(
                experiment_id="exp-001",
            ),
        )

    return BenchmarkRun(
        benchmark_id=benchmark_id,
        benchmark_name="CodeRAG-PoisonBench",
        total_cases=len(results),
        completed_cases=len(results),
        failed_cases=0,
        results=results,
        failures=(),
        duration_ms=125.0,
        metadata={
            "run_id": f"run-{benchmark_id}",
        },
    )


class TestResultStoreInitialization:
    """Tests for ResultStore initialization."""

    def test_creates_result_directory(
        self,
        tmp_path: Path,
    ) -> None:
        result_path = tmp_path / "results"

        store = ResultStore(result_path)

        assert store.root_path == result_path.resolve()
        assert result_path.exists()
        assert result_path.is_dir()

    def test_exposes_expected_file_paths(
        self,
        tmp_path: Path,
    ) -> None:
        store = ResultStore(tmp_path / "results")

        assert (
            store.experiments_path.name
            == "experiments.jsonl"
        )
        assert (
            store.benchmark_runs_path.name
            == "benchmark_runs.jsonl"
        )

    def test_rejects_file_as_root_path(
        self,
        tmp_path: Path,
    ) -> None:
        invalid_path = tmp_path / "not-a-directory.txt"
        invalid_path.write_text(
            "invalid",
            encoding="utf-8",
        )

        with pytest.raises(
            ValueError,
            match="not a directory",
        ):
            ResultStore(invalid_path)

    def test_can_initialize_without_creating_directory(
        self,
        tmp_path: Path,
    ) -> None:
        result_path = tmp_path / "results"

        store = ResultStore(
            result_path,
            create=False,
        )

        assert store.root_path == result_path.resolve()
        assert not result_path.exists()


class TestExperimentPersistence:
    """Tests for ExperimentResult persistence."""

    def test_save_and_load_experiment(
        self,
        tmp_path: Path,
    ) -> None:
        store = ResultStore(tmp_path / "results")

        result = make_result(
            experiment_id="exp-save-load",
        )

        store.save_experiment(result)

        loaded = store.load_experiment(
            "exp-save-load",
        )

        assert loaded.experiment_id == result.experiment_id
        assert loaded.condition == result.condition
        assert loaded.request_id == result.request_id
        assert loaded.utility_score == result.utility_score
        assert loaded.metadata == result.metadata

    def test_save_creates_jsonl_file(
        self,
        tmp_path: Path,
    ) -> None:
        store = ResultStore(tmp_path / "results")

        store.save_experiment(
            make_result(),
        )

        assert store.experiments_path.exists()

        lines = store.experiments_path.read_text(
            encoding="utf-8",
        ).splitlines()

        assert len(lines) == 1

        record = json.loads(lines[0])

        assert record["record_type"] == "experiment_result"
        assert record["experiment_id"] == "exp-001"

    def test_save_dispatches_experiment_result(
        self,
        tmp_path: Path,
    ) -> None:
        store = ResultStore(tmp_path / "results")

        result = make_result()

        store.save(result)

        assert store.count_experiments() == 1

    def test_duplicate_experiment_is_rejected(
        self,
        tmp_path: Path,
    ) -> None:
        store = ResultStore(tmp_path / "results")

        result = make_result()

        store.save_experiment(result)

        with pytest.raises(
            ResultStoreInputError,
            match="already exists",
        ):
            store.save_experiment(result)

    def test_overwrite_replaces_existing_experiment(
        self,
        tmp_path: Path,
    ) -> None:
        store = ResultStore(tmp_path / "results")

        original = make_result(
            utility_score=0.4,
        )

        updated = make_result(
            utility_score=0.9,
        )

        store.save_experiment(original)

        store.save_experiment(
            updated,
            overwrite=True,
        )

        loaded = store.load_experiment(
            "exp-001",
        )

        assert loaded.utility_score == 0.9
        assert store.count_experiments() == 1

    def test_load_missing_experiment_fails(
        self,
        tmp_path: Path,
    ) -> None:
        store = ResultStore(tmp_path / "results")

        with pytest.raises(
            ResultStoreInputError,
            match="not found",
        ):
            store.load_experiment("missing")

    def test_load_all_experiments(
        self,
        tmp_path: Path,
    ) -> None:
        store = ResultStore(tmp_path / "results")

        first = make_result(
            experiment_id="exp-001",
        )
        second = make_result(
            experiment_id="exp-002",
        )

        store.save_experiment(first)
        store.save_experiment(second)

        results = store.load_experiments()

        assert len(results) == 2
        assert {
            result.experiment_id
            for result in results
        } == {
            "exp-001",
            "exp-002",
        }

    def test_list_experiment_ids(
        self,
        tmp_path: Path,
    ) -> None:
        store = ResultStore(tmp_path / "results")

        store.save_experiment(
            make_result(
                experiment_id="exp-b",
            )
        )
        store.save_experiment(
            make_result(
                experiment_id="exp-a",
            )
        )

        assert store.list_experiment_ids() == (
            "exp-b",
            "exp-a",
        )

    def test_count_experiments(
        self,
        tmp_path: Path,
    ) -> None:
        store = ResultStore(tmp_path / "results")

        assert store.count_experiments() == 0

        store.save_experiment(
            make_result(
                experiment_id="exp-001",
            )
        )
        store.save_experiment(
            make_result(
                experiment_id="exp-002",
            )
        )

        assert store.count_experiments() == 2


class TestExperimentQueries:
    """Tests for experiment-result filtering."""

    @pytest.fixture
    def populated_store(
        self,
        tmp_path: Path,
    ) -> ResultStore:
        store = ResultStore(tmp_path / "results")

        store.save_experiment(
            make_result(
                experiment_id="clean-001",
                condition=(
                    ExperimentCondition.CLEAN_NO_DEFENSE
                ),
                repository_id="repo-a",
                task_type="code_generation",
                model_name="model-a",
                retriever_name="retriever-a",
                defense_enabled=False,
            )
        )

        store.save_experiment(
            make_result(
                experiment_id="poisoned-001",
                condition=(
                    ExperimentCondition.POISONED_NO_DEFENSE
                ),
                repository_id="repo-a",
                task_type="bug_fixing",
                model_name="model-b",
                retriever_name="retriever-a",
                defense_enabled=False,
            )
        )

        store.save_experiment(
            make_result(
                experiment_id="defended-001",
                condition=(
                    ExperimentCondition.POISONED_DEFENSE
                ),
                repository_id="repo-b",
                task_type="code_generation",
                model_name="model-b",
                retriever_name="retriever-b",
                defense_enabled=True,
            )
        )

        return store

    def test_query_by_experiment_id(
        self,
        populated_store: ResultStore,
    ) -> None:
        results = populated_store.query_experiments(
            experiment_id="poisoned-001",
        )

        assert len(results) == 1
        assert results[0].experiment_id == "poisoned-001"

    def test_query_by_condition(
        self,
        populated_store: ResultStore,
    ) -> None:
        results = populated_store.query_experiments(
            condition=ExperimentCondition.POISONED_DEFENSE,
        )

        assert len(results) == 1
        assert (
            results[0].condition
            == ExperimentCondition.POISONED_DEFENSE
        )

    def test_query_by_condition_string(
        self,
        populated_store: ResultStore,
    ) -> None:
        results = populated_store.query_experiments(
            condition="C_poisoned_defense",
        )

        assert len(results) == 1
        assert results[0].experiment_id == "defended-001"

    def test_query_by_repository(
        self,
        populated_store: ResultStore,
    ) -> None:
        results = populated_store.query_experiments(
            repository_id="repo-a",
        )

        assert len(results) == 2

    def test_query_by_task_type(
        self,
        populated_store: ResultStore,
    ) -> None:
        results = populated_store.query_experiments(
            task_type="bug_fixing",
        )

        assert len(results) == 1
        assert results[0].experiment_id == "poisoned-001"

    def test_query_by_model(
        self,
        populated_store: ResultStore,
    ) -> None:
        results = populated_store.query_experiments(
            model_name="model-b",
        )

        assert len(results) == 2

    def test_query_by_retriever(
        self,
        populated_store: ResultStore,
    ) -> None:
        results = populated_store.query_experiments(
            retriever_name="retriever-b",
        )

        assert len(results) == 1

    def test_query_by_defense_state(
        self,
        populated_store: ResultStore,
    ) -> None:
        results = populated_store.query_experiments(
            defense_enabled=True,
        )

        assert len(results) == 1
        assert results[0].experiment_id == "defended-001"

    def test_query_without_filters_returns_all(
        self,
        populated_store: ResultStore,
    ) -> None:
        results = populated_store.query_experiments()

        assert len(results) == 3

    def test_invalid_condition_is_rejected(
        self,
        populated_store: ResultStore,
    ) -> None:
        with pytest.raises(
            ResultStoreInputError,
            match="Unknown experiment condition",
        ):
            populated_store.query_experiments(
                condition="invalid-condition",
            )

    def test_empty_filter_identifier_is_rejected(
        self,
        populated_store: ResultStore,
    ) -> None:
        with pytest.raises(
            ResultStoreInputError,
            match="experiment_id cannot be empty",
        ):
            populated_store.query_experiments(
                experiment_id=" ",
            )


class TestBenchmarkPersistence:
    """Tests for BenchmarkRun persistence."""

    def test_save_and_load_benchmark_run(
        self,
        tmp_path: Path,
    ) -> None:
        store = ResultStore(tmp_path / "results")

        result = make_result(
            experiment_id="benchmark-exp",
        )

        run = make_benchmark_run(
            benchmark_id="benchmark-001",
            results=(result,),
        )

        store.save_benchmark_run(run)

        loaded_runs = store.load_benchmark_runs()

        assert len(loaded_runs) == 1

        loaded = loaded_runs[0]

        assert loaded.benchmark_id == run.benchmark_id
        assert loaded.benchmark_name == run.benchmark_name
        assert loaded.total_cases == run.total_cases
        assert loaded.completed_cases == run.completed_cases
        assert len(loaded.results) == 1
        assert (
            loaded.results[0].experiment_id
            == "benchmark-exp"
        )

    def test_save_dispatches_benchmark_run(
        self,
        tmp_path: Path,
    ) -> None:
        store = ResultStore(tmp_path / "results")

        run = make_benchmark_run()

        store.save(run)

        assert store.count_benchmark_runs() == 1

    def test_duplicate_benchmark_run_is_rejected(
        self,
        tmp_path: Path,
    ) -> None:
        store = ResultStore(tmp_path / "results")

        run = make_benchmark_run()

        store.save_benchmark_run(run)

        with pytest.raises(
            ResultStoreInputError,
            match="already exists",
        ):
            store.save_benchmark_run(run)

    def test_benchmark_run_overwrite(
        self,
        tmp_path: Path,
    ) -> None:
        store = ResultStore(tmp_path / "results")

        original = make_benchmark_run(
            benchmark_id="benchmark-001",
        )

        updated = BenchmarkRun(
            benchmark_id="benchmark-001",
            benchmark_name="CodeRAG-PoisonBench",
            total_cases=2,
            completed_cases=2,
            failed_cases=0,
            results=(
                make_result(
                    experiment_id="exp-001",
                ),
                make_result(
                    experiment_id="exp-002",
                ),
            ),
            failures=(),
            duration_ms=250.0,
            metadata={
                "run_id": "run-benchmark-001",
            },
        )

        store.save_benchmark_run(original)

        store.save_benchmark_run(
            updated,
            overwrite=True,
        )

        loaded = store.load_benchmark_runs()

        assert len(loaded) == 1
        assert loaded[0].total_cases == 2
        assert len(loaded[0].results) == 2

    def test_list_benchmark_ids(
        self,
        tmp_path: Path,
    ) -> None:
        store = ResultStore(tmp_path / "results")

        store.save_benchmark_run(
            make_benchmark_run(
                benchmark_id="benchmark-b",
            )
        )
        store.save_benchmark_run(
            make_benchmark_run(
                benchmark_id="benchmark-a",
            )
        )

        assert store.list_benchmark_ids() == (
            "benchmark-a",
            "benchmark-b",
        )


class TestResultStoreValidation:
    """Tests for invalid result-store inputs."""

    def test_save_rejects_invalid_result(
        self,
        tmp_path: Path,
    ) -> None:
        store = ResultStore(tmp_path / "results")

        with pytest.raises(
            ResultStoreInputError,
            match="ExperimentResult or BenchmarkRun",
        ):
            store.save("invalid")  # type: ignore[arg-type]

    def test_save_experiment_rejects_invalid_result(
        self,
        tmp_path: Path,
    ) -> None:
        store = ResultStore(tmp_path / "results")

        with pytest.raises(
            ResultStoreInputError,
            match="ExperimentResult",
        ):
            store.save_experiment(
                "invalid",  # type: ignore[arg-type]
            )

    def test_save_benchmark_rejects_invalid_run(
        self,
        tmp_path: Path,
    ) -> None:
        store = ResultStore(tmp_path / "results")

        with pytest.raises(
            ResultStoreInputError,
            match="BenchmarkRun",
        ):
            store.save_benchmark_run(
                "invalid",  # type: ignore[arg-type]
            )

    def test_non_boolean_overwrite_is_rejected(
        self,
        tmp_path: Path,
    ) -> None:
        store = ResultStore(tmp_path / "results")

        with pytest.raises(
            ResultStoreInputError,
            match="overwrite must be a boolean",
        ):
            store.save_experiment(
                make_result(),
                overwrite="yes",  # type: ignore[arg-type]
            )

    def test_invalid_defense_filter_is_rejected(
        self,
        tmp_path: Path,
    ) -> None:
        store = ResultStore(tmp_path / "results")

        with pytest.raises(
            ResultStoreInputError,
            match="defense_enabled must be a boolean",
        ):
            store.query_experiments(
                defense_enabled="yes",  # type: ignore[arg-type]
            )

    def test_non_string_identifier_is_rejected(
        self,
        tmp_path: Path,
    ) -> None:
        store = ResultStore(tmp_path / "results")

        with pytest.raises(
            ResultStoreInputError,
            match="experiment_id must be a string",
        ):
            store.load_experiment(
                123,  # type: ignore[arg-type]
            )


class TestResultStoreFileHandling:
    """Tests for filesystem and malformed-data handling."""

    def test_missing_files_are_treated_as_empty(
        self,
        tmp_path: Path,
    ) -> None:
        store = ResultStore(tmp_path / "results")

        assert store.load_experiments() == ()
        assert store.load_benchmark_runs() == ()
        assert store.count_experiments() == 0
        assert store.count_benchmark_runs() == 0

    def test_malformed_experiment_json_is_rejected(
        self,
        tmp_path: Path,
    ) -> None:
        store = ResultStore(tmp_path / "results")

        store.experiments_path.write_text(
            "{invalid-json}\n",
            encoding="utf-8",
        )

        with pytest.raises(
            ResultStoreError,
            match="Invalid JSON",
        ):
            store.load_experiments()

    def test_non_object_json_record_is_rejected(
        self,
        tmp_path: Path,
    ) -> None:
        store = ResultStore(tmp_path / "results")

        store.experiments_path.write_text(
            '["not-an-object"]\n',
            encoding="utf-8",
        )

        with pytest.raises(
            ResultStoreError,
            match="must be an object",
        ):
            store.load_experiments()

    def test_empty_lines_are_ignored(
        self,
        tmp_path: Path,
    ) -> None:
        store = ResultStore(tmp_path / "results")

        result = make_result()

        store.save_experiment(result)

        with store.experiments_path.open(
            "a",
            encoding="utf-8",
        ) as handle:
            handle.write("\n\n")

        loaded = store.load_experiments()

        assert len(loaded) == 1

    def test_clear_experiments(
        self,
        tmp_path: Path,
    ) -> None:
        store = ResultStore(tmp_path / "results")

        store.save_experiment(
            make_result(),
        )

        assert store.count_experiments() == 1

        store.clear_experiments()

        assert store.count_experiments() == 0
        assert not store.experiments_path.exists()

    def test_clear_benchmark_runs(
        self,
        tmp_path: Path,
    ) -> None:
        store = ResultStore(tmp_path / "results")

        store.save_benchmark_run(
            make_benchmark_run(),
        )

        assert store.count_benchmark_runs() == 1

        store.clear_benchmark_runs()

        assert store.count_benchmark_runs() == 0
        assert not store.benchmark_runs_path.exists()

    def test_clear_removes_all_results(
        self,
        tmp_path: Path,
    ) -> None:
        store = ResultStore(tmp_path / "results")

        store.save_experiment(
            make_result(),
        )
        store.save_benchmark_run(
            make_benchmark_run(),
        )

        store.clear()

        assert store.count_experiments() == 0
        assert store.count_benchmark_runs() == 0


class TestResultStoreOrdering:
    """Tests for deterministic result ordering."""

    def test_experiments_are_sorted_by_creation_time(
        self,
        tmp_path: Path,
    ) -> None:
        store = ResultStore(tmp_path / "results")

        later = make_result(
            experiment_id="later",
            created_at=datetime(
                2026,
                1,
                2,
                tzinfo=timezone.utc,
            ),
        )

        earlier = make_result(
            experiment_id="earlier",
            created_at=datetime(
                2026,
                1,
                1,
                tzinfo=timezone.utc,
            ),
        )

        store.save_experiment(later)
        store.save_experiment(earlier)

        results = store.load_experiments()

        assert [
            result.experiment_id
            for result in results
        ] == [
            "earlier",
            "later",
        ]


class TestResultStoreRoundTrip:
    """Tests for serialization round trips."""

    def test_all_experiment_metrics_survive_round_trip(
        self,
        tmp_path: Path,
    ) -> None:
        store = ResultStore(tmp_path / "results")

        result = ExperimentResult(
            experiment_id="round-trip",
            condition=(
                ExperimentCondition.POISONED_DEFENSE
            ),
            request_id="request-round-trip",
            poison_present=True,
            poison_retrieved=True,
            poison_in_context=True,
            generation_changed=True,
            vulnerability_introduced=True,
            poison_retrieval_rate=0.75,
            attack_success_rate=0.5,
            vulnerability_introduction_rate=0.25,
            utility_score=0.9,
            latency_ms=123.45,
            findings=[],
            metadata={
                "repository_id": "repo-round-trip",
                "model_name": "model-round-trip",
            },
            created_at=datetime(
                2026,
                9,
                23,
                12,
                30,
                tzinfo=timezone.utc,
            ),
        )

        store.save_experiment(result)

        loaded = store.load_experiment(
            "round-trip",
        )

        assert loaded.experiment_id == result.experiment_id
        assert loaded.condition == result.condition
        assert loaded.request_id == result.request_id
        assert (
            loaded.poison_present
            == result.poison_present
        )
        assert (
            loaded.poison_retrieved
            == result.poison_retrieved
        )
        assert (
            loaded.poison_in_context
            == result.poison_in_context
        )
        assert (
            loaded.generation_changed
            == result.generation_changed
        )
        assert (
            loaded.vulnerability_introduced
            == result.vulnerability_introduced
        )
        assert (
            loaded.poison_retrieval_rate
            == result.poison_retrieval_rate
        )
        assert (
            loaded.attack_success_rate
            == result.attack_success_rate
        )
        assert (
            loaded.vulnerability_introduction_rate
            == result.vulnerability_introduction_rate
        )
        assert loaded.utility_score == result.utility_score
        assert loaded.latency_ms == result.latency_ms
        assert loaded.metadata == result.metadata
        assert loaded.created_at == result.created_at

    def test_empty_store_is_safe_to_query(
        self,
        tmp_path: Path,
    ) -> None:
        store = ResultStore(tmp_path / "results")

        assert store.query_experiments() == ()
        assert store.list_experiment_ids() == ()
        assert store.list_benchmark_ids() == ()