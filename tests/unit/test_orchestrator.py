"""Unit tests for the experiment orchestrator."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.experiments.orchestrator import (
    ExperimentOrchestrationConfigurationError,
    ExperimentOrchestrationInputError,
    ExperimentOrchestrator,
    ExperimentPlan,
    ExperimentSuite,
)
from src.models import ExperimentCondition


class TestExperimentPlan:
    """Tests for ExperimentPlan validation."""

    def test_valid_plan_is_created(self, experiment_config) -> None:
        plan = ExperimentPlan(
            experiment_config=experiment_config,
            query="Generate secure code.",
            task="secure_code_generation",
            request_id="request-001",
        )

        assert plan.experiment_config is experiment_config
        assert plan.query == "Generate secure code."
        assert plan.task == "secure_code_generation"
        assert plan.request_id == "request-001"
        assert plan.baseline_generated_code is None
        assert plan.metadata == {}

    def test_empty_query_is_rejected(self, experiment_config) -> None:
        with pytest.raises(ExperimentOrchestrationInputError):
            ExperimentPlan(
                experiment_config=experiment_config,
                query="",
                task="secure_code_generation",
                request_id="request-001",
            )

    def test_empty_task_is_rejected(self, experiment_config) -> None:
        with pytest.raises(ExperimentOrchestrationInputError):
            ExperimentPlan(
                experiment_config=experiment_config,
                query="Generate code.",
                task="",
                request_id="request-001",
            )

    def test_empty_request_id_is_rejected(self, experiment_config) -> None:
        with pytest.raises(ExperimentOrchestrationInputError):
            ExperimentPlan(
                experiment_config=experiment_config,
                query="Generate code.",
                task="secure_code_generation",
                request_id="",
            )

    def test_invalid_experiment_config_is_rejected(self) -> None:
        with pytest.raises(ExperimentOrchestrationInputError):
            ExperimentPlan(
                experiment_config="invalid",
                query="Generate code.",
                task="secure_code_generation",
                request_id="request-001",
            )


class TestExperimentSuite:
    """Tests for ExperimentSuite behavior."""

    def test_plan_count(self, experiment_config) -> None:
        plan = ExperimentPlan(
            experiment_config=experiment_config,
            query="Generate code.",
            task="task-1",
            request_id="request-001",
        )

        suite = ExperimentSuite(
            matrix_name="test-matrix",
            matrix_version="1.0",
            plans=(plan,),
        )

        assert suite.plan_count == 1

    def test_conditions_returns_unique_conditions(
        self,
        experiment_config,
    ) -> None:
        plan_one = ExperimentPlan(
            experiment_config=experiment_config,
            query="Query one",
            task="task-1",
            request_id="request-001",
        )

        plan_two = ExperimentPlan(
            experiment_config=experiment_config,
            query="Query two",
            task="task-2",
            request_id="request-002",
        )

        suite = ExperimentSuite(
            matrix_name="test-matrix",
            matrix_version="1.0",
            plans=(plan_one, plan_two),
        )

        assert suite.conditions == (
            experiment_config.condition,
        )

    def test_to_runner_inputs_preserves_plan_data(
        self,
        experiment_config,
    ) -> None:
        plan = ExperimentPlan(
            experiment_config=experiment_config,
            query="Generate code.",
            task="task-1",
            request_id="request-001",
            baseline_generated_code="baseline code",
            metadata={"phase": 1},
        )

        suite = ExperimentSuite(
            matrix_name="test-matrix",
            matrix_version="1.0",
            plans=(plan,),
        )

        runner_inputs = suite.to_runner_inputs()

        assert len(runner_inputs) == 1

        runner_input = runner_inputs[0]

        assert runner_input.query == "Generate code."
        assert runner_input.task == "task-1"
        assert runner_input.experiment_config is experiment_config
        assert runner_input.request_id == "request-001"
        assert runner_input.baseline_generated_code == "baseline code"
        assert runner_input.metadata == {"phase": 1}


class TestExperimentOrchestrator:
    """Tests for ExperimentOrchestrator."""

    def test_constructor_creates_default_loader(self) -> None:
        orchestrator = ExperimentOrchestrator()

        assert orchestrator.config_loader is not None

    def test_custom_loader_is_preserved(self) -> None:
        class DummyLoader:
            pass

        loader = DummyLoader()

        orchestrator = ExperimentOrchestrator(
            config_loader=loader,
        )

        assert orchestrator.config_loader is loader

    def test_invalid_matrix_path_is_rejected(self) -> None:
        orchestrator = ExperimentOrchestrator()

        with pytest.raises(ExperimentOrchestrationInputError):
            orchestrator.load_configuration(123)

    def test_invalid_queries_type_is_rejected(self) -> None:
        with pytest.raises(ExperimentOrchestrationInputError):
            ExperimentOrchestrator._normalize_queries("invalid")

    def test_invalid_baselines_type_is_rejected(self) -> None:
        with pytest.raises(ExperimentOrchestrationInputError):
            ExperimentOrchestrator._normalize_baselines("invalid")

    def test_invalid_query_key_is_rejected(self) -> None:
        with pytest.raises(ExperimentOrchestrationInputError):
            ExperimentOrchestrator._normalize_queries(
                {"": ["query"]},
            )

    def test_invalid_query_value_is_rejected(self) -> None:
        with pytest.raises(ExperimentOrchestrationInputError):
            ExperimentOrchestrator._normalize_queries(
                {"condition": [""]},
            )

    def test_invalid_baseline_key_is_rejected(self) -> None:
        with pytest.raises(ExperimentOrchestrationInputError):
            ExperimentOrchestrator._normalize_baselines(
                {"": "code"},
            )

    def test_invalid_baseline_value_is_rejected(self) -> None:
        with pytest.raises(ExperimentOrchestrationInputError):
            ExperimentOrchestrator._normalize_baselines(
                {"task": 123},
            )

    def test_queries_are_normalized(self) -> None:
        result = ExperimentOrchestrator._normalize_queries(
            {
                "condition": ["query-1", "query-2"],
            },
        )

        assert result == {
            "condition": ("query-1", "query-2"),
        }

    def test_string_query_is_normalized(self) -> None:
        result = ExperimentOrchestrator._normalize_queries(
            {
                "condition": "query",
            },
        )

        assert result == {
            "condition": ("query",),
        }

    def test_baselines_are_normalized(self) -> None:
        result = ExperimentOrchestrator._normalize_baselines(
            {
                "task": "baseline code",
            },
        )

        assert result == {
            "task": "baseline code",
        }

    def test_request_id_is_deterministic(self) -> None:
        request_id = ExperimentOrchestrator._request_id(
            condition=ExperimentCondition.CLEAN_NO_DEFENSE,
            task="task",
            language="python",
            repetition=0,
        )

        assert request_id.startswith("request__")
        assert "task" in request_id
        assert "rep1" in request_id

    def test_baseline_key_is_deterministic(self) -> None:
        key = ExperimentOrchestrator._baseline_key(
            condition=ExperimentCondition.CLEAN_NO_DEFENSE,
            task="task",
            language="python",
            repetition=0,
        )

        assert key.startswith(
            ExperimentCondition.CLEAN_NO_DEFENSE.value,
        )
        assert "task" in key
        assert "rep1" in key


@pytest.fixture
def experiment_config():
    """Create a minimal valid ExperimentConfig for unit tests."""

    from src.models import ExperimentConfig, ProgrammingLanguage

    return ExperimentConfig(
        experiment_id="experiment-001",
        condition=ExperimentCondition.CLEAN_NO_DEFENSE,
        repository_id="dataset/test",
        task_type="secure_code_generation",
        language=ProgrammingLanguage.PYTHON,
        model_name="test-model",
        retriever_name="test-retriever",
        top_k=3,
        defense_layers=[],
        seed=42,
        metadata={},
    )