"""Controlled experiment orchestration for SecureCodeRAG."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.experiments.config_loader import (
    ExperimentConfigLoader,
    ExperimentFileConfig,
    ExperimentMatrixConfig,
)
from src.experiments.experiment_runner import ExperimentRunInput
from src.models import ExperimentCondition, ExperimentConfig, ProgrammingLanguage


class ExperimentOrchestrationError(RuntimeError):
    """Base exception for experiment orchestration failures."""


class ExperimentOrchestrationInputError(
    ExperimentOrchestrationError
):
    """Raised when orchestration input is invalid."""


class ExperimentOrchestrationConfigurationError(
    ExperimentOrchestrationError
):
    """Raised when experiment configuration cannot be materialized."""


@dataclass(frozen=True)
class ExperimentPlan:
    """Executable description of one controlled experiment."""

    experiment_config: ExperimentConfig
    query: str
    task: str
    request_id: str
    baseline_generated_code: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.experiment_config, ExperimentConfig):
            raise ExperimentOrchestrationInputError(
                "experiment_config must be an ExperimentConfig."
            )

        if not isinstance(self.query, str) or not self.query.strip():
            raise ExperimentOrchestrationInputError(
                "query must be a non-empty string."
            )

        if not isinstance(self.task, str) or not self.task.strip():
            raise ExperimentOrchestrationInputError(
                "task must be a non-empty string."
            )

        if (
            not isinstance(self.request_id, str)
            or not self.request_id.strip()
        ):
            raise ExperimentOrchestrationInputError(
                "request_id must be a non-empty string."
            )

        if not isinstance(self.metadata, dict):
            raise ExperimentOrchestrationInputError(
                "metadata must be a dictionary."
            )


@dataclass(frozen=True)
class ExperimentSuite:
    """Materialized plans generated from one experiment matrix."""

    matrix_name: str
    matrix_version: str
    plans: tuple[ExperimentPlan, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def plan_count(self) -> int:
        """Return the number of executable plans."""
        return len(self.plans)

    @property
    def conditions(self) -> tuple[ExperimentCondition, ...]:
        """Return unique conditions represented by the plans."""
        values: list[ExperimentCondition] = []

        for plan in self.plans:
            condition = plan.experiment_config.condition
            if condition not in values:
                values.append(condition)

        return tuple(values)

    def to_runner_inputs(self) -> tuple[ExperimentRunInput, ...]:
        """Convert plans into ExperimentRunner inputs."""
        return tuple(
            ExperimentRunInput(
                query=plan.query,
                task=plan.task,
                experiment_config=plan.experiment_config,
                request_id=plan.request_id,
                baseline_generated_code=plan.baseline_generated_code,
                metadata=dict(plan.metadata),
            )
            for plan in self.plans
        )


class ExperimentOrchestrator:
    """Build reproducible ExperimentRunner inputs from JSON configuration."""

    def __init__(
        self,
        *,
        config_loader: ExperimentConfigLoader | None = None,
    ) -> None:
        self._config_loader = (
            config_loader
            if config_loader is not None
            else ExperimentConfigLoader()
        )

    @property
    def config_loader(self) -> ExperimentConfigLoader:
        """Return the configuration loader."""
        return self._config_loader

    def load_configuration(
        self,
        matrix_path: str | Path,
    ) -> tuple[
        ExperimentMatrixConfig,
        dict[str, ExperimentFileConfig],
    ]:
        """Load the matrix and all condition configuration files."""
        if not isinstance(matrix_path, (str, Path)):
            raise ExperimentOrchestrationInputError(
                "matrix_path must be a string or Path."
            )

        try:
            return self._config_loader.load_all(Path(matrix_path))
        except Exception as exc:
            raise ExperimentOrchestrationConfigurationError(
                f"Unable to load experiment configuration: {exc}"
            ) from exc

    def build_suite(
        self,
        matrix_path: str | Path,
        *,
        queries: dict[str, Iterable[str]] | None = None,
        baseline_outputs: dict[str, str] | None = None,
        model_name: str = "configured-default",
    ) -> ExperimentSuite:
        """Load configuration and build the complete executable suite."""
        matrix, configurations = self.load_configuration(matrix_path)

        return self.build_suite_from_configuration(
            matrix,
            configurations,
            queries=queries,
            baseline_outputs=baseline_outputs,
            model_name=model_name,
        )

    def build_suite_from_configuration(
        self,
        matrix: ExperimentMatrixConfig,
        configurations: dict[str, ExperimentFileConfig],
        *,
        queries: dict[str, Iterable[str]] | None = None,
        baseline_outputs: dict[str, str] | None = None,
        model_name: str = "configured-default",
    ) -> ExperimentSuite:
        """Build plans from already validated loader outputs."""
        if not isinstance(matrix, ExperimentMatrixConfig):
            raise ExperimentOrchestrationInputError(
                "matrix must be an ExperimentMatrixConfig."
            )

        if not isinstance(configurations, dict):
            raise ExperimentOrchestrationInputError(
                "configurations must be a dictionary."
            )

        if not isinstance(model_name, str) or not model_name.strip():
            raise ExperimentOrchestrationInputError(
                "model_name must be a non-empty string."
            )

        query_map = self._normalize_queries(queries)
        baseline_map = self._normalize_baselines(baseline_outputs)

        plans: list[ExperimentPlan] = []

        for matrix_condition in matrix.conditions:
            condition_key = matrix_condition.condition.value
            file_config = configurations.get(condition_key)

            if file_config is None:
                raise ExperimentOrchestrationConfigurationError(
                    "No configuration was loaded for "
                    f"{condition_key!r}."
                )

            plans.extend(
                self._build_condition_plans(
                    matrix=matrix,
                    file_config=file_config,
                    query_map=query_map,
                    baseline_map=baseline_map,
                    model_name=model_name,
                )
            )

        if not plans:
            raise ExperimentOrchestrationConfigurationError(
                "The experiment matrix produced no executable plans."
            )

        return ExperimentSuite(
            matrix_name=matrix.name,
            matrix_version=matrix.version,
            plans=tuple(plans),
            metadata={
                "research_scope": matrix.research_scope,
                "synthetic_only": matrix.synthetic_only,
                "authorized_research_only": (
                    matrix.authorized_research_only
                ),
                "seed": matrix.seed,
                "condition_count": len(matrix.conditions),
                "plan_count": len(plans),
            },
        )

    def _build_condition_plans(
        self,
        *,
        matrix: ExperimentMatrixConfig,
        file_config: ExperimentFileConfig,
        query_map: dict[str, tuple[str, ...]],
        baseline_map: dict[str, str],
        model_name: str,
    ) -> list[ExperimentPlan]:
        """Expand one condition configuration into experiment plans."""
        if len(file_config.conditions) != 1:
            raise ExperimentOrchestrationConfigurationError(
                "Each A/B/C/D configuration file must contain exactly "
                "one condition."
            )

        condition_config = file_config.conditions[0]
        condition = condition_config.condition

        repetitions_value = matrix.replication.get("repetitions", 1)

        if (
            isinstance(repetitions_value, bool)
            or not isinstance(repetitions_value, int)
            or repetitions_value < 1
        ):
            raise ExperimentOrchestrationConfigurationError(
                "Matrix replication.repetitions must be a positive integer."
            )

        tasks = tuple(file_config.tasks)
        languages = tuple(file_config.languages)

        if not tasks:
            raise ExperimentOrchestrationConfigurationError(
                f"No tasks configured for {condition.value}."
            )

        if not languages:
            raise ExperimentOrchestrationConfigurationError(
                f"No languages configured for {condition.value}."
            )

        query_values = self._queries_for_condition(
            condition=condition,
            tasks=tasks,
            query_map=query_map,
        )

        plans: list[ExperimentPlan] = []

        if len(query_values) != len(tasks):
            raise ExperimentOrchestrationConfigurationError(
                f"Expected exactly one query per task for {condition.value}; "
                f"received {len(query_values)} queries for {len(tasks)} tasks."
            )

        for task, query in zip(tasks, query_values, strict=True):
            for language in languages:
                for repetition in range(repetitions_value):
                    experiment_config = self._build_experiment_config(
                        matrix=matrix,
                        file_config=file_config,
                        condition_config=condition_config,
                        task=task,
                        language=language,
                        repetition=repetition,
                        model_name=model_name,
                    )

                    request_id = self._request_id(
                        condition=condition,
                        task=task,
                        language=language,
                        repetition=repetition,
                    )

                    baseline_key = self._baseline_key(
                        condition=condition,
                        task=task,
                        language=language,
                        repetition=repetition,
                    )

                    plans.append(
                        ExperimentPlan(
                            experiment_config=experiment_config,
                            query=query,
                            task=task,
                            request_id=request_id,
                            baseline_generated_code=baseline_map.get(
                                baseline_key,
                                baseline_map.get(task),
                            ),
                            metadata={
                                "matrix_name": matrix.name,
                                "matrix_version": matrix.version,
                                "condition": condition.value,
                                "task": task,
                                "language": language.value,
                                "repetition": repetition,
                                "synthetic_only": matrix.synthetic_only,
                                "authorized_research_only": (
                                    matrix.authorized_research_only
                                ),
                                "dataset_type": (
                                    condition_config.dataset.dataset_type
                                ),
                                "dataset_path": self._dataset_path(
                                    condition_config
                                ),
                                "generation_provider": (
                                    file_config.generation.provider
                                ),
                            },
                        )
                    )

        return plans

    @staticmethod
    def _build_experiment_config(
        *,
        matrix: ExperimentMatrixConfig,
        file_config: ExperimentFileConfig,
        condition_config: Any,
        task: str,
        language: ProgrammingLanguage,
        repetition: int,
        model_name: str,
    ) -> ExperimentConfig:
        """Create the shared ExperimentConfig contract."""
        condition = condition_config.condition

        experiment_id = (
            f"{condition.value}"
            f"__{task}"
            f"__{language.value}"
            f"__rep{repetition + 1}"
        )

        generation_model = file_config.generation.model or model_name

        dataset_path = ExperimentOrchestrator._dataset_path(
            condition_config
        )

        return ExperimentConfig(
            experiment_id=experiment_id,
            condition=condition,
            repository_id=dataset_path,
            task_type=task,
            language=language,
            model_name=generation_model,
            retriever_name=file_config.retrieval.retriever,
            top_k=file_config.retrieval.top_k,
            defense_layers=[
                layer.name
                for layer in condition_config.defense.layers
                if layer.enabled
            ],
            seed=file_config.seed + repetition,
            metadata={
                "dataset_type": (
                    condition_config.dataset.dataset_type
                ),
                "dataset_path": dataset_path,
                "dataset_base_path": (
                    str(condition_config.dataset.base_path)
                    if condition_config.dataset.base_path is not None
                    else None
                ),
                "dataset_output_path": (
                    str(condition_config.dataset.output_path)
                    if condition_config.dataset.output_path is not None
                    else None
                ),
                "poisoning_enabled": (
                    condition_config.poisoning.enabled
                ),
                "poisoning_categories": list(
                    condition_config.poisoning.categories
                ),
                "defense_enabled": condition_config.defense.enabled,
                "defense_fail_closed": (
                    condition_config.defense.fail_closed
                ),
                "build_safe_context": (
                    condition_config.defense.build_safe_context
                ),
                "analyze_generated_code": (
                    condition_config.defense.analyze_generated_code
                ),
                "generation_provider": (
                    file_config.generation.provider
                ),
                "temperature": file_config.generation.temperature,
                "max_tokens": file_config.generation.max_tokens,
                "evaluation": {
                    "calculate_poison_retrieval_rate": (
                        file_config.evaluation
                        .calculate_poison_retrieval_rate
                    ),
                    "calculate_attack_success_rate": (
                        file_config.evaluation
                        .calculate_attack_success_rate
                    ),
                    "calculate_vulnerability_introduction_rate": (
                        file_config.evaluation
                        .calculate_vulnerability_introduction_rate
                    ),
                    "calculate_utility_score": (
                        file_config.evaluation.calculate_utility_score
                    ),
                    "track_generation_changes": (
                        file_config.evaluation.track_generation_changes
                    ),
                    "run_security_analysis": (
                        file_config.evaluation.run_security_analysis
                    ),
                    "track_causal_chain": (
                        file_config.evaluation.track_causal_chain
                    ),
                    "record_defense_decisions": (
                        file_config.evaluation.record_defense_decisions
                    ),
                    "record_blocked_context": (
                        file_config.evaluation.record_blocked_context
                    ),
                },
                "matrix_seed": matrix.seed,
                "repetition": repetition,
            },
        )

    @staticmethod
    def _dataset_path(condition_config: Any) -> str:
        """Return the concrete dataset location for a condition."""
        dataset = condition_config.dataset

        if dataset.path is not None:
            return str(dataset.path)

        if dataset.output_path is not None:
            return str(dataset.output_path)

        if dataset.base_path is not None:
            return str(dataset.base_path)

        raise ExperimentOrchestrationConfigurationError(
            "Dataset configuration has no usable path."
        )

    @staticmethod
    def _queries_for_condition(
        *,
        condition: ExperimentCondition,
        tasks: tuple[str, ...],
        query_map: dict[str, tuple[str, ...]],
    ) -> tuple[str, ...]:
        """Return queries, falling back to deterministic task prompts."""
        values = query_map.get(condition.value)

        if values is None:
            values = query_map.get("*")

        if values is not None:
            if len(values) == 1 and len(tasks) > 1:
                return tuple(values[0] for _ in tasks)
            if len(values) != len(tasks):
                raise ExperimentOrchestrationConfigurationError(
                    f"Expected one query per task for {condition.value}; "
                    f"received {len(values)} queries for {len(tasks)} tasks."
                )
            return values

        return tuple(f"Generate code for task: {task}" for task in tasks)

    @staticmethod
    def _normalize_queries(
        queries: dict[str, Iterable[str]] | None,
    ) -> dict[str, tuple[str, ...]]:
        """Normalize optional query definitions."""
        if queries is None:
            return {}

        if not isinstance(queries, dict):
            raise ExperimentOrchestrationInputError(
                "queries must be a dictionary."
            )

        normalized: dict[str, tuple[str, ...]] = {}

        for key, values in queries.items():
            if not isinstance(key, str) or not key.strip():
                raise ExperimentOrchestrationInputError(
                    "Query-map keys must be non-empty strings."
                )

            if isinstance(values, str):
                values = (values,)

            try:
                materialized = tuple(values)
            except TypeError as exc:
                raise ExperimentOrchestrationInputError(
                    f"Queries for {key!r} must be iterable."
                ) from exc

            if not materialized or any(
                not isinstance(value, str) or not value.strip()
                for value in materialized
            ):
                raise ExperimentOrchestrationInputError(
                    f"Queries for {key!r} must contain non-empty strings."
                )

            normalized[key] = materialized

        return normalized

    @staticmethod
    def _normalize_baselines(
        baseline_outputs: dict[str, str] | None,
    ) -> dict[str, str]:
        """Normalize optional baseline generated-code outputs."""
        if baseline_outputs is None:
            return {}

        if not isinstance(baseline_outputs, dict):
            raise ExperimentOrchestrationInputError(
                "baseline_outputs must be a dictionary."
            )

        normalized: dict[str, str] = {}

        for key, value in baseline_outputs.items():
            if not isinstance(key, str) or not key.strip():
                raise ExperimentOrchestrationInputError(
                    "Baseline-map keys must be non-empty strings."
                )

            if not isinstance(value, str):
                raise ExperimentOrchestrationInputError(
                    f"Baseline output for {key!r} must be a string."
                )

            normalized[key] = value

        return normalized

    @staticmethod
    def _request_id(
        *,
        condition: ExperimentCondition,
        task: str,
        language: ProgrammingLanguage | str,
        repetition: int,
    ) -> str:
        """Create a deterministic request identifier."""
        language_value = (
            language.value
            if isinstance(language, ProgrammingLanguage)
            else str(language)
        )

        return (
            f"request__{condition.value}"
            f"__{task}"
            f"__{language_value}"
            f"__rep{repetition + 1}"
        )


    @staticmethod
    def _baseline_key(
        *,
        condition: ExperimentCondition,
        task: str,
        language: ProgrammingLanguage | str,
        repetition: int,
    ) -> str:
        """Create a deterministic baseline lookup key."""
        language_value = (
            language.value
            if isinstance(language, ProgrammingLanguage)
            else str(language)
        )

        return (
            f"{condition.value}"
            f"__{task}"
            f"__{language_value}"
            f"__rep{repetition + 1}"
        )

    def __repr__(self) -> str:
        return "ExperimentOrchestrator()"
