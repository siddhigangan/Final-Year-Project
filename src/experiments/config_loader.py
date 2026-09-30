"""Experiment configuration loading and validation.

This module loads and validates the JSON configuration files used by
SecureCodeRAG experiments and CodeRAG-PoisonBench.

Infrastructure configuration remains in ``src.config``. This module is
responsible only for experiment definitions and experiment-matrix validation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.models import ExperimentCondition, ProgrammingLanguage


class ExperimentConfigurationError(ValueError):
    """Raised when an experiment configuration is invalid."""


class ExperimentConfigurationInputError(ExperimentConfigurationError):
    """Raised when an experiment configuration cannot be loaded."""


ALLOWED_DATASET_TYPES = frozenset({"clean", "poisoned"})

ALLOWED_GENERATION_PROVIDERS = frozenset({"ollama", "openai"})

ALLOWED_TASKS = frozenset(
    {
        "code_generation",
        "code_completion",
        "bug_fixing",
        "test_generation",
        "api_usage",
        "repository_qa",
    }
)

POISONING_CATEGORIES = (
    "misleading_code",
    "vulnerable_code",
    "false_api_guidance",
    "contradictory_documentation",
    "instruction_like_content",
    "false_repository_conventions",
    "context_manipulation",
)

DEFENSE_LAYER_NAMES = (
    "source_trust_scoring",
    "anomaly_detection",
    "context_validation",
    "instruction_data_separation",
    "static_security_analysis",
)

DEFENSE_LAYER_KEYS = {
    "L1": "source_trust_scoring",
    "L2": "anomaly_detection",
    "L3": "context_validation",
    "L4": "instruction_data_separation",
    "L5": "static_security_analysis",
}


@dataclass(frozen=True)
class DatasetConfig:
    """Dataset configuration for an experiment condition."""

    dataset_type: str
    path: str | None = None
    base_path: str | None = None
    output_path: str | None = None

    def __post_init__(self) -> None:
        if self.dataset_type not in ALLOWED_DATASET_TYPES:
            raise ExperimentConfigurationError(
                f"Unsupported dataset type: {self.dataset_type!r}."
            )

        if self.dataset_type == "clean" and not self.path:
            raise ExperimentConfigurationError(
                "Clean datasets require a non-empty 'path'."
            )

        if self.dataset_type == "poisoned":
            has_direct_path = bool(self.path)
            has_generation_paths = bool(
                self.base_path and self.output_path
            )

            if not has_direct_path and not has_generation_paths:
                raise ExperimentConfigurationError(
                    "Poisoned datasets require either 'path' or both "
                    "'base_path' and 'output_path'."
                )


@dataclass(frozen=True)
class PoisoningConfig:
    """Poisoning configuration for an experiment condition."""

    enabled: bool
    categories: tuple[str, ...] = ()
    synthetic_only: bool = True
    authorized_research_only: bool = True

    def __post_init__(self) -> None:
        unknown = set(self.categories) - set(POISONING_CATEGORIES)

        if unknown:
            raise ExperimentConfigurationError(
                f"Unsupported poisoning categories: {sorted(unknown)}."
            )

        if len(set(self.categories)) != len(self.categories):
            raise ExperimentConfigurationError(
                "Poisoning categories must not contain duplicates."
            )

        if self.enabled:
            if not self.categories:
                raise ExperimentConfigurationError(
                    "Enabled poisoning requires at least one poisoning category."
                )

            if not self.synthetic_only:
                raise ExperimentConfigurationError(
                    "Poisoning experiments must use synthetic-only data."
                )

            if not self.authorized_research_only:
                raise ExperimentConfigurationError(
                    "Poisoning experiments must be marked "
                    "authorized_research_only."
                )

        elif self.categories:
            raise ExperimentConfigurationError(
                "Disabled poisoning cannot define poisoning categories."
            )


@dataclass(frozen=True)
class DefenseLayerConfig:
    """Configuration for one defense layer."""

    name: str
    enabled: bool
    threshold: float | None = None

    def __post_init__(self) -> None:
        if self.name not in DEFENSE_LAYER_NAMES:
            raise ExperimentConfigurationError(
                f"Unsupported defense layer: {self.name!r}."
            )

        if self.threshold is not None and not 0.0 <= self.threshold <= 1.0:
            raise ExperimentConfigurationError(
                f"Defense threshold for {self.name!r} must be between 0 and 1."
            )


@dataclass(frozen=True)
class DefenseConfig:
    """Defense configuration for an experiment condition."""

    enabled: bool
    layers: tuple[DefenseLayerConfig, ...] = ()
    fail_closed: bool = True
    build_safe_context: bool = True
    analyze_generated_code: bool = True

    def __post_init__(self) -> None:
        layer_names = tuple(
            layer.name for layer in self.layers
        )

        if len(set(layer_names)) != len(layer_names):
            raise ExperimentConfigurationError(
                "Defense layers must not contain duplicates."
            )

        if self.enabled:
            missing = set(DEFENSE_LAYER_NAMES) - set(layer_names)

            if missing:
                raise ExperimentConfigurationError(
                    "Enabled defense configuration is missing layers: "
                    f"{sorted(missing)}."
                )

            disabled = [
                layer.name
                for layer in self.layers
                if not layer.enabled
            ]

            if disabled:
                raise ExperimentConfigurationError(
                    "All L1-L5 defense layers must be enabled when defense "
                    f"is enabled. Disabled layers: {disabled}."
                )


@dataclass(frozen=True)
class RetrievalConfig:
    """Retrieval configuration."""

    top_k: int = 5
    retriever: str = "default"

    def __post_init__(self) -> None:
        if self.top_k < 1:
            raise ExperimentConfigurationError(
                "Retrieval 'top_k' must be at least 1."
            )

        if not self.retriever.strip():
            raise ExperimentConfigurationError(
                "Retrieval 'retriever' must not be empty."
            )


@dataclass(frozen=True)
class GenerationConfig:
    """Generation configuration."""

    provider: str = "ollama"
    model: str | None = None
    temperature: float = 0.0
    max_tokens: int = 1024

    def __post_init__(self) -> None:
        if self.provider not in ALLOWED_GENERATION_PROVIDERS:
            raise ExperimentConfigurationError(
                f"Unsupported generation provider: {self.provider!r}."
            )

        if self.model is not None and not self.model.strip():
            raise ExperimentConfigurationError(
                "Generation 'model' must not be empty."
            )

        if self.temperature < 0.0:
            raise ExperimentConfigurationError(
                "Generation 'temperature' must be non-negative."
            )

        if self.max_tokens < 1:
            raise ExperimentConfigurationError(
                "Generation 'max_tokens' must be at least 1."
            )


@dataclass(frozen=True)
class EvaluationConfig:
    """Evaluation configuration."""

    calculate_poison_retrieval_rate: bool = True
    calculate_attack_success_rate: bool = True
    calculate_vulnerability_introduction_rate: bool = True
    calculate_utility_score: bool = True
    track_generation_changes: bool = True
    run_security_analysis: bool = True
    track_causal_chain: bool = True
    record_defense_decisions: bool = False
    record_blocked_context: bool = False


@dataclass(frozen=True)
class ExperimentConditionConfig:
    """Complete configuration for one experiment condition."""

    condition: ExperimentCondition
    dataset: DatasetConfig
    poisoning: PoisoningConfig
    defense: DefenseConfig

    def __post_init__(self) -> None:
        expected_poisoning = self.condition in {
            ExperimentCondition.POISONED_NO_DEFENSE,
            ExperimentCondition.POISONED_DEFENSE,
        }

        expected_defense = self.condition in {
            ExperimentCondition.POISONED_DEFENSE,
            ExperimentCondition.CLEAN_DEFENSE,
        }

        expected_dataset = (
            "poisoned"
            if expected_poisoning
            else "clean"
        )

        if self.poisoning.enabled != expected_poisoning:
            raise ExperimentConfigurationError(
                f"Condition {self.condition.value!r} has inconsistent "
                f"poisoning.enabled={self.poisoning.enabled!r}."
            )

        if self.defense.enabled != expected_defense:
            raise ExperimentConfigurationError(
                f"Condition {self.condition.value!r} has inconsistent "
                f"defense.enabled={self.defense.enabled!r}."
            )

        if self.dataset.dataset_type != expected_dataset:
            raise ExperimentConfigurationError(
                f"Condition {self.condition.value!r} requires the "
                f"{expected_dataset!r} dataset."
            )


@dataclass(frozen=True)
class ExperimentFileConfig:
    """Validated representation of one experiment JSON file."""

    experiment_group: str
    description: str
    version: str
    seed: int
    conditions: tuple[ExperimentConditionConfig, ...]
    tasks: tuple[str, ...]
    languages: tuple[ProgrammingLanguage, ...]
    retrieval: RetrievalConfig
    generation: GenerationConfig
    evaluation: EvaluationConfig
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.experiment_group.strip():
            raise ExperimentConfigurationError(
                "experiment_group must not be empty."
            )

        if not self.description.strip():
            raise ExperimentConfigurationError(
                "description must not be empty."
            )

        if not self.version.strip():
            raise ExperimentConfigurationError(
                "version must not be empty."
            )

        if self.seed < 0:
            raise ExperimentConfigurationError(
                "seed must be non-negative."
            )

        if not self.conditions:
            raise ExperimentConfigurationError(
                "At least one experiment condition is required."
            )

        condition_values = [
            condition.condition
            for condition in self.conditions
        ]

        if len(set(condition_values)) != len(condition_values):
            raise ExperimentConfigurationError(
                "Experiment conditions must not contain duplicates."
            )

        if not self.tasks:
            raise ExperimentConfigurationError(
                "At least one experiment task is required."
            )

        unknown_tasks = set(self.tasks) - set(ALLOWED_TASKS)

        if unknown_tasks:
            raise ExperimentConfigurationError(
                f"Unsupported tasks: {sorted(unknown_tasks)}."
            )

        if not self.languages:
            raise ExperimentConfigurationError(
                "At least one programming language is required."
            )

    @property
    def condition_values(self) -> tuple[ExperimentCondition, ...]:
        """Return configured condition enum values."""

        return tuple(
            condition.condition
            for condition in self.conditions
        )

    @property
    def is_poisoning_experiment(self) -> bool:
        """Return whether poisoning is enabled."""

        return any(
            condition.poisoning.enabled
            for condition in self.conditions
        )

    @property
    def is_defense_experiment(self) -> bool:
        """Return whether defense is enabled."""

        return any(
            condition.defense.enabled
            for condition in self.conditions
        )


@dataclass(frozen=True)
class MatrixConditionConfig:
    """One condition entry from experiment_matrix.json."""

    condition_id: str
    name: str
    condition: ExperimentCondition
    dataset: str
    defense_enabled: bool
    poisoning_enabled: bool
    configuration: str


@dataclass(frozen=True)
class ExperimentMatrixConfig:
    """Validated CodeRAG-PoisonBench experiment matrix."""

    name: str
    version: str
    description: str
    research_scope: str
    synthetic_only: bool
    authorized_research_only: bool
    seed: int
    conditions: tuple[MatrixConditionConfig, ...]
    tasks: tuple[str, ...]
    languages: tuple[ProgrammingLanguage, ...]
    retrieval: RetrievalConfig
    generation: GenerationConfig
    replication: dict[str, Any]
    evaluation: dict[str, Any]
    poisoning_categories: tuple[str, ...]
    comparisons: tuple[dict[str, str], ...]
    output: dict[str, Any]
    metadata: dict[str, Any]

    @property
    def condition_values(self) -> tuple[ExperimentCondition, ...]:
        """Return matrix condition enum values."""

        return tuple(
            condition.condition
            for condition in self.conditions
        )


class ExperimentConfigLoader:
    """Load and validate SecureCodeRAG experiment configurations."""

    def __init__(
        self,
        base_directory: str | Path | None = None,
    ) -> None:
        self._base_directory = (
            Path(base_directory).resolve()
            if base_directory is not None
            else Path.cwd().resolve()
        )

    @property
    def base_directory(self) -> Path:
        """Return the configuration resolution base directory."""

        return self._base_directory

    def load(
        self,
        path: str | Path,
    ) -> ExperimentFileConfig:
        """Load and validate one experiment configuration."""

        file_path = self._resolve_file(path)
        payload = self._read_json(file_path)

        return self._parse_experiment_file(
            payload,
            file_path,
        )

    def load_matrix(
        self,
        path: str | Path,
    ) -> ExperimentMatrixConfig:
        """Load and validate experiment_matrix.json."""

        file_path = self._resolve_file(path)
        payload = self._read_json(file_path)

        matrix = payload.get("experiment_matrix")

        if not isinstance(matrix, dict):
            raise ExperimentConfigurationInputError(
                "Matrix configuration requires an 'experiment_matrix' object."
            )

        return self._parse_matrix(
            payload,
            file_path,
        )

    def load_all(
        self,
        matrix_path: str | Path,
    ) -> tuple[
        ExperimentMatrixConfig,
        dict[str, ExperimentFileConfig],
    ]:
        """Load the matrix and every configuration referenced by it."""

        matrix_file = self._resolve_file(matrix_path)
        matrix = self.load_matrix(matrix_file)

        configurations: dict[str, ExperimentFileConfig] = {}

        for condition in matrix.conditions:
            configuration_path = self._resolve_configuration_reference(
                condition.configuration,
                matrix_file,
            )

            configuration = self.load(configuration_path)

            if condition.condition not in configuration.condition_values:
                raise ExperimentConfigurationError(
                    f"Matrix condition {condition.condition.value!r} is not "
                    f"defined by {configuration_path}."
                )

            configurations[condition.condition.value] = configuration

        return matrix, configurations

    def _resolve_file(
        self,
        path: str | Path,
    ) -> Path:
        candidate = Path(path)

        if not candidate.is_absolute():
            candidate = self._base_directory / candidate

        candidate = candidate.resolve()

        if not candidate.exists():
            raise ExperimentConfigurationInputError(
                f"Configuration file does not exist: {candidate}"
            )

        if not candidate.is_file():
            raise ExperimentConfigurationInputError(
                f"Configuration path is not a file: {candidate}"
            )

        return candidate

    def _resolve_configuration_reference(
        self,
        reference: str,
        matrix_file: Path,
    ) -> Path:
        """Resolve a matrix configuration reference."""

        if not isinstance(reference, str) or not reference.strip():
            raise ExperimentConfigurationInputError(
                "Matrix configuration reference must be a non-empty string."
            )

        reference_path = Path(reference)

        if reference_path.is_absolute():
            candidate = reference_path.resolve()

            if candidate.exists() and candidate.is_file():
                return candidate

            raise ExperimentConfigurationInputError(
                f"Configuration file does not exist: {candidate}"
            )

        project_relative = (
            self._base_directory / reference_path
        ).resolve()

        if project_relative.exists() and project_relative.is_file():
            return project_relative

        matrix_relative = (
            matrix_file.parent / reference_path
        ).resolve()

        if matrix_relative.exists() and matrix_relative.is_file():
            return matrix_relative

        raise ExperimentConfigurationInputError(
            "Configuration reference does not resolve to a file: "
            f"{reference!r}. Checked {project_relative} and "
            f"{matrix_relative}."
        )

    @staticmethod
    def _read_json(
        path: Path,
    ) -> dict[str, Any]:
        try:
            with path.open(
                "r",
                encoding="utf-8",
            ) as handle:
                payload = json.load(handle)
        except json.JSONDecodeError as exc:
            raise ExperimentConfigurationInputError(
                f"Invalid JSON in {path}: {exc}"
            ) from exc
        except OSError as exc:
            raise ExperimentConfigurationInputError(
                f"Unable to read configuration file {path}: {exc}"
            ) from exc

        if not isinstance(payload, dict):
            raise ExperimentConfigurationInputError(
                f"Configuration root must be a JSON object: {path}"
            )

        return payload

    def _parse_experiment_file(
        self,
        payload: dict[str, Any],
        path: Path,
    ) -> ExperimentFileConfig:
        experiment_group = self._required_string(
            payload,
            "experiment_group",
            path,
        )

        description = self._required_string(
            payload,
            "description",
            path,
        )

        version = self._required_string(
            payload,
            "version",
            path,
        )

        seed = self._non_negative_int(
            payload,
            "seed",
            path,
        )

        conditions_raw = self._required_list(
            payload,
            "conditions",
            path,
        )

        conditions = tuple(
            self._parse_condition(
                item,
                path,
                index,
            )
            for index, item in enumerate(conditions_raw)
        )

        tasks = self._parse_tasks(
            payload.get("tasks"),
            path,
        )

        languages = self._parse_languages(
            payload.get("languages"),
            path,
        )

        retrieval = self._parse_retrieval(
            payload.get("retrieval"),
            path,
        )

        generation = self._parse_generation(
            payload.get("generation"),
            path,
        )

        evaluation = self._parse_evaluation(
            payload.get("evaluation"),
            path,
        )

        metadata = payload.get("metadata", {})

        if not isinstance(metadata, dict):
            raise ExperimentConfigurationError(
                f"'metadata' must be an object in {path}."
            )

        return ExperimentFileConfig(
            experiment_group=experiment_group,
            description=description,
            version=version,
            seed=seed,
            conditions=conditions,
            tasks=tasks,
            languages=languages,
            retrieval=retrieval,
            generation=generation,
            evaluation=evaluation,
            metadata=dict(metadata),
        )

    def _parse_condition(
        self,
        value: Any,
        path: Path,
        index: int,
    ) -> ExperimentConditionConfig:
        if not isinstance(value, dict):
            raise ExperimentConfigurationError(
                f"conditions[{index}] must be an object in {path}."
            )

        condition_value = self._required_string(
            value,
            "condition",
            path,
            prefix=f"conditions[{index}].",
        )

        try:
            condition = ExperimentCondition(condition_value)
        except ValueError as exc:
            raise ExperimentConfigurationError(
                f"Unsupported condition {condition_value!r} in {path}."
            ) from exc

        dataset = self._parse_dataset(
            value.get("dataset"),
            path,
            index,
        )

        poisoning = self._parse_poisoning(
            value.get("poisoning"),
            path,
            index,
        )

        defense = self._parse_defense(
            value.get("defense"),
            path,
            index,
        )

        return ExperimentConditionConfig(
            condition=condition,
            dataset=dataset,
            poisoning=poisoning,
            defense=defense,
        )

    def _parse_dataset(
        self,
        value: Any,
        path: Path,
        index: int,
    ) -> DatasetConfig:
        if not isinstance(value, dict):
            raise ExperimentConfigurationError(
                f"conditions[{index}].dataset must be an object in {path}."
            )

        dataset_type = value.get("type")

        if not isinstance(dataset_type, str):
            raise ExperimentConfigurationError(
                f"conditions[{index}].dataset.type must be a string."
            )

        path_value = value.get("path")
        base_path = value.get("base_path")
        output_path = value.get("output_path")

        for name, item in (
            ("path", path_value),
            ("base_path", base_path),
            ("output_path", output_path),
        ):
            if item is not None and not isinstance(item, str):
                raise ExperimentConfigurationError(
                    f"conditions[{index}].dataset.{name} must be a string."
                )

        return DatasetConfig(
            dataset_type=dataset_type,
            path=Path(path_value) if path_value is not None else (
                Path(output_path) if output_path is not None else None
            ),
            base_path=Path(base_path) if base_path is not None else None,
            output_path=Path(output_path) if output_path is not None else None,
        )

    def _parse_poisoning(
        self,
        value: Any,
        path: Path,
        index: int,
    ) -> PoisoningConfig:
        if not isinstance(value, dict):
            raise ExperimentConfigurationError(
                f"conditions[{index}].poisoning must be an object in {path}."
            )

        enabled = self._required_bool(
            value,
            "enabled",
            path,
            prefix=f"conditions[{index}].poisoning.",
        )

        categories = value.get("categories", [])

        if not isinstance(categories, list):
            raise ExperimentConfigurationError(
                f"conditions[{index}].poisoning.categories must be a list."
            )

        if not all(isinstance(item, str) for item in categories):
            raise ExperimentConfigurationError(
                f"conditions[{index}].poisoning.categories must contain "
                "only strings."
            )

        synthetic_only = self._optional_bool(
            value,
            "synthetic_only",
            True,
            path,
        )

        authorized_research_only = self._optional_bool(
            value,
            "authorized_research_only",
            True,
            path,
        )

        return PoisoningConfig(
            enabled=enabled,
            categories=tuple(categories),
            synthetic_only=synthetic_only,
            authorized_research_only=authorized_research_only,
        )

    def _parse_defense(
        self,
        value: Any,
        path: Path,
        index: int,
    ) -> DefenseConfig:
        if not isinstance(value, dict):
            raise ExperimentConfigurationError(
                f"conditions[{index}].defense must be an object in {path}."
            )

        enabled = self._required_bool(
            value,
            "enabled",
            path,
            prefix=f"conditions[{index}].defense.",
        )

        layers_value = value.get("layers", [])

        if isinstance(layers_value, list):
            if layers_value:
                raise ExperimentConfigurationError(
                    f"conditions[{index}].defense.layers must be an object "
                    "when layers are configured."
                )

            layers: tuple[DefenseLayerConfig, ...] = ()

        elif isinstance(layers_value, dict):
            layers = tuple(
                self._parse_defense_layer(
                    key,
                    layer_value,
                    path,
                    index,
                )
                for key, layer_value in layers_value.items()
            )

        else:
            raise ExperimentConfigurationError(
                f"conditions[{index}].defense.layers must be an object "
                "or an empty list."
            )

        return DefenseConfig(
            enabled=enabled,
            layers=layers,
            fail_closed=self._optional_bool(
                value,
                "fail_closed",
                True,
                path,
            ),
            build_safe_context=self._optional_bool(
                value,
                "build_safe_context",
                True,
                path,
            ),
            analyze_generated_code=self._optional_bool(
                value,
                "analyze_generated_code",
                True,
                path,
            ),
        )

    def _parse_defense_layer(
        self,
        key: str,
        value: Any,
        path: Path,
        index: int,
    ) -> DefenseLayerConfig:
        """Parse one L1-L5 defense layer."""

        if not isinstance(value, dict):
            raise ExperimentConfigurationError(
                f"conditions[{index}].defense.layers[{key!r}] "
                "must be an object."
            )

        if key not in DEFENSE_LAYER_KEYS:
            raise ExperimentConfigurationError(
                f"Unsupported defense layer key: {key!r}. "
                "Expected L1, L2, L3, L4, or L5."
            )

        expected_name = DEFENSE_LAYER_KEYS[key]
        configured_name = value.get("name")

        if configured_name is None:
            configured_name = expected_name

        if not isinstance(configured_name, str):
            raise ExperimentConfigurationError(
                f"conditions[{index}].defense.layers[{key!r}].name "
                "must be a string."
            )

        if configured_name != expected_name:
            raise ExperimentConfigurationError(
                f"Defense layer {key} must use name "
                f"{expected_name!r}, got {configured_name!r}."
            )

        enabled = value.get("enabled")

        if not isinstance(enabled, bool):
            raise ExperimentConfigurationError(
                f"conditions[{index}].defense.layers[{key!r}].enabled "
                "must be boolean."
            )

        threshold = value.get("threshold")

        if threshold is not None and (
            isinstance(threshold, bool)
            or not isinstance(threshold, (int, float))
        ):
            raise ExperimentConfigurationError(
                f"conditions[{index}].defense.layers[{key!r}]"
                ".threshold must be numeric."
            )

        return DefenseLayerConfig(
            name=configured_name,
            enabled=enabled,
            threshold=(
                float(threshold)
                if threshold is not None
                else None
            ),
        )

    def _parse_tasks(
        self,
        value: Any,
        path: Path,
    ) -> tuple[str, ...]:
        if not isinstance(value, list) or not value:
            raise ExperimentConfigurationError(
                f"'tasks' must be a non-empty list in {path}."
            )

        if not all(isinstance(item, str) for item in value):
            raise ExperimentConfigurationError(
                f"'tasks' must contain only strings in {path}."
            )

        tasks = tuple(value)

        if len(set(tasks)) != len(tasks):
            raise ExperimentConfigurationError(
                f"'tasks' contains duplicate entries in {path}."
            )

        unknown = set(tasks) - set(ALLOWED_TASKS)

        if unknown:
            raise ExperimentConfigurationError(
                f"Unsupported tasks in {path}: {sorted(unknown)}."
            )

        return tasks

    def _parse_languages(
        self,
        value: Any,
        path: Path,
    ) -> tuple[ProgrammingLanguage, ...]:
        if not isinstance(value, list) or not value:
            raise ExperimentConfigurationError(
                f"'languages' must be a non-empty list in {path}."
            )

        if not all(isinstance(item, str) for item in value):
            raise ExperimentConfigurationError(
                f"'languages' must contain only strings in {path}."
            )

        languages: list[ProgrammingLanguage] = []

        for language_value in value:
            try:
                language = ProgrammingLanguage(
                    language_value.lower()
                )
            except ValueError as exc:
                raise ExperimentConfigurationError(
                    f"Unsupported programming language "
                    f"{language_value!r} in {path}."
                ) from exc

            languages.append(language)

        if len(set(languages)) != len(languages):
            raise ExperimentConfigurationError(
                f"'languages' contains duplicate entries in {path}."
            )

        return tuple(languages)

    def _parse_retrieval(
        self,
        value: Any,
        path: Path,
    ) -> RetrievalConfig:
        if not isinstance(value, dict):
            raise ExperimentConfigurationError(
                f"'retrieval' must be an object in {path}."
            )

        top_k = value.get("top_k", 5)
        retriever = value.get("retriever", "default")

        if isinstance(top_k, bool) or not isinstance(top_k, int):
            raise ExperimentConfigurationError(
                f"'retrieval.top_k' must be an integer in {path}."
            )

        if not isinstance(retriever, str):
            raise ExperimentConfigurationError(
                f"'retrieval.retriever' must be a string in {path}."
            )

        return RetrievalConfig(
            top_k=top_k,
            retriever=retriever,
        )

    def _parse_generation(
        self,
        value: Any,
        path: Path,
    ) -> GenerationConfig:
        if not isinstance(value, dict):
            raise ExperimentConfigurationError(
                f"'generation' must be an object in {path}."
            )

        provider = value.get("provider", "ollama")
        model = value.get("model")
        temperature = value.get("temperature", 0.0)
        max_tokens = value.get("max_tokens", 1024)

        if not isinstance(provider, str):
            raise ExperimentConfigurationError(
                f"'generation.provider' must be a string in {path}."
            )

        if model is not None and not isinstance(model, str):
            raise ExperimentConfigurationError(
                f"'generation.model' must be a string in {path}."
            )

        if isinstance(temperature, bool) or not isinstance(
            temperature,
            (int, float),
        ):
            raise ExperimentConfigurationError(
                f"'generation.temperature' must be numeric in {path}."
            )

        if isinstance(max_tokens, bool) or not isinstance(
            max_tokens,
            int,
        ):
            raise ExperimentConfigurationError(
                f"'generation.max_tokens' must be an integer in {path}."
            )

        return GenerationConfig(
            provider=provider,
            model=model,
            temperature=float(temperature),
            max_tokens=max_tokens,
        )

    def _parse_evaluation(
        self,
        value: Any,
        path: Path,
    ) -> EvaluationConfig:
        if value is None:
            return EvaluationConfig()

        if not isinstance(value, dict):
            raise ExperimentConfigurationError(
                f"'evaluation' must be an object in {path}."
            )

        defaults = EvaluationConfig()

        return EvaluationConfig(
            calculate_poison_retrieval_rate=self._evaluation_bool(
                value,
                "calculate_poison_retrieval_rate",
                defaults.calculate_poison_retrieval_rate,
                path,
            ),
            calculate_attack_success_rate=self._evaluation_bool(
                value,
                "calculate_attack_success_rate",
                defaults.calculate_attack_success_rate,
                path,
            ),
            calculate_vulnerability_introduction_rate=self._evaluation_bool(
                value,
                "calculate_vulnerability_introduction_rate",
                defaults.calculate_vulnerability_introduction_rate,
                path,
            ),
            calculate_utility_score=self._evaluation_bool(
                value,
                "calculate_utility_score",
                defaults.calculate_utility_score,
                path,
            ),
            track_generation_changes=self._evaluation_bool(
                value,
                "track_generation_changes",
                defaults.track_generation_changes,
                path,
            ),
            run_security_analysis=self._evaluation_bool(
                value,
                "run_security_analysis",
                defaults.run_security_analysis,
                path,
            ),
            track_causal_chain=self._evaluation_bool(
                value,
                "track_causal_chain",
                defaults.track_causal_chain,
                path,
            ),
            record_defense_decisions=self._evaluation_bool(
                value,
                "record_defense_decisions",
                defaults.record_defense_decisions,
                path,
            ),
            record_blocked_context=self._evaluation_bool(
                value,
                "record_blocked_context",
                defaults.record_blocked_context,
                path,
            ),
        )

    def _parse_matrix(
        self,
        payload: dict[str, Any],
        path: Path,
    ) -> ExperimentMatrixConfig:
        matrix = payload["experiment_matrix"]

        name = self._required_string(
            matrix,
            "name",
            path,
            prefix="experiment_matrix.",
        )

        version = self._required_string(
            matrix,
            "version",
            path,
            prefix="experiment_matrix.",
        )

        description = self._required_string(
            matrix,
            "description",
            path,
            prefix="experiment_matrix.",
        )

        research_scope = self._required_string(
            matrix,
            "research_scope",
            path,
            prefix="experiment_matrix.",
        )

        synthetic_only = self._required_bool(
            matrix,
            "synthetic_only",
            path,
            prefix="experiment_matrix.",
        )

        authorized_research_only = self._required_bool(
            matrix,
            "authorized_research_only",
            path,
            prefix="experiment_matrix.",
        )

        seed = self._non_negative_int(
            matrix,
            "seed",
            path,
            prefix="experiment_matrix.",
        )

        if not synthetic_only or not authorized_research_only:
            raise ExperimentConfigurationError(
                "The experiment matrix must be synthetic-only and "
                "authorized-research-only."
            )

        conditions_raw = self._required_list(
            payload,
            "conditions",
            path,
        )

        conditions = tuple(
            self._parse_matrix_condition(
                item,
                path,
                index,
            )
            for index, item in enumerate(conditions_raw)
        )

        if len({item.condition for item in conditions}) != len(conditions):
            raise ExperimentConfigurationError(
                "Experiment matrix contains duplicate conditions."
            )

        tasks = self._parse_tasks(
            payload.get("tasks"),
            path,
        )

        languages = self._parse_languages(
            payload.get("languages"),
            path,
        )

        retrieval = self._parse_retrieval(
            payload.get("retrieval"),
            path,
        )

        generation = self._parse_generation(
            payload.get("generation"),
            path,
        )

        replication = payload.get("replication", {})
        evaluation = payload.get("evaluation", {})
        output = payload.get("output", {})
        metadata = payload.get("metadata", {})

        for field_name, field_value in (
            ("replication", replication),
            ("evaluation", evaluation),
            ("output", output),
            ("metadata", metadata),
        ):
            if not isinstance(field_value, dict):
                raise ExperimentConfigurationError(
                    f"'{field_name}' must be an object in {path}."
                )

        self._validate_replication(
            replication,
            path,
        )

        poisoning_categories = (
            self._parse_matrix_poisoning_categories(
                payload.get("poisoning_categories"),
                path,
            )
        )

        comparisons = self._parse_comparisons(
            payload.get("comparisons"),
            {item.condition.value for item in conditions},
            path,
        )

        return ExperimentMatrixConfig(
            name=name,
            version=version,
            description=description,
            research_scope=research_scope,
            synthetic_only=synthetic_only,
            authorized_research_only=authorized_research_only,
            seed=seed,
            conditions=conditions,
            tasks=tasks,
            languages=languages,
            retrieval=retrieval,
            generation=generation,
            replication=dict(replication),
            evaluation=dict(evaluation),
            poisoning_categories=poisoning_categories,
            comparisons=comparisons,
            output=dict(output),
            metadata=dict(metadata),
        )

    def _parse_matrix_condition(
        self,
        value: Any,
        path: Path,
        index: int,
    ) -> MatrixConditionConfig:
        if not isinstance(value, dict):
            raise ExperimentConfigurationError(
                f"conditions[{index}] in {path} must be an object."
            )

        required_fields = (
            "id",
            "name",
            "condition",
            "dataset",
            "defense_enabled",
            "poisoning_enabled",
            "configuration",
        )

        for field_name in required_fields:
            if field_name not in value:
                raise ExperimentConfigurationError(
                    f"conditions[{index}] is missing '{field_name}'."
                )

        condition_id = value["id"]
        name = value["name"]
        condition_value = value["condition"]
        dataset = value["dataset"]
        defense_enabled = value["defense_enabled"]
        poisoning_enabled = value["poisoning_enabled"]
        configuration = value["configuration"]

        string_values = (
            condition_id,
            name,
            condition_value,
            dataset,
            configuration,
        )

        if not all(isinstance(item, str) for item in string_values):
            raise ExperimentConfigurationError(
                f"conditions[{index}] contains invalid string fields."
            )

        if not isinstance(defense_enabled, bool):
            raise ExperimentConfigurationError(
                f"conditions[{index}].defense_enabled must be boolean."
            )

        if not isinstance(poisoning_enabled, bool):
            raise ExperimentConfigurationError(
                f"conditions[{index}].poisoning_enabled must be boolean."
            )

        try:
            condition = ExperimentCondition(condition_value)
        except ValueError as exc:
            raise ExperimentConfigurationError(
                f"Unsupported matrix condition {condition_value!r}."
            ) from exc

        expected_poisoning = condition in {
            ExperimentCondition.POISONED_NO_DEFENSE,
            ExperimentCondition.POISONED_DEFENSE,
        }

        expected_defense = condition in {
            ExperimentCondition.POISONED_DEFENSE,
            ExperimentCondition.CLEAN_DEFENSE,
        }

        if poisoning_enabled != expected_poisoning:
            raise ExperimentConfigurationError(
                f"Matrix condition {condition_value!r} has inconsistent "
                "poisoning_enabled value."
            )

        if defense_enabled != expected_defense:
            raise ExperimentConfigurationError(
                f"Matrix condition {condition_value!r} has inconsistent "
                "defense_enabled value."
            )

        expected_dataset = (
            "poisoned"
            if expected_poisoning
            else "clean"
        )

        if dataset != expected_dataset:
            raise ExperimentConfigurationError(
                f"Matrix condition {condition_value!r} requires dataset "
                f"{expected_dataset!r}."
            )

        return MatrixConditionConfig(
            condition_id=condition_id,
            name=name,
            condition=condition,
            dataset=dataset,
            defense_enabled=defense_enabled,
            poisoning_enabled=poisoning_enabled,
            configuration=configuration,
        )

    @staticmethod
    def _validate_replication(
        replication: dict[str, Any],
        path: Path,
    ) -> None:
        repetitions = replication.get("repetitions")

        if repetitions is not None:
            if isinstance(repetitions, bool) or not isinstance(
                repetitions,
                int,
            ):
                raise ExperimentConfigurationError(
                    f"'replication.repetitions' must be an integer in {path}."
                )

            if repetitions < 1:
                raise ExperimentConfigurationError(
                    "'replication.repetitions' must be at least 1."
                )

        independent_runs = replication.get("independent_runs")

        if independent_runs is not None and not isinstance(
            independent_runs,
            bool,
        ):
            raise ExperimentConfigurationError(
                f"'replication.independent_runs' must be boolean in {path}."
            )

        seed_strategy = replication.get("seed_strategy")

        if seed_strategy is not None and (
            not isinstance(seed_strategy, str)
            or not seed_strategy.strip()
        ):
            raise ExperimentConfigurationError(
                f"'replication.seed_strategy' must be a non-empty string "
                f"in {path}."
            )

    @staticmethod
    def _parse_matrix_poisoning_categories(
        value: Any,
        path: Path,
    ) -> tuple[str, ...]:
        if value is None:
            return ()

        if not isinstance(value, list):
            raise ExperimentConfigurationError(
                f"'poisoning_categories' must be a list in {path}."
            )

        if not all(isinstance(item, str) for item in value):
            raise ExperimentConfigurationError(
                f"'poisoning_categories' must contain only strings in {path}."
            )

        categories = tuple(value)

        unknown = set(categories) - set(POISONING_CATEGORIES)

        if unknown:
            raise ExperimentConfigurationError(
                f"Unsupported poisoning categories in {path}: "
                f"{sorted(unknown)}."
            )

        if len(set(categories)) != len(categories):
            raise ExperimentConfigurationError(
                f"'poisoning_categories' contains duplicates in {path}."
            )

        return categories

    @staticmethod
    def _parse_comparisons(
        value: Any,
        condition_values: set[str],
        path: Path,
    ) -> tuple[dict[str, str], ...]:
        if value is None:
            return ()

        if not isinstance(value, list):
            raise ExperimentConfigurationError(
                f"'comparisons' must be a list in {path}."
            )

        comparisons: list[dict[str, str]] = []

        for index, item in enumerate(value):
            if not isinstance(item, dict):
                raise ExperimentConfigurationError(
                    f"comparisons[{index}] must be an object in {path}."
                )

            name = item.get("name")
            baseline = item.get("baseline")
            comparison = item.get("comparison")

            if not all(
                isinstance(field, str)
                for field in (
                    name,
                    baseline,
                    comparison,
                )
            ):
                raise ExperimentConfigurationError(
                    f"comparisons[{index}] requires string name, baseline, "
                    "and comparison fields."
                )

            if baseline not in condition_values:
                raise ExperimentConfigurationError(
                    f"comparisons[{index}] references unknown baseline "
                    f"{baseline!r}."
                )

            if comparison not in condition_values:
                raise ExperimentConfigurationError(
                    f"comparisons[{index}] references unknown comparison "
                    f"{comparison!r}."
                )

            comparisons.append(
                {
                    "name": name,
                    "baseline": baseline,
                    "comparison": comparison,
                }
            )

        return tuple(comparisons)

    @staticmethod
    def _required_string(
        mapping: dict[str, Any],
        key: str,
        path: Path,
        prefix: str = "",
    ) -> str:
        value = mapping.get(key)

        if not isinstance(value, str) or not value.strip():
            raise ExperimentConfigurationError(
                f"{prefix}{key!r} must be a non-empty string in {path}."
            )

        return value

    @staticmethod
    def _required_bool(
        mapping: dict[str, Any],
        key: str,
        path: Path,
        prefix: str = "",
    ) -> bool:
        value = mapping.get(key)

        if not isinstance(value, bool):
            raise ExperimentConfigurationError(
                f"{prefix}{key!r} must be boolean in {path}."
            )

        return value

    @staticmethod
    def _optional_bool(
        mapping: dict[str, Any],
        key: str,
        default: bool,
        path: Path,
    ) -> bool:
        value = mapping.get(key, default)

        if not isinstance(value, bool):
            raise ExperimentConfigurationError(
                f"'{key}' must be boolean in {path}."
            )

        return value

    @staticmethod
    def _evaluation_bool(
        mapping: dict[str, Any],
        key: str,
        default: bool,
        path: Path,
    ) -> bool:
        value = mapping.get(key, default)

        if not isinstance(value, bool):
            raise ExperimentConfigurationError(
                f"'evaluation.{key}' must be boolean in {path}."
            )

        return value

    @staticmethod
    def _non_negative_int(
        mapping: dict[str, Any],
        key: str,
        path: Path,
        prefix: str = "",
    ) -> int:
        value = mapping.get(key)

        if isinstance(value, bool) or not isinstance(value, int):
            raise ExperimentConfigurationError(
                f"{prefix}{key!r} must be an integer in {path}."
            )

        if value < 0:
            raise ExperimentConfigurationError(
                f"{prefix}{key!r} must be non-negative in {path}."
            )

        return value

    @staticmethod
    def _required_list(
        mapping: dict[str, Any],
        key: str,
        path: Path,
    ) -> list[Any]:
        value = mapping.get(key)

        if not isinstance(value, list) or not value:
            raise ExperimentConfigurationError(
                f"'{key}' must be a non-empty list in {path}."
            )

        return value


__all__ = [
    "ALLOWED_DATASET_TYPES",
    "ALLOWED_GENERATION_PROVIDERS",
    "ALLOWED_TASKS",
    "DEFENSE_LAYER_NAMES",
    "DatasetConfig",
    "DefenseConfig",
    "DefenseLayerConfig",
    "EvaluationConfig",
    "ExperimentConditionConfig",
    "ExperimentConfigLoader",
    "ExperimentConfigurationError",
    "ExperimentConfigurationInputError",
    "ExperimentFileConfig",
    "ExperimentMatrixConfig",
    "GenerationConfig",
    "MatrixConditionConfig",
    "PoisoningConfig",
    "RetrievalConfig",
]