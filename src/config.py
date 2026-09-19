"""Configuration management for the SecureCodeRAG project.

Configuration is loaded from a JSON file and can be overridden by
environment variables. The resulting AppConfig object provides a single,
typed configuration contract for the rest of the application.

Configuration precedence:

    dataclass defaults
        ↓
    JSON configuration
        ↓
    environment variables

Environment variables therefore have the highest priority.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


class ConfigurationError(ValueError):
    """Raised when SecureCodeRAG configuration is invalid."""


@dataclass(slots=True)
class AppConfig:
    """Application-wide SecureCodeRAG configuration."""

    # ------------------------------------------------------------------
    # Application
    # ------------------------------------------------------------------

    project_name: str = "SecureCodeRAG"
    version: str = "0.1.0"

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    log_level: str = "INFO"
    log_file: str = "logs/app.log"

    # ------------------------------------------------------------------
    # Ingestion / chunking
    # ------------------------------------------------------------------

    chunk_size: int = 512
    chunk_overlap: int = 64

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    top_k: int = 5
    reranking_enabled: bool = False

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------

    embedding_provider: str = "huggingface"
    embedding_model: str = "microsoft/unixcoder-base"

    # ------------------------------------------------------------------
    # Vector store
    # ------------------------------------------------------------------

    vectorstore_provider: str = "faiss"
    vectorstore_path: str = "data/processed/faiss_index"

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    code_llm_provider: str = "ollama"
    code_llm: str = "deepseek-coder:6.7b"
    code_llm_base_url: str = "http://localhost:11434"

    # Optional hosted fallback.
    hosted_llm_provider: str = "openai"
    hosted_llm_model: str = "gpt-4o-mini"

    # ------------------------------------------------------------------
    # Data paths
    # ------------------------------------------------------------------

    data_clean_dir: str = "data/clean"
    data_poisoned_dir: str = "data/poisoned"
    data_processed_dir: str = "data/processed"
    benchmark_dir: str = "data/benchmarks"

    # ------------------------------------------------------------------
    # Results / experiments
    # ------------------------------------------------------------------

    results_dir: str = "results"
    experiments_dir: str = "experiments"
    seed: int = 42

    # ------------------------------------------------------------------
    # Poisoning
    # ------------------------------------------------------------------

    poisoning_enabled: bool = False
    poisoning_rate: float = 0.10

    # ------------------------------------------------------------------
    # Defense
    # ------------------------------------------------------------------

    defense_enabled: bool = True

    defense_trust_scoring: bool = True
    defense_anomaly_detection: bool = True
    defense_context_validation: bool = True
    defense_instruction_separation: bool = True
    defense_static_analysis: bool = True

    # ------------------------------------------------------------------
    # Security analysis
    # ------------------------------------------------------------------

    static_analysis_enabled: bool = True
    semgrep_enabled: bool = False

    # ------------------------------------------------------------------
    # Experiment execution
    # ------------------------------------------------------------------

    experiment_repetitions: int = 1
    save_intermediate_results: bool = True

    # Additional extensibility configuration.
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""

        if not self.project_name.strip():
            raise ConfigurationError("project_name cannot be empty.")

        if not self.version.strip():
            raise ConfigurationError("version cannot be empty.")

        if not self.log_level.strip():
            raise ConfigurationError("log_level cannot be empty.")

        if self.chunk_size <= 0:
            raise ConfigurationError("chunk_size must be greater than zero.")

        if self.chunk_overlap < 0:
            raise ConfigurationError("chunk_overlap cannot be negative.")

        if self.chunk_overlap >= self.chunk_size:
            raise ConfigurationError(
                "chunk_overlap must be smaller than chunk_size."
            )

        if self.top_k <= 0:
            raise ConfigurationError("top_k must be greater than zero.")

        if not self.embedding_provider.strip():
            raise ConfigurationError(
                "embedding_provider cannot be empty."
            )

        if not self.embedding_model.strip():
            raise ConfigurationError("embedding_model cannot be empty.")

        if not self.vectorstore_provider.strip():
            raise ConfigurationError(
                "vectorstore_provider cannot be empty."
            )

        if not self.vectorstore_path.strip():
            raise ConfigurationError(
                "vectorstore_path cannot be empty."
            )

        if not self.code_llm_provider.strip():
            raise ConfigurationError(
                "code_llm_provider cannot be empty."
            )

        if not self.code_llm.strip():
            raise ConfigurationError("code_llm cannot be empty.")

        if not self.code_llm_base_url.strip():
            raise ConfigurationError(
                "code_llm_base_url cannot be empty."
            )

        if not self.hosted_llm_provider.strip():
            raise ConfigurationError(
                "hosted_llm_provider cannot be empty."
            )

        if not self.hosted_llm_model.strip():
            raise ConfigurationError(
                "hosted_llm_model cannot be empty."
            )

        if self.seed < 0:
            raise ConfigurationError("seed cannot be negative.")

        if not 0.0 <= self.poisoning_rate <= 1.0:
            raise ConfigurationError(
                "poisoning_rate must be between 0.0 and 1.0."
            )

        if self.experiment_repetitions <= 0:
            raise ConfigurationError(
                "experiment_repetitions must be greater than zero."
            )

    def to_dict(self) -> dict[str, Any]:
        """Return the configuration as a dictionary."""
        return asdict(self)

    def resolve_path(self, path_value: str) -> Path:
        """Resolve a project-relative path.

        Absolute paths are returned unchanged. Relative paths are resolved
        relative to the project root inferred from this source file.
        """

        path = Path(path_value)

        if path.is_absolute():
            return path

        project_root = Path(__file__).resolve().parent.parent
        return project_root / path

    def resolved_paths(self) -> dict[str, Path]:
        """Return important project paths as pathlib.Path objects."""

        return {
            "log_file": self.resolve_path(self.log_file),
            "clean_data": self.resolve_path(self.data_clean_dir),
            "poisoned_data": self.resolve_path(self.data_poisoned_dir),
            "processed_data": self.resolve_path(self.data_processed_dir),
            "benchmark": self.resolve_path(self.benchmark_dir),
            "vectorstore": self.resolve_path(self.vectorstore_path),
            "results": self.resolve_path(self.results_dir),
            "experiments": self.resolve_path(self.experiments_dir),
        }


def _load_dotenv() -> None:
    """Load .env when python-dotenv is installed.

    Configuration remains functional without python-dotenv.
    """

    try:
        from dotenv import load_dotenv
    except ImportError:
        return

    load_dotenv()


def _get_environment_value(
    *names: str,
    default: Any = None,
) -> Any:
    """Return the first configured environment variable.

    Multiple names are supported to preserve backward compatibility.
    """

    for name in names:
        value = os.getenv(name)

        if value is not None:
            return value

    return default


def _parse_int(name: str, value: Any) -> int:
    """Parse an integer configuration value."""

    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(
            f"{name} must be an integer; received {value!r}."
        ) from exc


def _parse_float(name: str, value: Any) -> float:
    """Parse a floating-point configuration value."""

    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(
            f"{name} must be a number; received {value!r}."
        ) from exc


def _parse_bool(name: str, value: Any) -> bool:
    """Parse a boolean configuration value."""

    if isinstance(value, bool):
        return value

    normalized = str(value).strip().lower()

    if normalized in {"1", "true", "yes", "y", "on"}:
        return True

    if normalized in {"0", "false", "no", "n", "off"}:
        return False

    raise ConfigurationError(
        f"{name} must be a boolean value; received {value!r}."
    )


def _nested(
    data: dict[str, Any],
    section: str,
) -> dict[str, Any]:
    """Safely retrieve a configuration section."""

    value = data.get(section, {})

    if value is None:
        return {}

    if not isinstance(value, dict):
        raise ConfigurationError(
            f"Configuration section '{section}' must be an object."
        )

    return value


def _load_json_config(config_path: Path) -> dict[str, Any]:
    """Load and validate a JSON configuration file."""

    if not config_path.exists():
        raise ConfigurationError(
            f"Configuration file does not exist: {config_path}"
        )

    if not config_path.is_file():
        raise ConfigurationError(
            f"Configuration path is not a file: {config_path}"
        )

    try:
        with config_path.open("r", encoding="utf-8") as config_file:
            data = json.load(config_file)
    except json.JSONDecodeError as exc:
        raise ConfigurationError(
            f"Invalid JSON configuration: {config_path}"
        ) from exc
    except OSError as exc:
        raise ConfigurationError(
            f"Unable to read configuration file: {config_path}"
        ) from exc

    if not isinstance(data, dict):
        raise ConfigurationError(
            "Root configuration object must be a JSON object."
        )

    return data


def load_config(
    config_path: str | Path | None = None,
) -> AppConfig:
    """Load SecureCodeRAG configuration.

    Precedence:

        1. dataclass defaults
        2. JSON configuration
        3. environment variables

    Environment variables have the highest priority.
    """

    _load_dotenv()

    if config_path is None:
        config_path = Path("configs/default_config.json")

    config_path = Path(config_path)

    if not config_path.is_absolute():
        project_root = Path(__file__).resolve().parent.parent
        config_path = project_root / config_path

    data = _load_json_config(config_path)

    logging_config = _nested(data, "logging")
    pipeline_config = _nested(data, "pipeline")
    models_config = _nested(data, "models")
    paths_config = _nested(data, "paths")
    experiment_config = _nested(data, "experiment")
    defense_config = _nested(data, "defense")
    security_config = _nested(data, "security")
    poisoning_config = _nested(data, "poisoning")
    generation_config = _nested(data, "generation")
    embeddings_config = _nested(data, "embeddings")
    vectorstore_config = _nested(data, "vectorstore")

    chunk_size = _parse_int(
        "CHUNK_SIZE",
        _get_environment_value(
            "CHUNK_SIZE",
            default=pipeline_config.get("chunk_size", 512),
        ),
    )

    chunk_overlap = _parse_int(
        "CHUNK_OVERLAP",
        _get_environment_value(
            "CHUNK_OVERLAP",
            default=pipeline_config.get("chunk_overlap", 64),
        ),
    )

    top_k = _parse_int(
        "TOP_K",
        _get_environment_value(
            "TOP_K",
            "RETRIEVAL_TOP_K",
            default=pipeline_config.get("top_k", 5),
        ),
    )

    seed = _parse_int(
        "SEED",
        _get_environment_value(
            "SEED",
            default=experiment_config.get("seed", 42),
        ),
    )

    poisoning_rate = _parse_float(
        "POISONING_RATE",
        _get_environment_value(
            "POISONING_RATE",
            default=poisoning_config.get("rate", 0.10),
        ),
    )

    experiment_repetitions = _parse_int(
        "EXPERIMENT_REPETITIONS",
        _get_environment_value(
            "EXPERIMENT_REPETITIONS",
            default=experiment_config.get("repetitions", 1),
        ),
    )

    config = AppConfig(
        project_name=str(
            data.get("project_name", "SecureCodeRAG")
        ),
        version=str(
            data.get("version", "0.1.0")
        ),
        log_level=str(
            _get_environment_value(
                "LOG_LEVEL",
                default=logging_config.get("level", "INFO"),
            )
        ).upper(),
        log_file=str(
            _get_environment_value(
                "LOG_FILE",
                default=logging_config.get(
                    "log_file",
                    "logs/app.log",
                ),
            )
        ),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        top_k=top_k,
        reranking_enabled=_parse_bool(
            "RERANKING_ENABLED",
            _get_environment_value(
                "RERANKING_ENABLED",
                default=pipeline_config.get(
                    "reranking_enabled",
                    False,
                ),
            ),
        ),
        embedding_provider=str(
            _get_environment_value(
                "EMBEDDING_PROVIDER",
                default=embeddings_config.get(
                    "provider",
                    models_config.get(
                        "embedding_provider",
                        "huggingface",
                    ),
                ),
            )
        ),
        embedding_model=str(
            _get_environment_value(
                "EMBEDDING_MODEL",
                default=embeddings_config.get(
                    "model",
                    models_config.get(
                        "embedding_model",
                        "microsoft/unixcoder-base",
                    ),
                )
            )
        ),
        vectorstore_provider=str(
            _get_environment_value(
                "VECTORSTORE_PROVIDER",
                default=vectorstore_config.get(
                    "provider",
                    "faiss",
                ),
            )
        ),
        vectorstore_path=str(
            _get_environment_value(
                "VECTORSTORE_PATH",
                default=paths_config.get(
                    "vectorstore",
                    "data/processed/faiss_index",
                ),
            )
        ),
        code_llm_provider=str(
            _get_environment_value(
                "CODE_LLM_PROVIDER",
                default=generation_config.get(
                    "provider",
                    models_config.get(
                        "code_llm_provider",
                        "ollama",
                    ),
                ),
            )
        ),
        code_llm=str(
            _get_environment_value(
                "CODE_LLM_MODEL",
                default=generation_config.get(
                    "model",
                    models_config.get(
                        "code_llm",
                        "deepseek-coder:6.7b",
                    ),
                ),
            )
        ),
        code_llm_base_url=str(
            _get_environment_value(
                "CODE_LLM_BASE_URL",
                "OLLAMA_BASE_URL",
                default=generation_config.get(
                    "base_url",
                    "http://localhost:11434",
                ),
            )
        ),
        hosted_llm_provider=str(
            _get_environment_value(
                "HOSTED_LLM_PROVIDER",
                default=generation_config.get(
                    "hosted_provider",
                    "openai",
                ),
            )
        ),
        hosted_llm_model=str(
            _get_environment_value(
                "HOSTED_LLM_MODEL",
                default=generation_config.get(
                    "hosted_model",
                    "gpt-4o-mini",
                ),
            )
        ),
        data_clean_dir=str(
            _get_environment_value(
                "CLEAN_DATA_DIR",
                default=paths_config.get(
                    "data_clean",
                    "data/clean",
                ),
            )
        ),
        data_poisoned_dir=str(
            _get_environment_value(
                "POISONED_DATA_DIR",
                default=paths_config.get(
                    "data_poisoned",
                    "data/poisoned",
                ),
            )
        ),
        data_processed_dir=str(
            _get_environment_value(
                "PROCESSED_DATA_DIR",
                default=paths_config.get(
                    "data_processed",
                    "data/processed",
                ),
            )
        ),
        benchmark_dir=str(
            _get_environment_value(
                "BENCHMARK_DIR",
                default=paths_config.get(
                    "benchmarks",
                    "data/benchmarks",
                ),
            )
        ),
        results_dir=str(
            _get_environment_value(
                "RESULTS_DIR",
                default=paths_config.get(
                    "results",
                    "results",
                ),
            )
        ),
         experiments_dir=str(
            _get_environment_value(
                "EXPERIMENTS_DIR",
                default=paths_config.get(
                    "experiments",
                    "experiments",
                ),
            )
        ),
        seed=seed,
        poisoning_enabled=_parse_bool(
            "POISONING_ENABLED",
            _get_environment_value(
                "POISONING_ENABLED",
                default=poisoning_config.get(
                    "enabled",
                    False,
                ),
            ),
        ),
        poisoning_rate=poisoning_rate,
        defense_enabled=_parse_bool(
            "DEFENSE_ENABLED",
            _get_environment_value(
                "DEFENSE_ENABLED",
                default=defense_config.get(
                    "enabled",
                    True,
                ),
            ),
        ),
        defense_trust_scoring=_parse_bool(
            "DEFENSE_TRUST_SCORING",
            _get_environment_value(
                "DEFENSE_TRUST_SCORING",
                default=defense_config.get(
                    "trust_scoring",
                    True,
                ),
            ),
        ),
        defense_anomaly_detection=_parse_bool(
            "DEFENSE_ANOMALY_DETECTION",
            _get_environment_value(
                "DEFENSE_ANOMALY_DETECTION",
                default=defense_config.get(
                    "anomaly_detection",
                    True,
                ),
            ),
        ),
        defense_context_validation=_parse_bool(
            "DEFENSE_CONTEXT_VALIDATION",
            _get_environment_value(
                "DEFENSE_CONTEXT_VALIDATION",
                default=defense_config.get(
                    "context_validation",
                    True,
                ),
            ),
        ),
        defense_instruction_separation=_parse_bool(
            "DEFENSE_INSTRUCTION_SEPARATION",
            _get_environment_value(
                "DEFENSE_INSTRUCTION_SEPARATION",
                default=defense_config.get(
                    "instruction_separation",
                    True,
                ),
            ),
        ),
        defense_static_analysis=_parse_bool(
            "DEFENSE_STATIC_ANALYSIS",
            _get_environment_value(
                "DEFENSE_STATIC_ANALYSIS",
                default=defense_config.get(
                    "static_analysis",
                    True,
                ),
            ),
        ),
        static_analysis_enabled=_parse_bool(
            "STATIC_ANALYSIS_ENABLED",
            _get_environment_value(
                "STATIC_ANALYSIS_ENABLED",
                default=security_config.get(
                    "static_analysis_enabled",
                    True,
                ),
            ),
        ),
        semgrep_enabled=_parse_bool(
            "SEMGREP_ENABLED",
            _get_environment_value(
                "SEMGREP_ENABLED",
                default=security_config.get(
                    "semgrep_enabled",
                    False,
                ),
            ),
        ),
        experiment_repetitions=experiment_repetitions,
        save_intermediate_results=_parse_bool(
            "SAVE_INTERMEDIATE_RESULTS",
            _get_environment_value(
                "SAVE_INTERMEDIATE_RESULTS",
                default=experiment_config.get(
                    "save_intermediate_results",
                    True,
                ),
            ),
        ),
        extra=data.get("extra", {}),
    )

    return config


__all__ = [
    "AppConfig",
    "ConfigurationError",
    "load_config",
]