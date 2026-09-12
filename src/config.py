"""
SecureCodeRAG Configuration Management Module.

Provides standard configuration loading from default JSON file with optional
environment variable overrides.
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

# Attempt to load dotenv if available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


@dataclass
class AppConfig:
    """Dataclass holding application configuration settings."""

    project_name: str = "SecureCodeRAG"
    version: str = "0.1.0"
    log_level: str = "INFO"
    log_file: str = "logs/app.log"
    chunk_size: int = 512
    chunk_overlap: int = 64
    top_k: int = 5
    embedding_model: str = "text-embedding-3-small"
    code_llm: str = "gpt-4o-mini"
    data_clean_dir: str = "data/clean"
    data_poisoned_dir: str = "data/poisoned"
    data_processed_dir: str = "data/processed"
    vectorstore_path: str = "data/processed/faiss_index"
    results_dir: str = "results"
    seed: int = 42

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration instance to dictionary."""
        return {
            "project_name": self.project_name,
            "version": self.version,
            "log_level": self.log_level,
            "log_file": self.log_file,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "top_k": self.top_k,
            "embedding_model": self.embedding_model,
            "code_llm": self.code_llm,
            "data_clean_dir": self.data_clean_dir,
            "data_poisoned_dir": self.data_poisoned_dir,
            "data_processed_dir": self.data_processed_dir,
            "vectorstore_path": self.vectorstore_path,
            "results_dir": self.results_dir,
            "seed": self.seed,
        }


def load_config(config_path: Optional[str] = None) -> AppConfig:
    """
    Load configuration settings from a JSON file and apply environment variable overrides.

    Args:
        config_path: Path to custom JSON configuration file. If None, default path is used.

    Returns:
        AppConfig object with loaded parameters.
    """
    project_root = Path(__file__).resolve().parent.parent

    if config_path is None:
        config_path = str(project_root / "configs" / "default_config.json")

    config_file = Path(config_path)
    file_data: Dict[str, Any] = {}

    if config_file.exists():
        with open(config_file, "r", encoding="utf-8") as f:
            file_data = json.load(f)

    # Extract nested sections with defaults
    logging_cfg = file_data.get("logging", {})
    pipeline_cfg = file_data.get("pipeline", {})
    models_cfg = file_data.get("models", {})
    paths_cfg = file_data.get("paths", {})
    experiment_cfg = file_data.get("experiment", {})

    # Construct AppConfig with overrides from environment variables if present
    config = AppConfig(
        project_name=file_data.get("project_name", "SecureCodeRAG"),
        version=file_data.get("version", "0.1.0"),
        log_level=os.getenv("LOG_LEVEL", logging_cfg.get("level", "INFO")),
        log_file=os.getenv("LOG_FILE", logging_cfg.get("log_file", "logs/app.log")),
        chunk_size=int(os.getenv("CHUNK_SIZE", pipeline_cfg.get("chunk_size", 512))),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", pipeline_cfg.get("chunk_overlap", 64))),
        top_k=int(os.getenv("TOP_K", pipeline_cfg.get("top_k", 5))),
        embedding_model=os.getenv("EMBEDDING_MODEL", models_cfg.get("embedding_model", "text-embedding-3-small")),
        code_llm=os.getenv("CODE_LLM_MODEL", models_cfg.get("code_llm", "gpt-4o-mini")),
        data_clean_dir=os.getenv("CLEAN_DATA_DIR", paths_cfg.get("data_clean", "data/clean")),
        data_poisoned_dir=os.getenv("POISONED_DATA_DIR", paths_cfg.get("data_poisoned", "data/poisoned")),
        data_processed_dir=os.getenv("PROCESSED_DATA_DIR", paths_cfg.get("data_processed", "data/processed")),
        vectorstore_path=os.getenv("VECTORSTORE_PATH", paths_cfg.get("vectorstore", "data/processed/faiss_index")),
        results_dir=os.getenv("RESULTS_DIR", paths_cfg.get("results", "results")),
        seed=int(os.getenv("SEED", experiment_cfg.get("seed", 42))),
    )

    return config
