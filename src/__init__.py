"""
SecureCodeRAG: Security Evaluation and Defense Pipeline for Retrieval-Augmented Code Generation.
"""

from src.config import AppConfig, load_config
from src.logger import get_logger

__version__ = "0.1.0"

__all__ = [
    "AppConfig",
    "load_config",
    "get_logger",
    "__version__",
]
