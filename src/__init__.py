"""
SecureCodeRAG package.

This package provides the core foundation for the SecureCodeRAG research
system, including configuration management and logging.
"""

from src.config import AppConfig, load_config
from src.logger import get_logger

__version__ = "0.1.0"

__all__ = [
    "AppConfig",
    "__version__",
    "get_logger",
    "load_config",
]