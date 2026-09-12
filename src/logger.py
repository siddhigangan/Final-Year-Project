"""
SecureCodeRAG Logging System.

Provides reusable logging initialization for tracking execution across all
pipeline stages (ingestion, chunking, embeddings, retrieval, generation, poisoning, defense, evaluation).
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def get_logger(
    name: str = "SecureCodeRAG",
    log_file: Optional[str] = "logs/app.log",
    level: str = "INFO",
) -> logging.Logger:
    """
    Configure and return a Logger instance.

    Args:
        name: Module or logger name.
        log_file: Path to log file. If None, file logging is disabled.
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).

    Returns:
        Configured logging.Logger object.
    """
    logger = logging.getLogger(name)

    # Convert string level to logging integer constant
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(numeric_level)

    # Prevent duplicate handlers if logger is already initialized
    if logger.handlers:
        return logger

    # Log line format
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(name)-20s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console Handler (stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(numeric_level)
    logger.addHandler(console_handler)

    # File Handler (Optional)
    if log_file:
        try:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)

            file_handler = logging.FileHandler(log_path, encoding="utf-8")
            file_handler.setFormatter(formatter)
            file_handler.setLevel(numeric_level)
            logger.addHandler(file_handler)
        except Exception as e:
            logger.warning(f"Could not initialize file handler for {log_file}: {e}")

    return logger
