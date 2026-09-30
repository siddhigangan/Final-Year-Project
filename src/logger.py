"""
Centralized logging utilities for SecureCodeRAG.
"""

import logging
from pathlib import Path


def get_logger(
    name: str = "SecureCodeRAG",
    log_file: str | None = "logs/app.log",
    level: str = "INFO",
) -> logging.Logger:
    """
    Create or retrieve a configured application logger.

    Parameters
    ----------
    name:
        Logger name.

    log_file:
        Optional path for a file handler. If None, only console logging
        is configured.

    level:
        Logging level such as DEBUG, INFO, WARNING, ERROR, or CRITICAL.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    logger = logging.getLogger(name)

    numeric_level = getattr(
        logging,
        level.upper(),
        logging.INFO,
    )

    logger.setLevel(numeric_level)

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)

    if log_file:
        try:
            log_path = Path(log_file)
            log_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            file_handler = logging.FileHandler(
                log_path,
                encoding="utf-8",
            )
            file_handler.setLevel(numeric_level)
            file_handler.setFormatter(formatter)

            logger.addHandler(file_handler)

        except (OSError, ValueError) as exc:
            logger.warning(
                "Could not initialize file handler for %s: %s",
                log_file,
                exc,
            )

    logger.propagate = False

    return logger