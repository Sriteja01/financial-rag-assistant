"""Shared utilities for the Financial Statement RAG Assistant."""

import logging
from logging.handlers import RotatingFileHandler

from config import settings


def configure_logging() -> logging.Logger:
    """Configure and return the application logger.

    Logs are written to both the terminal and a rotating file. Rotation prevents
    log files from growing forever in a long-running Streamlit application.
    """
    settings.ensure_directories()

    logger = logging.getLogger("financial_rag")
    logger.setLevel(settings.log_level)
    logger.propagate = False

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    file_handler = RotatingFileHandler(
        settings.log_dir / "financial_rag.log",
        maxBytes=1_000_000,
        backupCount=3,
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


logger = configure_logging()