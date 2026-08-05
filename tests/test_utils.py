import logging

import utils
from config import Settings


def reset_project_logger() -> logging.Logger:
    """Remove existing handlers so every test starts with a clean logger."""
    logger = logging.getLogger("financial_rag")

    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)

    return logger


def test_configure_logging_creates_log_file(tmp_path, monkeypatch):
    """Logging should write application events to the configured log folder."""
    test_settings = Settings(
        data_dir=tmp_path / "data",
        vectorstore_dir=tmp_path / "vectorstore",
        log_dir=tmp_path / "logs",
        log_level="INFO",
    )

    monkeypatch.setattr(utils, "settings", test_settings)
    reset_project_logger()

    logger = utils.configure_logging()
    logger.info("Logging test message")

    for handler in logger.handlers:
        handler.flush()

    log_file = test_settings.log_dir / "financial_rag.log"

    assert log_file.exists()
    assert "Logging test message" in log_file.read_text(encoding="utf-8")

    reset_project_logger()


def test_configure_logging_does_not_duplicate_handlers(tmp_path, monkeypatch):
    """Repeated setup must not duplicate log messages or handlers."""
    test_settings = Settings(
        data_dir=tmp_path / "data",
        vectorstore_dir=tmp_path / "vectorstore",
        log_dir=tmp_path / "logs",
    )

    monkeypatch.setattr(utils, "settings", test_settings)
    reset_project_logger()

    first_logger = utils.configure_logging()
    handler_count = len(first_logger.handlers)

    second_logger = utils.configure_logging()

    assert first_logger is second_logger
    assert len(second_logger.handlers) == handler_count

    reset_project_logger()