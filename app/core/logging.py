import logging
import os
from pathlib import Path


def resolve_log_file() -> Path:
    base_dir = Path(__file__).resolve().parents[1]
    logs_dir = Path(os.getenv("LOG_DIR", base_dir / "logs"))
    log_file = Path(os.getenv("APP_LOG", logs_dir / "hunyuan_app.log"))
    log_file.parent.mkdir(parents=True, exist_ok=True)
    return log_file


def configure_logging() -> Path:
    log_file = resolve_log_file()
    root_logger = logging.getLogger()

    if getattr(root_logger, "_hunyuan_logging_configured", False):
        return log_file

    root_logger.setLevel(logging.INFO)
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    logging.captureWarnings(True)
    setattr(root_logger, "_hunyuan_logging_configured", True)
    logging.getLogger(__name__).info("Logging initialized. file=%s", log_file)
    return log_file
