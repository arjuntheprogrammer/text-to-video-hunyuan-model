import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path


def resolve_log_file() -> Path:
    base_dir = Path(__file__).resolve().parents[1]
    logs_dir = Path(os.getenv("LOG_DIR", base_dir / "logs"))
    log_file = Path(os.getenv("APP_LOG", logs_dir / "hunyuan_app.log"))
    log_file.parent.mkdir(parents=True, exist_ok=True)
    return log_file


class JsonFormatter(logging.Formatter):
    _reserved = {
        "name",
        "msg",
        "args",
        "levelname",
        "levelno",
        "pathname",
        "filename",
        "module",
        "exc_info",
        "exc_text",
        "stack_info",
        "lineno",
        "funcName",
        "created",
        "msecs",
        "relativeCreated",
        "thread",
        "threadName",
        "processName",
        "process",
    }

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "line": record.lineno,
            "process": record.process,
            "thread": record.threadName,
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stack"] = self.formatStack(record.stack_info)

        for key, value in record.__dict__.items():
            if key in self._reserved or key.startswith("_"):
                continue
            if key in payload:
                continue
            try:
                json.dumps(value)
                payload[key] = value
            except TypeError:
                payload[key] = str(value)

        return json.dumps(payload, ensure_ascii=False)


def configure_logging() -> Path:
    log_file = resolve_log_file()
    root_logger = logging.getLogger()

    if getattr(root_logger, "_hunyuan_logging_configured", False):
        return log_file

    root_logger.setLevel(logging.INFO)
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)

    formatter = JsonFormatter()
    file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    logging.captureWarnings(True)
    setattr(root_logger, "_hunyuan_logging_configured", True)
    logging.getLogger(__name__).info("Logging initialized. file=%s", log_file)
    return log_file
