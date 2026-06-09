"""JSON Lines logging for Runtime runs (architecture.md: ``logging``, JSON Lines)."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ._jsonutil import json_safe

LOGGER_NAME = "taskwright.runs"
_LOG_FILENAME = "taskwright.jsonl"


class JsonLinesFormatter(logging.Formatter):
    """Render each log record as a single-line JSON object."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
        }
        context = getattr(record, "context", None)
        if isinstance(context, dict):
            payload.update(json_safe(context))
        return json.dumps(payload, ensure_ascii=False, default=str)


def get_run_logger(base_dir: Path) -> logging.Logger:
    """Return a logger that appends JSON lines to ``base_dir/taskwright.jsonl``.

    The file handler is attached at most once per destination, so repeated runs
    in the same process do not duplicate log lines.
    """
    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    log_path = (base_dir / _LOG_FILENAME).resolve()

    logger = logging.getLogger(f"{LOGGER_NAME}.{log_path}")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    already = any(
        isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", None) == str(log_path)
        for h in logger.handlers
    )
    if not already:
        handler = logging.FileHandler(log_path, encoding="utf-8")
        handler.setFormatter(JsonLinesFormatter())
        logger.addHandler(handler)
    return logger
