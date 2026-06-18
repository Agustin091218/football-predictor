"""Structured logging configuration for API, CLI, and scheduler.

Must be imported BEFORE any logger emits messages to ensure
consistent formatting across all components.

Supports two modes:
- Default: plain text with timestamp, level, logger, message
- JSON: one JSON object per line (set LOG_FORMAT=json)
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        return json.dumps(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
            },
            default=str,
        )


def setup_logging() -> None:
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    fmt_mode = os.getenv("LOG_FORMAT", "text").lower()

    if fmt_mode == "json":
        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
        logging.basicConfig(level=level, handlers=[handler], force=True)
    else:
        fmt = os.getenv(
            "LOG_FORMAT",
            "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        )
        datefmt = os.getenv("LOG_DATEFMT", "%Y-%m-%d %H:%M:%S")
        logging.basicConfig(level=level, format=fmt, datefmt=datefmt, force=True)

    # Reduce noise from third-party libraries
    logging.getLogger("apscheduler").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    logger = logging.getLogger(__name__)
    logger.info("Logging configured (level=%s, format=%s)", level_name, fmt_mode)
