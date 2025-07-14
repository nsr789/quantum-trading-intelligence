"""Unified logging: std-lib → Loguru → Structlog (colourful and safe)."""

from __future__ import annotations

import logging
import sys
from typing import Any, Dict

import structlog
from loguru import logger as _loguru

# ── configure Loguru sink ──────────────────────────────────────────────────────
LOG_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> "
    "<level>{level: <8}</level> "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> "
    "- <level>{message}</level>"
)
_loguru.remove()
_loguru.add(sys.stderr, format=LOG_FORMAT, level="INFO")

# ── route std-lib logging through Loguru ───────────────────────────────────────
class InterceptHandler(logging.Handler):
    """Redirect `logging.*` calls to Loguru."""

    def emit(self, record: logging.LogRecord) -> None:  # noqa: D401
        try:
            level = _loguru.level(record.levelname).name
        except ValueError:
            level = record.levelno
        _loguru.log(level, record.getMessage())


# clear default handlers then attach intercept
logging.root.handlers.clear()
logging.root.addHandler(InterceptHandler())
logging.root.setLevel(logging.INFO)

# ── Structlog pretty renderer ──────────────────────────────────────────────────
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.dev.ConsoleRenderer(colors=True),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
    logger_factory=structlog.PrintLoggerFactory(),  # piggyback on Loguru
)

log: structlog.BoundLogger = structlog.get_logger()


def get_logger(**kwargs: Dict[str, Any]):  # noqa: D401
    """Get a bound struct-logger with extra context."""
    return log.bind(**kwargs)
