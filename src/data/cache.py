"""Tiny disk-cache decorator to avoid repeated API hits.

Uses joblib.Memory under the hood (pickle-based, RAM-friendly).
"""

from __future__ import annotations

from functools import wraps
from pathlib import Path
from typing import Any, Callable

from joblib import Memory

from src.config.settings import settings

_mem = Memory(location=settings.CACHE_DIR, verbose=0)


def cached(func: Callable[..., Any]):  # noqa: D401
    """Decorator: caches function result keyed by args + kwargs."""

    cached_func = _mem.cache(func)

    @wraps(func)
    def wrapper(*args, **kwargs):  # noqa: D401
        return cached_func(*args, **kwargs)

    return wrapper
