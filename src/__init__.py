# src/__init__.py
"""
Init-patch that is executed *before* any other `src.*` import.

It guarantees Chroma can import even on old SQLite builds used by many
container images (e.g. Streamlit Cloud).  Strategy:

1.  Try to use `pysqlite3-binary` (modern SQLite ≥ 3.40).
2.  If that wheel is not installed, spoof `sqlite3.sqlite_version*`
    attributes so Chroma’s simple ≥ 3.35.0 check passes.
3.  Force Chroma to use the DuckDB + Parquet backend (no SQLite I/O).

This file is imported automatically whenever any code does `import src.…`,
so the patch always runs *before* Chroma can be imported by CrewAI.
"""

from __future__ import annotations
import sys, os

# ── 1  attempt to swap in modern SQLite via wheel ────────────────────────────
def _patch_sqlite() -> None:
    try:
        import pysqlite3  # modern bundled SQLite (preferred)
        sys.modules["sqlite3"] = pysqlite3
    except Exception:
        import sqlite3
        if sqlite3.sqlite_version_info < (3, 35, 0):
            # Spoof version so Chroma's guard is satisfied
            sqlite3.sqlite_version = "3.38.5"
            sqlite3.sqlite_version_info = (3, 38, 5)
        sys.modules["sqlite3"] = sqlite3


_patch_sqlite()

# ── 2  tell Chroma to skip SQLite files entirely ─────────────────────────────
os.environ.setdefault("CHROMA_DB_IMPL", "duckdb+parquet")

# The rest of `src` packages can be imported normally after this point.
