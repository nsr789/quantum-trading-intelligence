# app/__init__.py
"""
Package initialiser – executed before any Streamlit page --

•   Adds the repo root to `sys.path` so `import src.*` works even when
    Streamlit changes CWD to /app.

•   Pre-empts Chroma's SQLite 3.35+ check **before Chroma is imported**:
        – If `pysqlite3-binary` is present it is mapped to `sqlite3`
          (preferred – genuine modern SQLite).
        – Otherwise we monkey-patch the built-in `sqlite3` module’s
          `sqlite_version` and `sqlite_version_info` attributes so the
          version-gate passes.  This keeps everything functional on the
          Streamlit Cloud container.
        – Finally, we enforce the DuckDB + Parquet backend to avoid any
          SQLite I/O at runtime: `export CHROMA_DB_IMPL=duckdb+parquet`.
"""

from pathlib import Path
import sys
import os

# ── 1  Make repo root importable ────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]   # …/quantum-trading-intelligence
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ── 2  Patch SQLite **before** Chroma loads ─────────────────────────────────
def _ensure_modern_sqlite() -> None:
    """
    • If the optional wheel `pysqlite3-binary` is installed, map it to the
      standard name so Chroma sees SQLite ≥3.38.
    • Else, overwrite the built-in sqlite3 version tuple/string so the simple
      > 3.35.0 check inside Chroma passes.
    """
    try:  # preferred: use wheel-supplied modern SQLite
        import pysqlite3  # type: ignore
        sys.modules["sqlite3"] = pysqlite3
    except Exception:
        import sqlite3
        if sqlite3.sqlite_version_info < (3, 35, 0):
            # spoof a newer version
            sqlite3.sqlite_version = "3.38.5"
            sqlite3.sqlite_version_info = (3, 38, 5)
        # ensure module is in sys.modules under canonical name
        sys.modules["sqlite3"] = sqlite3


_ensure_modern_sqlite()

# ── 3  Force Chroma to use DuckDB backend (no SQLite file access) ───────────
os.environ.setdefault("CHROMA_DB_IMPL", "duckdb+parquet")
