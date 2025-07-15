# app/__init__.py
"""
Package initializer – runs before any Streamlit page is executed.

1.  Ensures the project root (…/quantum-trading-intelligence) is on sys.path
    so that `import src.*` works when Streamlit changes CWD to /app.

2.  Fixes the Streamlit-Cloud “unsupported version of sqlite3” error:

    • Forces Chroma to use DuckDB + Parquet (no SQLite dependency).
    • Optionally swaps in the `pysqlite3-binary` wheel if it’s installed,
      giving you a modern SQLite implementation for local development.
"""

from pathlib import Path
import sys
import os

# ── 1) Make repo root importable ────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]  # …/quantum-trading-intelligence
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ── 2) Patch environment before CrewAI → Chroma loads ───────────────────────
# Tell Chroma to use DuckDB backend (avoids SQLite version check)
os.environ.setdefault("CHROMA_DB_IMPL", "duckdb+parquet")

# Optional: if pysqlite3-binary wheel is present, map it to `sqlite3`
try:
    import pysqlite3  # wheel provides SQLite ≥ 3.38
    sys.modules["sqlite3"] = pysqlite3
except Exception:
    # Wheel not installed or already patched – ignore
    pass
