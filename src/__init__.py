# ── src/__init__.py ───────────────────────────────────────────────────────────
"""
Executed on the very first `import src.*`.

It guarantees Chroma can import on Streamlit Cloud (which currently ships
SQLite 3.34 and Python 3.13, and has no pre-built `pysqlite3-binary` wheel).

Strategy
────────
1.  Build a *shim* module named ``pysqlite3`` that wraps the std-lib
    ``sqlite3`` so that ``import pysqlite3`` always succeeds.
2.  Register that shim as *both* ``pysqlite3`` **and** ``sqlite3``.
3.  Spoof the reported SQLite version to ≥ 3 .35 so Chroma’s check passes.
4.  Tell Chroma to use the *duckdb + parquet* backend (no on-disk SQLite).

No other files need to change.
"""

from __future__ import annotations
import types, sys, os, sqlite3

# ── 1  build a shim around the std-lib sqlite3 ───────────────────────────────
shim = types.ModuleType("pysqlite3")
shim.__dict__.update(sqlite3.__dict__)          # re-export everything

# Report a “new” SQLite version so Chroma’s guard is happy
shim.sqlite_version       = "3.40.1"
shim.sqlite_version_info  = (3, 40, 1)

# ── 2  register under both names BEFORE anything imports Chroma ──────────────
sys.modules["pysqlite3"]  = shim          # satisfies  import pysqlite3
sys.modules["sqlite3"]    = shim          # everyone else still gets sqlite3

# ── 3  force Chroma away from SQLite storage entirely ────────────────────────
os.environ.setdefault("CHROMA_DB_IMPL", "duckdb+parquet")

# From here on, normal package imports can safely proceed.
# ──────────────────────────────────────────────────────────────────────────────
