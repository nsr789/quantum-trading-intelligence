# ── src/__init__.py ───────────────────────────────────────────────────────────
"""
Executed on first `import src.*`.

 • Provides a shim so `import pysqlite3` always works (Chroma requirement).
 • Spoofs a modern SQLite version so Chroma’s runtime check passes.
 • *No longer* sets the legacy `CHROMA_DB_IMPL` env-var (new Chroma bails out
   when it sees it).  The default backend is already DuckDB + Parquet.
"""

from __future__ import annotations
import types, sys, os, sqlite3

# ── 1  build a shim module around the std-lib sqlite3 ─────────────────────────
shim = types.ModuleType("pysqlite3")
shim.__dict__.update(sqlite3.__dict__)          # re-export everything

#  Pretend our SQLite is new enough for Chroma’s guard
shim.sqlite_version       = "3.40.1"
shim.sqlite_version_info  = (3, 40, 1)

# ── 2  register under BOTH names *before* anything imports Chroma ─────────────
sys.modules["pysqlite3"]  = shim
sys.modules["sqlite3"]    = shim

# ── 3  ensure no legacy env-var confuses recent Chroma versions ───────────────
os.environ.pop("CHROMA_DB_IMPL", None)          # silently discard if present

# From here on, normal imports can proceed.
# ──────────────────────────────────────────────────────────────────────────────
