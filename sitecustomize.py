"""
sitecustomize.py  –  auto-imported by Python.

1. Force Chroma to use DuckDB instead of SQLite (avoids version issues).
2. (Optional) Fall back to pysqlite3-binary if you later want SQLite again.
"""

import os
os.environ.setdefault("CHROMA_DB_IMPL", "duckdb+parquet")  # bye-bye SQLite 👋

# --- optional safety net: modern SQLite wheel ---
try:
    import pysqlite3                 # only present locally if you installed it
    import sys                       # pragma: no cover
    sys.modules["sqlite3"] = pysqlite3
except Exception:
    # Either wheel not installed or already fine – no action needed
    pass
