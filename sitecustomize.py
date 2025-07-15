"""
sitecustomize.py  – loaded automatically on interpreter start.
Monkey-patches the stdlib 'sqlite3' to use the modern wheels shipped
in pysqlite3-binary (needed for Chroma on Streamlit Cloud).
"""
try:
    import pysqlite3  # modern SQLite (>=3.44) compiled with FTS5 etc.
    import sys
    sys.modules["sqlite3"] = pysqlite3
except ImportError:
    # local env already has a good sqlite3 → no action needed
    pass
