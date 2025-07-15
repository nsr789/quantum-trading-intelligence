# sitecustomize.py
"""
Auto-loaded on interpreter start (PEP 370). 
Replaces stdlib sqlite3 with the modern version shipped
inside the pysqlite3-binary wheel — required by ChromaDB.
"""
try:
    import pysqlite3          # ≥3.44 compiled with FTS5, etc.
    import sys
    sys.modules["sqlite3"] = pysqlite3
except ImportError:
    # Local env already has a new-enough sqlite → nothing to do
    pass
