# app/__init__.py
"""
Make project root importable when Streamlit changes cwd to /app.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]   # …/quantum-trading-intelligence
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
