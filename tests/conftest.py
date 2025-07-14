import sys
from pathlib import Path
import pytest

# ensure src/ is on path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

def pytest_configure(config):
    config.addinivalue_line("markers", "asyncio: mark test as asyncio")
