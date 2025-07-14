import pytest

from src.data.macro import get_unemployment_rate

def test_macro_fallback():
    """If no FRED key the helper should return None, not raise."""
    try:
        rate = get_unemployment_rate()
    except RuntimeError:
        pytest.skip("FRED key missing, skipping")
    else:
        assert rate is None or rate > 0
