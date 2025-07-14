import pandas as pd
import pytest

from src.data.market import get_price_history, get_fundamentals


def test_price_history():
    df = get_price_history("AAPL", start="2024-01-01", end="2024-01-15")
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert {"Open", "Close", "Volume"}.issubset(df.columns)


def test_fundamentals():
    data = get_fundamentals("AAPL")
    assert "MarketCap" in data
    # key may be None depending on provider – just ensure function returns dict
    assert isinstance(data, dict)


@pytest.mark.asyncio
async def test_price_batch_async():
    from src.data.market import get_price_batch

    out = await get_price_batch(["AAPL", "MSFT"], start="2024-01-10", end="2024-01-15")
    assert "AAPL" in out and "MSFT" in out
