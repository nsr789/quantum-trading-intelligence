import pandas as pd
from src.strategies.mean_reversion import MeanReversion
from src.strategies.momentum import Momentum
from src.strategies.utils import backtest

def _fake_price():
    return pd.Series(range(100), dtype=float).cumsum()  # monotonic uptrend

def test_mean_reversion():
    eq = backtest(_fake_price(), MeanReversion()).get("equity")
    assert len(eq) == 100

def test_momentum():
    eq = backtest(_fake_price(), Momentum()).get("equity")
    assert len(eq) == 100
