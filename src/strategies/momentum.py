from __future__ import annotations
import pandas as pd
from src.strategies.base import Strategy


class Momentum(Strategy):
    """Simple SMA-crossover momentum."""

    name = "Momentum (SMA 20/50)"

    def __init__(self, fast: int = 20, slow: int = 50):
        assert fast < slow, "fast SMA must be shorter"
        self.fast, self.slow = fast, slow

    def generate_signals(self, price: pd.Series) -> pd.Series:
        fast_sma = price.rolling(self.fast).mean()
        slow_sma = price.rolling(self.slow).mean()
        pos = (fast_sma > slow_sma).astype(int)
        return pos.reindex(price.index).fillna(0)
