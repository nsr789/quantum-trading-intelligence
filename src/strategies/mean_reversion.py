from __future__ import annotations
import pandas as pd
from src.strategies.base import Strategy


class MeanReversion(Strategy):
    """Bollinger-band mean-reversion."""

    name = "Mean-Reversion (Bollinger)"

    def __init__(self, window: int = 20, z: float = 2.0):
        self.window, self.z = window, z

    def generate_signals(self, price: pd.Series) -> pd.Series:
        ma = price.rolling(self.window).mean()
        sd = price.rolling(self.window).std()
        upper, lower = ma + self.z * sd, ma - self.z * sd
        long = (price < lower).astype(int)
        short = (price > upper).astype(int) * -1
        pos = long + short
        return pos.reindex(price.index).fillna(0)
