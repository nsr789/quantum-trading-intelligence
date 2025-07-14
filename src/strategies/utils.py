from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict
from src.strategies.base import Strategy


def backtest(price: pd.Series, strategy: Strategy) -> Dict[str, float | pd.Series]:
    """Vectorised back-test with no slippage / fees."""
    sig = strategy.generate_signals(price)
    sig = sig.shift(1).fillna(0)  # trade on next bar open
    ret = price.pct_change().fillna(0)
    strat_ret = sig * ret
    equity = (1 + strat_ret).cumprod()
    sharpe = (strat_ret.mean() / strat_ret.std()) * np.sqrt(252) if strat_ret.std() else 0
    hit = (strat_ret > 0).mean()
    return {
        "equity": equity,
        "sharpe": sharpe,
        "hit_rate": hit,
        "cum_return": equity.iloc[-1] - 1,
    }
