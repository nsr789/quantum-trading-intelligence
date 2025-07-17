# src/strategies/__init__.py
"""
Factory that exposes the rule-based strategies available to the Back-tester.
(⚠️  LSTM has been removed.)
"""

from .mean_reversion import MeanReversion
from .momentum import Momentum

STRATEGY_MAP = {
    "Mean Reversion": MeanReversion,
    "Momentum":       Momentum,
}
