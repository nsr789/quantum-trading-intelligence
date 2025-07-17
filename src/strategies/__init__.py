# src/strategies/__init__.py
"""
Factory that exposes the rule-based strategies available to the Back-tester.
(⚠️  LSTM has been removed.)
"""

from .mean_reversion import MeanReversionStrategy
from .momentum import MomentumStrategy

STRATEGY_MAP = {
    "Mean Reversion": MeanReversionStrategy,
    "Momentum":       MomentumStrategy,
}
