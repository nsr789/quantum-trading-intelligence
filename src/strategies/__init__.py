# src/strategies/__init__.py
"""
Factory that exposes the rule-based strategies available to the Back-tester.
LSTM has been removed – only Mean-Reversion & Momentum remain.
"""

from .mean_reversion import MeanReversionStrategy
from .momentum import MomentumStrategy

# Public map used by the Back-tester page
STRATEGY_MAP = {
    "Mean Reversion": MeanReversionStrategy,
    "Momentum": MomentumStrategy,
}
