"""Public strategy registry."""
from .base import Strategy                 # noqa: F401 (re-export)

# existing rule-based strategies
from .mean_reversion import MeanReversion
from .momentum        import Momentum

# ▶ NEW
from .lstm            import LSTMStrategy

# a single dict the UI & tests import
STRATEGY_MAP = {
    "Mean-Reversion": MeanReversion,
    "Momentum":       Momentum,
    "LSTM":           LSTMStrategy,        # <-- integrated
}

__all__ = ["STRATEGY_MAP", "MeanReversion", "Momentum", "LSTMStrategy"]
