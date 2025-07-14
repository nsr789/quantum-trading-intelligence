from .mean_reversion import MeanReversion
from .momentum import Momentum

STRATEGY_MAP = {
    "Mean-Reversion (Bollinger)": MeanReversion,
    "Momentum (SMA 20/50)": Momentum,
}
