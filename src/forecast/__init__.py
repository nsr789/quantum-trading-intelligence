# ─────────── src/forecast/__init__.py  (drop-in replacement)
from __future__ import annotations
from importlib import import_module
from types import ModuleType
from typing import Callable, Dict


def _lazy(mod_name: str, func_name: str) -> Callable:
    """Return a wrapper that imports the real forecaster only on first call."""
    _func: Callable | None = None

    def _wrapper(series, **kwargs):
        nonlocal _func
        if _func is None:
            mod: ModuleType = import_module(f"src.forecast.{mod_name}")
            _func = getattr(mod, func_name)
        return _func(series, **kwargs)

    return _wrapper


# Public mapping used by the Streamlit page
FORECASTERS: Dict[str, Callable] = {
    "LSTM": _lazy("lstm_forecaster", "lstm_forecast"),
    "Linear Reg.": _lazy("linear_forecaster", "linear_regression_forecast"),
    "ARIMA": _lazy("arima_forecaster", "arima_forecast"),
    "SARIMA": _lazy("arima_forecaster", "arima_forecast"),          # seasonal flag set later
    "XGBoost": _lazy("xgb_forecaster", "xgboost_forecast"),
}

# Helper to inject seasonal flag without double-importing
def get_forecaster(name: str):
    if name == "SARIMA":
        return lambda s, **kw: FORECASTERS["SARIMA"](s, seasonal=True, **kw)
    return FORECASTERS[name]
