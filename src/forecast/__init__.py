# ─────────── src/forecast/__init__.py
from importlib import import_module, reload
from types import ModuleType
from typing import Callable

__all__ = ["FORECASTERS"]

_FORECAST_CACHE: dict[str, Callable] = {}


def _lazy(name: str, func_name: str) -> Callable:
    mod_name = f"src.forecast.{name}"

    def _wrapper(series, **kwargs):
        fn = _FORECAST_CACHE.get(mod_name)
        if fn is None:                 # first call – import module
            mod: ModuleType = import_module(mod_name)
            fn = getattr(mod, func_name)
            _FORECAST_CACHE[mod_name] = fn
        else:                          # reload if the source changed
            fn = getattr(reload(import_module(mod_name)), func_name)
            _FORECAST_CACHE[mod_name] = fn
        return fn(series, **kwargs)

    return _wrapper


FORECASTERS = {
    "Linear Regression": _lazy("linear_forecaster", "linear_forecast"),
    "ARIMA": _lazy("arima_forecaster", "arima_forecast"),
    "SARIMA": _lazy("arima_forecaster", "arima_forecast"),
    "LSTM": _lazy("lstm_forecaster", "lstm_forecast"),
    "XGBoost": _lazy("xgb_forecaster", "xgb_forecast"),
}
