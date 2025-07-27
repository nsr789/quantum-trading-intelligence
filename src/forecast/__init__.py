# ─────────── src/forecast/__init__.py
from __future__ import annotations

from .lstm_forecaster import lstm_forecast
from .linear_forecaster import linear_regression_forecast
from .arima_forecaster import arima_forecast
from .xgb_forecaster import xgboost_forecast

FORECASTERS = {
    "LSTM": lstm_forecast,
    "Linear Reg.": linear_regression_forecast,
    "ARIMA": lambda s, **kw: arima_forecast(s, seasonal=False, **kw),
    "SARIMA": lambda s, **kw: arima_forecast(s, seasonal=True, **kw),
    "XGBoost": xgboost_forecast,
}
