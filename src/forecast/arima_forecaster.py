# ─────────── src/forecast/arima_forecaster.py
from __future__ import annotations
import warnings
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX


def arima_forecast(
    series: pd.Series,
    horizon: int = 5,
    order: tuple[int, int, int] = (1, 1, 1),
    seasonal: bool = False,
    s_order: tuple[int, int, int, int] = (1, 1, 1, 12),
) -> list[float]:
    """ARIMA / SARIMA forecast wrapper (statsmodels, CPU-only)."""
    warnings.filterwarnings("ignore")
    y = series.astype("float32")
    if seasonal:
        model = SARIMAX(y, order=order, seasonal_order=s_order, enforce_stationarity=False, enforce_invertibility=False)
    else:
        model = ARIMA(y, order=order)
    res = model.fit(method="statespace", disp=0)
    preds = res.forecast(steps=horizon).tolist()
    return preds
