# ─────────── src/forecast/arima_forecaster.py
from __future__ import annotations

from typing import List

import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX


def arima_forecast(
    series: pd.Series,
    horizon: int = 5,
    order: tuple[int, int, int] = (5, 1, 0),
    seasonal: bool = False,
    seasonal_order: tuple[int, int, int, int] | None = None,
    **_,
) -> List[float]:
    """
    Lightweight ARIMA / SARIMA forecaster.

    Parameters
    ----------
    series         : price series (pd.Series, date-index)
    horizon        : number of future steps to predict
    order          : non-seasonal (p, d, q)
    seasonal       : if True → use SARIMAX
    seasonal_order : (P, D, Q, s) — ignored when *seasonal* is False
    """
    y = series.dropna().astype("float32")

    if seasonal:
        seasonal_order = seasonal_order or (1, 0, 0, 5)
        model = SARIMAX(
            y,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        res = model.fit()              # statsmodels ≥0.14 (no 'disp=' kwarg)
    else:
        model = ARIMA(y, order=order)
        res = model.fit()              # same here – defaults are fine

    return res.forecast(steps=horizon).tolist()


# ─────────────────────────── quick self-test ────────────────────────────
if __name__ == "__main__":
    import yfinance as yf

    s = yf.download("AAPL", start="2024-01-01")["Close"]
    print("ARIMA(5,1,0) ➜ next 3 values:", arima_forecast(s, horizon=3))
    print("SARIMA(1,0,0)x(1,0,0,5) ➜ next 3 values:",
          arima_forecast(s, horizon=3, seasonal=True))
