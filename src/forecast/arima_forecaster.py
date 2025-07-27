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
    ARIMA / SARIMA forecaster
    ------------------------
    * `series` – price series (pd.Series, date-index)
    * `horizon` – number of future steps to predict
    * `order` – non-seasonal (p, d, q)
    * `seasonal` – set True to use SARIMA
    * `seasonal_order` – (P, D, Q, s) when seasonal=True
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
        res = model.fit()
    else:
        model = ARIMA(y, order=order)
        res = model.fit()

    return res.forecast(steps=horizon).tolist()
