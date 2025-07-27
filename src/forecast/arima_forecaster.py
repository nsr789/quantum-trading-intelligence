# ─────────── src/forecast/arima_forecaster.py  (drop-in replacement)
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
    Generic ARIMA / SARIMA forecaster.

    Parameters
    ----------
    series : pd.Series
        Price series – **must be indexed by date** (any frequency).
    horizon : int
        Number of future steps to predict.
    order : (p,d,q)
        ARIMA order for non-seasonal part.
    seasonal : bool
        If True ➜ use SARIMAX with `seasonal_order`.
    seasonal_order : (P,D,Q,s)
        Seasonal component (ignored when `seasonal=False`).

    Returns
    -------
    list[float]  – length == `horizon`
    """
    y = series.dropna().astype("float32")

    if seasonal:
        # sensible default if caller didn’t specify
        seasonal_order = seasonal_order or (1, 0, 0, 5)
        model = SARIMAX(
            y,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        res = model.fit()          # ← NO disp kwarg
    else:
        model = ARIMA(y, order=order)
        res = model.fit()          # ← NO disp kwarg

    return res.forecast(steps=horizon).tolist()
