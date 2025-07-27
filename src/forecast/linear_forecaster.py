"""
src/forecast/linear_forecaster.py
A feather-weight Ordinary-Least-Squares forecaster.

* No external GPU / heavy deps
* Learns a linear trend over the last *lookback* bars
* Returns a list[float] of length *horizon*
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def linear_forecast(                   # <-  ⚠  name must match __init__.py
    series: pd.Series | Sequence[float],
    *,
    horizon: int = 5,
    lookback: int = 120,
) -> list[float]:
    """
    Parameters
    ----------
    series   : price series (pd.Series or 1-D array-like)
    horizon  : number of future steps to predict
    lookback : window of recent observations to fit on
    """
    # ---------- data prep ----------------------------------------------------
    s = np.asarray(series[-lookback:], dtype="float64")
    X = np.arange(len(s)).reshape(-1, 1)           # time index 0 … lookback-1
    y = s

    # ---------- model fit ----------------------------------------------------
    model = LinearRegression()
    model.fit(X, y)

    # ---------- forecast -----------------------------------------------------
    future_X = np.arange(len(s), len(s) + horizon).reshape(-1, 1)
    y_hat = model.predict(future_X)

    return y_hat.tolist()
