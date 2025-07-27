# ─────────────────────────── src/forecast/xgb_forecaster.py
"""
Light-weight XGBoost regressor forecaster.

* CPU-only; trains in <½ s on a few hundred points
* Uses simple lag features   y_{t-1} … y_{t-n}
* Recursive multi-step forecast
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
from xgboost import XGBRegressor


def _make_lags(arr: np.ndarray, n_lags: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (X, y) where X has shape (samples, n_lags)."""
    X, y = [], []
    for i in range(n_lags, len(arr)):
        X.append(arr[i - n_lags : i])
        y.append(arr[i])
    return np.asarray(X, dtype="float32"), np.asarray(y, dtype="float32")


def xgb_forecast(
    series: pd.Series,
    horizon: int = 5,
    n_lags: int = 60,
    **_,
) -> List[float]:
    """
    XGBoost price forecaster.

    Parameters
    ----------
    series   : pd.Series of prices, date index
    horizon  : number of future steps to predict
    n_lags   : how many past closes are used as features
    """
    y_hist = series.dropna().astype("float32").values
    # keep roughly 3× the lag window for training
    y_hist = y_hist[-(n_lags * 3) :]
    X, y = _make_lags(y_hist, n_lags)

    model = XGBRegressor(
        objective="reg:squarederror",
        max_depth=3,
        n_estimators=200,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        n_jobs=1,
        verbosity=0,
    )
    model.fit(X, y)

    # ── recursive forecast
    last_window = y_hist[-n_lags:].copy()
    preds: list[float] = []
    for _ in range(horizon):
        nxt = float(model.predict(last_window.reshape(1, -1))[0])
        preds.append(nxt)
        last_window = np.roll(last_window, -1)
        last_window[-1] = nxt

    return preds
