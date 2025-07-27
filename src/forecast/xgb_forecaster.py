# ─────────── src/forecast/xgb_forecaster.py
from __future__ import annotations
import numpy as np
import pandas as pd
from xgboost import XGBRegressor


def _lagged_matrix(y: np.ndarray, n_lags: int):
    X, tgt = [], []
    for i in range(n_lags, len(y)):
        X.append(y[i - n_lags : i])
        tgt.append(y[i])
    return np.array(X), np.array(tgt)


def xgboost_forecast(
    series: pd.Series,
    horizon: int = 5,
    n_lags: int = 60,
    n_estimators: int = 200,
) -> list[float]:
    """Gradient-boosted tree forecast (univariate)."""
    y = series.values.astype("float32")
    y = y[-(n_lags * 3) :]  # keep recent
    X, tgt = _lagged_matrix(y, n_lags)

    model = XGBRegressor(
        n_estimators=n_estimators,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        objective="reg:squarederror",
        n_jobs=1,
    ).fit(X, tgt)

    window = y[-n_lags:].tolist()
    preds = []
    for _ in range(horizon):
        p = model.predict(np.array(window[-n_lags:]).reshape(1, -1))[0]
        preds.append(float(p))
        window.append(p)
    return preds
