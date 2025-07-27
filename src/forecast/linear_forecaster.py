# ─────────── src/forecast/linear_forecaster.py
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def linear_regression_forecast(
    series: pd.Series,
    horizon: int = 5,
) -> list[float]:
    """OLS on time index ⇒ forecast."""
    y = series.values.astype("float32")
    X = np.arange(len(y), dtype="float32").reshape(-1, 1)
    model = LinearRegression().fit(X, y)

    X_fut = np.arange(len(y), len(y) + horizon, dtype="float32").reshape(-1, 1)
    preds = model.predict(X_fut).tolist()
    return preds
