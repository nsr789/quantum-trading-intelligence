# src/ml/predictor.py
"""
Lightweight ONNX runtime wrapper for next-day price forecasts.

If no pre-exported ONNX model is found we fit a tiny linear-regression on the
fly so the Streamlit UI never crashes.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal, Sequence

import numpy as np
import onnxruntime as ort
import pandas as pd
from sklearn.linear_model import LinearRegression  # fallback trainer

from src.data.market import get_price_history
from src.utils.logging import get_logger

Engine = Literal["linear", "xgb", "lstm"]
MODEL_DIR = Path(__file__).resolve().parent / "models"

log = get_logger(module="predictor")


class PricePredictor:
    """Runtime inference helper — trains only when ONNX is missing."""

    # ------------------------------------------------------------------ #
    def __init__(self, ticker: str, window: int = 30, engine: Engine = "linear"):
        self.ticker = ticker.upper()
        self.window = int(window)
        self.engine: Engine = engine

        self.onnx_path = MODEL_DIR / f"{engine}-{window}.onnx"
        self.scaler_mu = MODEL_DIR / f"scaler-{window}.npy"
        self.scaler_sd = MODEL_DIR / f"scale-{window}.npy"

        # --- load model or fallback ------------------------------------
        if self.onnx_path.exists():
            self._sess = ort.InferenceSession(str(self.onnx_path))
            log.info("ONNX model loaded: %s", self.onnx_path.name)
        else:
            log.warning("ONNX model for %s not found – training fallback.", self.engine)
            self._train_fallback()

        # --- get price history once per instance -----------------------
        self.series = get_price_history(self.ticker, interval="1d")["Close"]

        # --- load scaler if present ------------------------------------
        if self.scaler_mu.exists() and self.scaler_sd.exists():
            self.mean_ = np.load(self.scaler_mu)
            self.scale_ = np.load(self.scaler_sd)
        else:
            self.mean_ = None
            self.scale_ = None

    # ------------------------------------------------------------------ #
    # helpers
    def _windowed(self) -> np.ndarray:
        """Return last `window` closes as (1, window) float32 tensor."""
        x = self.series.iloc[-self.window :].values.astype(np.float32).reshape(1, -1)
        if self.mean_ is not None:
            x = (x - self.mean_) / self.scale_
        return x

    # ------------------------------------------------------------------ #
    # inference
    def predict_next_close(self) -> float:
        if hasattr(self, "_sess"):
            inp = {self._sess.get_inputs()[0].name: self._windowed()}
            return float(self._sess.run(None, inp)[0].item())

        # linear-regression fallback
        x = self._windowed().ravel()
        return float(self._coef_ @ x + self._bias_)

    def predict_series(self, horizon: int = 5) -> pd.Series:
        """Naïve recursive forecast for `horizon` business days."""
        preds: list[float] = []
        temp = self.series.copy()

        for _ in range(horizon):
            self.series = temp
            nxt = self.predict_next_close()
            preds.append(nxt)
            temp = pd.concat([temp, pd.Series([nxt])])

        idx = pd.date_range(
            start=self.series.index[-1] + pd.Timedelta(days=1),
            periods=horizon,
            freq="B",
        )
        return pd.Series(preds, index=idx, name="forecast")

    # ------------------------------------------------------------------ #
    # tiny fallback trainer
    def _train_fallback(self) -> None:
        closes = self.series.values.astype(np.float32)
        X, y = _build_xy(closes, self.window)
        reg = LinearRegression().fit(X, y)
        self._coef_ = reg.coef_
        self._bias_ = float(reg.intercept_)
        log.info("Fallback linear model fitted in-memory.")


# -------------------------------------------------------------------------- #
def _build_xy(arr: Sequence[float], window: int) -> tuple[np.ndarray, np.ndarray]:
    """Rolling-window design matrix."""
    X, y = [], []
    for i in range(window, len(arr) - 1):
        X.append(arr[i - window : i])
        y.append(arr[i])
    return np.asarray(X), np.asarray(y)
