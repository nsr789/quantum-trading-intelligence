# src/ml/predictor.py
"""
Lightweight ONNX runtime wrapper for next-day price forecasts.

If an ONNX model for the requested `engine` / `window` combo is missing
(e.g. in a fresh Git clone) we fall back to a *very* small on-the-fly
linear regression so the Streamlit UI never crashes.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal, Sequence

import numpy as np
import onnxruntime as ort
import pandas as pd

from src.data.market import get_price_history
from src.utils.logging import get_logger

Engine = Literal["linear", "xgb", "lstm"]

MODEL_DIR = Path(__file__).resolve().parent / "models"
log = get_logger(module="predictor")


class PricePredictor:
    """Run-time inference helper – no training, just ONNX execution."""

    def __init__(self, ticker: str, window: int = 30, engine: Engine = "linear"):
        self.ticker = ticker.upper()
        self.window = int(window)
        self.engine: Engine = engine

        self.onnx_path = MODEL_DIR / f"{engine}-{window}.onnx"
        self.scaler_mu = MODEL_DIR / f"scaler-{window}.npy"
        self.scaler_sd = MODEL_DIR / f"scale-{window}.npy"

        if self.onnx_path.exists():
            self._sess = ort.InferenceSession(str(self.onnx_path))
            log.info("ONNX model loaded: {}", self.onnx_path.name)
        else:
            log.warning(
                "ONNX model for {} not found – training fallback.", self.engine
            )
            self._train_fallback()

        # fetch historical series once
        self.series = get_price_history(self.ticker, interval="1d")["Close"]

        # ensure scaler parameters exist (linear / xgb)
        if self.scaler_mu.exists() and self.scaler_sd.exists():
            self.mean_ = np.load(self.scaler_mu)
            self.scale_ = np.load(self.scaler_sd)
        else:  # allow LSTM or fresh fallback
            self.mean_ = None
            self.scale_ = None

    # --------------------------------------------------------------------- #
    #                               helpers                                 #
    # --------------------------------------------------------------------- #
    def _windowed(self) -> np.ndarray:
        """Return last `window` closes as (1, window) tensor."""
        arr = self.series.iloc[-self.window :].values.astype(np.float32).reshape(1, -1)
        if self.mean_ is not None:
            arr = (arr - self.mean_) / self.scale_
        return arr

    # --------------------------------------------------------------------- #
    #                              inference                                #
    # --------------------------------------------------------------------- #
    def predict_next_close(self) -> float:
        if hasattr(self, "_sess"):
            inputs = {self._sess.get_inputs()[0].name: self._windowed()}
            out = self._sess.run(None, inputs)[0].item()
            return float(out)
        # fallback model
        x = self._windowed()
        return float(self._coef_ @ x.ravel() + self._bias_)

    def predict_series(self, horizon: int = 5) -> pd.Series:
        """Recursive multi-step forecast (very rough)."""
        preds: list[float] = []
        temp = self.series.copy()
        for _ in range(horizon):
            self.series = temp  # override for sliding window
            nxt = self.predict_next_close()
            preds.append(nxt)
            temp = pd.concat([temp, pd.Series([nxt])])
        self.series = temp  # restore latest append
        idx = pd.date_range(start=self.series.index[-1] + pd.Timedelta(days=1),
                            periods=horizon, freq="B")
        return pd.Series(preds, index=idx, name="forecast")

    # --------------------------------------------------------------------- #
    #                        tiny linear fallback                            #
    # --------------------------------------------------------------------- #
    def _train_fallback(self) -> None:
        """2-liner OLS so the UI still works without pretrained files."""
        from sklearn.linear_model import LinearRegression

        closes = self.series.values.astype(np.float32)
        X, y = _build_xy(closes, self.window)
        reg = LinearRegression().fit(X, y)
        self._coef_: np.ndarray = reg.coef_
        self._bias_: float = float(reg.intercept_)
        log.info("Fallback linear model fitted in-memory.")


# ========================= util outside the class =========================== #
def _build_xy(arr: Sequence[float], window: int) -> tuple[np.ndarray, np.ndarray]:
    """Simple rolling-window design matrix builder (np version)."""
    X, y = [], []
    for i in range(window, len(arr) - 1):
        X.append(arr[i - window : i])
        y.append(arr[i])
    return np.asarray(X), np.asarray(y)
