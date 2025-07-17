# src/ml/onnx_io.py
"""
Lightweight ONNX runtime wrapper used by Streamlit + unit-tests.

1. `load_model()` – caches an onnxruntime.InferenceSession
2. `predict_price()` – LSTM or XGBoost regression (single-ticker)
"""

from __future__ import annotations
import functools
from pathlib import Path
from typing import Union, Iterable, Literal

import numpy as np
import onnxruntime as ort
import pandas as pd
import yfinance as yf

from src.utils.logging import get_logger

log = get_logger("onnx")

_MODELS = {
    "lstm": Path(__file__).resolve().parents[1] / "models" / "price_lstm.onnx",
    "xgb": Path(__file__).resolve().parents[1] / "models" / "price_xgb.onnx",
}

@functools.lru_cache(maxsize=None)
def load_model(model: Literal["lstm", "xgb"] = "lstm") -> ort.InferenceSession:
    path = _MODELS[model]
    if not path.exists():
        raise FileNotFoundError(
            f"ONNX model {path.name!s} is missing – train & export first."
        )
    log.info("Loading ONNX model %s", path.name)
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def _fetch_close(ticker: str, n: int = 60) -> np.ndarray:
    """Last n closes, scaled to -1..1 (min-max)."""
    df = yf.download(ticker, period=f"{n+30}d", interval="1d")["Close"].tail(n)
    x = df.values.astype("float32")
    if x.ptp() == 0:
        return np.zeros_like(x)
    return (2 * (x - x.min()) / x.ptp() - 1).reshape(1, -1, 1)  # (1, seq, 1)


def predict_price(
    ticker: str,
    horizon: int = 30,
    model: Literal["lstm", "xgb"] = "lstm",
) -> pd.Series:
    """
    ➜ Returns Series of length `horizon` with predicted close prices.

    *LSTM* model expects input shape (1, seq_len, 1)  
    *XGBoost* model expects flattened lag vector
    """
    sess = load_model(model)

    if model == "lstm":
        x = _fetch_close(ticker)
        ort_inputs = {sess.get_inputs()[0].name: x}
    else:  # xgb
        x = _fetch_close(ticker).reshape(1, -1)
        ort_inputs = {sess.get_inputs()[0].name: x}

    y_hat = sess.run(None, ort_inputs)[0].squeeze()

    idx = pd.date_range(
        start=pd.Timestamp.today().normalize() + pd.Timedelta(days=1),
        periods=horizon,
        freq="B",
    )
    # reverse scale to dollars using last close value
    last_px = yf.download(ticker, period="2d", interval="1d")["Close"].iloc[-1]
    return pd.Series(last_px * (1 + np.cumsum(y_hat)), index=idx, name="pred")
