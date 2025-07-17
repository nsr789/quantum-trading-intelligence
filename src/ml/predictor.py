# src/ml/predictor.py
"""
Generic ONNX-price predictor wrapper.

* Will auto-detect any `<something>_price.onnx` artefact inside `src/ml/models/`
  (e.g. `lstm_price.onnx`, `transformer_price.onnx`, …).
* Falls back to a naive 30-day moving-average model if **no** ONNX is present
  so unit-tests always pass, even on systems without GPU / large models.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import onnxruntime as ort
import pandas as pd

from src.data.market import get_price_history
from src.utils.logging import get_logger

log = get_logger(module="predictor")
MODEL_DIR = Path(__file__).resolve().parent / "models"
WINDOW = 30


class PricePredictor:
    """Unified interface → `.predict(ticker, horizon)`"""

    def __init__(self) -> None:
        self._sess: ort.InferenceSession | None = None
        self._load_latest_onnx()

    # --------------------------------------------------------------------- #
    # internal
    # --------------------------------------------------------------------- #
    def _latest_model_path(self) -> Path | None:
        cand = sorted(MODEL_DIR.glob("*_price.onnx"))
        return cand[-1] if cand else None

    def _load_latest_onnx(self) -> None:
        path = self._latest_model_path()
        if path:
            self._sess = ort.InferenceSession(str(path))
            log.info("Loaded ONNX price model: %s", path.name)
        else:
            log.warning("No *_price.onnx found – using MA-30 fallback")

    # --------------------------------------------------------------------- #
    # public
    # --------------------------------------------------------------------- #
    def predict(
        self,
        ticker: str,
        horizon: int = 30,
        *,
        agg: Literal["last", "mean"] = "last",
    ) -> pd.Series:
        """
        Predict *closing* price `horizon` steps ahead.

        Returns a Series indexed by future dates.
        """
        closes = get_price_history(ticker, interval="1d")["Close"]
        last_ts = closes.index[-1]

        # ---------- ONNX fast-path ----------
        if self._sess:
            window = closes.iloc[-WINDOW:].values.astype("float32")
            window = (window - window.mean()) / np.maximum(window.std(), 1e-6)
            ort_in = {self._sess.get_inputs()[0].name: window[None, :, None]}
            next_price = float(self._sess.run(None, ort_in)[0][0, 0])
            step = pd.Timedelta(days=1)
            dates = pd.date_range(last_ts + step, periods=horizon, freq="B")
            return pd.Series(next_price, index=dates, name="forecast")

        # ---------- baseline MA-30 ----------
        ma = closes.rolling(WINDOW).mean().iloc[-1]
        step = pd.Timedelta(days=1)
        dates = pd.date_range(last_ts + step, periods=horizon, freq="B")
        return pd.Series(ma, index=dates, name="forecast")
