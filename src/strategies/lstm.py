from __future__ import annotations

import warnings, os, numpy as np, pandas as pd
from functools import lru_cache
from typing import Final

from .base import Strategy

# ──────────────────────────────────────────
# Optional heavy deps – degrade gracefully
# ──────────────────────────────────────────
try:
    # tf-keras keeps wheel size small (CPU-only); any 2.x is fine
    import tensorflow as tf
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import LSTM, Dense
except ModuleNotFoundError:  # pragma: no cover
    tf = None  # type: ignore
    warnings.warn("tensorflow not installed – LSTMStrategy will fall back to Momentum signals.")


class LSTMStrategy(Strategy):
    """
    Very-small single-layer LSTM that is **trained on-the-fly**
    (3 epochs – fast) and emits long / short trading signals:
        long  if next-day close > today close,
        short otherwise.

    Falls back to SMA-momentum if TensorFlow is unavailable so the
    whole project **never breaks** in minimal environments.
    """

    WINDOW: Final = 10          # look-back timesteps
    EPOCHS: Final = 3           # quick fit (fast in CI)
    _FALLBACK_SMA: Final = 20   # if TF missing

    # ---------------------------
    def generate_signals(self, price: pd.Series) -> pd.Series:
        if tf is None:                             # fallback branch
            roll = price.rolling(self._FALLBACK_SMA).mean()
            sig  = (price > roll).astype(int).replace(0, -1)
            return sig.fillna(0)

        # --- Train tiny LSTM -------------------------------------
        x, y = self._prep_dataset(price.values.astype("float32"))
        model = _cached_model(self.WINDOW, price.values.std())
        model.fit(x, y, epochs=self.EPOCHS, verbose=0)

        # --- Predict next-bar closes -----------------------------
        preds = model.predict(x, verbose=0).ravel()
        today = price.values[self.WINDOW:-1]           # aligned actual
        pos   = np.where(preds > today,  1, -1)        # long / short

        # build aligned Series: first WINDOW+1 bars = flat (0)
        sig = pd.Series(0, index=price.index, dtype="int8")
        sig.iloc[self.WINDOW+1:] = pos
        return sig

    # ---------------------------
    def _prep_dataset(self, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Sliding window → (samples, WINDOW, 1), targets = next close"""
        X, y = [], []
        for i in range(len(p) - self.WINDOW - 1):
            X.append(p[i : i + self.WINDOW])
            y.append(p[i + self.WINDOW])
        return (np.asarray(X)[..., None], np.asarray(y))


# ──────────────────────────────────────────
# Model cache so we *reuse* graph between calls
# speed & deterministic tests
# ──────────────────────────────────────────
@lru_cache(maxsize=4)
def _cached_model(window: int, scale: float) -> "tf.keras.Model":  # type: ignore
    tf.random.set_seed(42)
    model = Sequential([
        LSTM(12, input_shape=(window, 1)),
        Dense(1),
    ])
    model.compile("adam", "mse")
    # scale layer weights down – helps tiny datasets / CI stability
    for w in model.get_weights():
        w[:] = w / (scale if scale else 1.0)
    model.set_weights(model.get_weights())  # re-assign for lru_cache pickling
    return model
