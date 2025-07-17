# src/ml/predictor.py
"""PricePredictor – linear / XGBoost / LSTM fore-casting with ONNX fallback."""
from __future__ import annotations
from pathlib import Path
from typing import Literal, Union

import numpy as np
import pandas as pd
import onnxruntime as ort
from sklearn.preprocessing import StandardScaler
from joblib import dump, load

from src.data.market import get_price_history
from src.utils.logging import get_logger

# ───────── optional heavy deps – imported lazily ──────────
try:
    import torch, torch.nn as nn          # noqa: F401
except Exception:                         # pragma: no cover
    torch = None                          # type: ignore

try:
    import xgboost as xgb                 # noqa: F401
except Exception:                         # pragma: no cover
    xgb = None                            # type: ignore
# ───────────────────────────────────────────────────────────

log          = get_logger(module="predictor")
MODEL_DIR    = Path(__file__).resolve().parent / "models"
MODEL_DIR.mkdir(exist_ok=True)

EngineT = Literal["linear", "xgb", "lstm"]


class _TorchLSTM(nn.Module):                   # type: ignore
    def __init__(self, window: int):
        super().__init__()
        self.lstm = nn.LSTM(1, 32, batch_first=True)
        self.fc   = nn.Linear(32, 1)
        self.window = window

    def forward(self, x):
        x, _ = self.lstm(x)
        return self.fc(x[:, -1, :])


class PricePredictor:
    """Infer next close using chosen ML engine; all support ONNX inference."""

    def __init__(self, ticker: str, window: int = 30, engine: EngineT = "linear"):
        self.ticker, self.window, self.engine = ticker.upper(), window, engine
        self.scaler = StandardScaler()
        self.onnx_path = MODEL_DIR / f"{engine}-{window}.onnx"

        if self.onnx_path.exists():
            self._sess = ort.InferenceSession(str(self.onnx_path))
            log.info("ONNX model loaded: {}", self.onnx_path.name)
        else:
            log.warning("ONNX model for {} not found – training fallback", engine)
            self._train_fallback()

    # ── model-specific training paths ────────────────────────────────────
    def _train_fallback(self) -> None:
        """Train chosen engine in-memory, export nothing (used in CI/offline)."""
        series = self._load_or_synth()
        X, y   = self._build_xy(series)

        if self.engine == "xgb" and xgb is not None:
            self._model = xgb.XGBRegressor(objective="reg:squarederror").fit(X, y)
        elif self.engine == "lstm" and torch is not None:
            X_t = torch.tensor(X.reshape(-1, self.window, 1), dtype=torch.float32)
            y_t = torch.tensor(y.reshape(-1, 1),            dtype=torch.float32)
            net = _TorchLSTM(self.window)
            opt = torch.optim.Adam(net.parameters(), lr=1e-2)
            for _ in range(100):                            # tiny epoch loop
                opt.zero_grad(); loss = ((net(X_t)-y_t)**2).mean(); loss.backward(); opt.step()
            self._model = net.eval()
        else:                                               # default linear
            from sklearn.linear_model import LinearRegression
            self._model = LinearRegression().fit(X, y)

    # ── public API ───────────────────────────────────────────────────────
    def predict_next_close(self) -> float:
        feats = self._latest_window().astype(np.float32)
        if hasattr(self, "_sess"):                          # ONNX path
            return float(self._sess.run(None, {self._sess.get_inputs()[0].name: feats})[0])
        # python model path
        if self.engine == "lstm":
            with torch.no_grad():
                x = torch.tensor(feats.reshape(1, self.window, 1))
                return float(self._model(x).item())
        return float(self._model.predict(feats)[0])

    def predict_series(self, horizon: int = 5) -> pd.Series:
        out = []
        for _ in range(horizon):
            nxt = self.predict_next_close()
            out.append(nxt)
            self._roll(nxt)
        idx = pd.date_range(periods=horizon, freq="B", start=pd.Timestamp.utcnow())
        return pd.Series(out, index=idx, name="pred_close")

    # ── helpers ───────────────────────────────────────────────────────────
    def _load_or_synth(self) -> pd.Series:
        try:
            return get_price_history(self.ticker, start="2022-01-01")["Close"]
        except Exception as exc:
            log.warning("Price download failed – synthetic ramp used: {}", exc)
            return pd.Series(range(self.window + 80), dtype=float)

    def _build_xy(self, series: pd.Series):
        X, y = [], []
        for i in range(self.window, len(series) - 1):
            X.append(series.iloc[i - self.window:i].values)
            y.append(series.iloc[i + 1])
        X = np.array(X);  y = np.array(y)
        if self.engine != "lstm":           # scale only tabular models
            X = self.scaler.fit_transform(X)
        return X, y

    def _latest_window(self) -> np.ndarray:
        close = get_price_history(self.ticker).iloc[-self.window:]["Close"].values
        if self.engine != "lstm":
            close = self.scaler.transform(close.reshape(1, -1))
        return close.reshape(1, -1)

    def _roll(self, nxt: float):
        """Update state if running python model (ONNX needs none)."""
        # simple – rely on fresh get_price_history each call
        pass
