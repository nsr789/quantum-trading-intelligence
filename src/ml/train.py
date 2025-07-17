# src/ml/train.py
"""
Train all forecasting back-ends (Linear → ONNX, XGBoost → ONNX, LSTM → ONNX).

Run from project root:

    python -m src.ml.train --ticker AAPL --window 30

The exported models are written to  **src/ml/models/{engine}-{window}.onnx**
so that `PricePredictor` can pick them up at runtime without re-training.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import onnx
import onnxruntime as ort
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from onnxmltools import convert_sklearn, convert_xgboost
from onnxmltools.utils import save_model
from skl2onnx.common.data_types import FloatTensorType

from src.data.market import get_price_history
from src.utils.logging import get_logger

# ───────── optional heavy deps ────────────────────────────────────────────────
try:                                         # PyTorch for LSTM
    import torch
    import torch.nn as nn
except Exception:
    torch = None  # type: ignore

try:                                         # XGBoost
    import xgboost as xgb
except Exception:
    xgb = None  # type: ignore
# ──────────────────────────────────────────────────────────────────────────────

log        = get_logger(module="train")
MODEL_DIR  = Path(__file__).resolve().parent / "models"
MODEL_DIR.mkdir(exist_ok=True)


# ─────────────────────────── helpers ──────────────────────────────────────────
def build_xy(series: pd.Series, window: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return rolling feature matrix X and next-day target y."""
    X, y = [], []
    for i in range(window, len(series) - 1):
        X.append(series.iloc[i - window : i].values)
        y.append(series.iloc[i])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


class LSTMNet(nn.Module):                    # type: ignore[name-defined]
    def __init__(self, window: int):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=32, batch_first=True)
        self.fc   = nn.Linear(32, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


def export_lstm_to_onnx(model: nn.Module, window: int, onnx_path: Path) -> None:  # type: ignore[name-defined]
    model.eval()
    dummy = torch.randn(1, window, 1)
    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        input_names=["input"],
        output_names=["pred"],
        dynamic_axes={"input": {0: "batch"}, "pred": {0: "batch"}},
        opset_version=17,
    )
    log.info("✅  LSTM exported → {}", onnx_path.name)


# ─────────────────────────── main training routine ───────────────────────────
def train_and_export(ticker: str, window: int) -> None:
    series  = get_price_history(ticker, start="2020-01-01")["Close"]
    X, y    = build_xy(series, window)

    # ── Linear  ────────────────────────────────────────────────────────────
    scaler  = StandardScaler()
    X_std   = scaler.fit_transform(X)
    lin_reg = LinearRegression().fit(X_std, y)

    initial_type = [("input", FloatTensorType([None, window]))]
    onx = convert_sklearn(lin_reg, initial_types=initial_type)
    save_model(onx, MODEL_DIR / f"linear-{window}.onnx")
    log.info("✅  Linear ONNX saved.")

    # store scaler for runtime use
    np.save(MODEL_DIR / f"scaler-{window}.npy", scaler.mean_)
    np.save(MODEL_DIR / f"scale-{window}.npy",  scaler.scale_)

    # ── XGBoost  ───────────────────────────────────────────────────────────
    if xgb is not None:
        booster = xgb.XGBRegressor(objective="reg:squarederror")
        booster.fit(X_std, y)
        onx = convert_xgboost(booster, initial_types=initial_type, opset=17)
        save_model(onx, MODEL_DIR / f"xgb-{window}.onnx")
        log.info("✅  XGBoost ONNX saved.")
    else:
        log.warning("❌  XGBoost not installed – skipping.")

    # ── LSTM  ──────────────────────────────────────────────────────────────
    if torch is not None:
        X_l = torch.tensor(X.reshape(-1, window, 1))
        y_l = torch.tensor(y.reshape(-1, 1))
        net = LSTMNet(window)
        opt = torch.optim.Adam(net.parameters(), lr=1e-2)
        for _ in range(120):
            opt.zero_grad()
            loss = ((net(X_l) - y_l) ** 2).mean()
            loss.backward()
            opt.step()
        export_lstm_to_onnx(net, window, MODEL_DIR / f"lstm-{window}.onnx")
    else:
        log.warning("❌  PyTorch not installed – skipping LSTM.")


# ───────────────────────── CLI entry-point ────────────────────────────────────
def _cli() -> None:
    p = argparse.ArgumentParser(description="Train & export price-forecast models.")
    p.add_argument("--ticker", default="AAPL")
    p.add_argument("--window", type=int, default=30)
    args = p.parse_args()
    train_and_export(args.ticker.upper(), args.window)


if __name__ == "__main__":  # pragma: no cover
    _cli()
