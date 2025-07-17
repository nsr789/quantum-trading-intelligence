"""Minimal LSTM forecaster exported to ONNX.
Trains on the *same* closing‑price window logic used by `PricePredictor` so the
calling code never changes.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pandas as pd
import torch
from joblib import dump
from sklearn.preprocessing import StandardScaler
from torch import nn

from src.data.market import get_price_history
from src.utils.logging import get_logger

log = get_logger(module="lstm")
MODEL_DIR = Path(__file__).resolve().parent / "models"
MODEL_DIR.mkdir(exist_ok=True)

WINDOW = 30
HIDDEN = 32
class _LSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=HIDDEN, num_layers=1, batch_first=True)
        self.fc = nn.Linear(HIDDEN, 1)

    def forward(self, x):  # x.shape = (B, T, 1)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


def export_onnx(ticker: str = "AAPL", epochs: int = 4):
    # 5y of closes
    df = get_price_history(ticker, start="2019-01-01")["Close"]
    X, y = [], []
    for i in range(WINDOW, len(df) - 1):
        X.append(df.iloc[i - WINDOW : i].values)
        y.append(df.iloc[i + 1])
    X, y = np.array(X), np.array(y)

    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)[:, :, None]  # add channel dim

    model = _LSTM()
    criterion = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    tensor_x = torch.tensor(X_scaled, dtype=torch.float32)
    tensor_y = torch.tensor(y, dtype=torch.float32)[:, None]

    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        pred = model(tensor_x)
        loss = criterion(pred, tensor_y)
        loss.backward()
        opt.step()
    log.info("Finished LSTM training with final loss={:.4f}", loss.item())

    # export to ONNX so we can reuse generic predictor wrapper
    onnx_path = MODEL_DIR / "lstm_price.onnx"
    torch.onnx.export(
        model,
        torch.randn(1, WINDOW, 1),
        str(onnx_path),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}},
        opset_version=16,
    )
    dump(scaler, onnx_path.with_suffix(".scaler.joblib"))
    log.info("Exported → {}", onnx_path)
