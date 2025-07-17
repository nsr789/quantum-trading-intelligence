# ──────────────────────────────────────────────── src/forecast/lstm_forecaster.py
"""Lightweight LSTM forecaster (CPU‐only, ≤ few dozen epochs)."""

from __future__ import annotations
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from typing import Sequence


class _LSTM(nn.Module):
    def __init__(self, n_features: int = 1, hidden: int = 32, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
        )
        self.lin = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.lin(out[:, -1])


def _create_dataset(series: Sequence[float], lookback: int):
    X, y = [], []
    for i in range(len(series) - lookback):
        X.append(series[i : i + lookback])
        y.append(series[i + lookback])
    X = np.array(X, dtype="float32")[..., None]  # (samples, lookback, 1)
    y = np.array(y, dtype="float32")[:, None]    # (samples, 1)
    return torch.from_numpy(X), torch.from_numpy(y)


def lstm_forecast(
    series: pd.Series,
    horizon: int = 5,
    lookback: int = 120,
    epochs: int = 15,
) -> list[float]:
    """Return list[float] forecast of length *horizon*."""
    # ── prep ─────────────────────────────────────────────────────────────────
    device = torch.device("cpu")
    series_f = series[-(lookback * 3) :].values.astype("float32")  # keep ≈3× look-back
    scaler = series_f.mean(), series_f.std() or 1.0
    s_norm = (series_f - scaler[0]) / scaler[1]
    X, y = _create_dataset(s_norm, lookback)

    # ── model ───────────────────────────────────────────────────────────────
    net = _LSTM().to(device)
    optim = torch.optim.Adam(net.parameters(), lr=1e-2)
    loss_fn = nn.MSELoss()

    net.train()
    for _ in range(epochs):
        optim.zero_grad()
        pred = net(X)
        loss = loss_fn(pred, y)
        loss.backward()
        optim.step()

    # ── recursive forecast ─────────────────────────────────────────────────
    net.eval()
    last_window = torch.tensor(s_norm[-lookback:], dtype=torch.float32)[None, :, None]
    preds = []
    with torch.no_grad():
        for _ in range(horizon):
            nxt = net(last_window).item()
            preds.append(nxt)
            last_window = torch.cat(
                [last_window[:, 1:], torch.tensor([[[nxt]]])], dim=1
            )

    # inverse transform
    preds = [p * scaler[1] + scaler[0] for p in preds]
    return preds
