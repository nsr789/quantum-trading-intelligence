"""
Train a tiny LinearRegression on 5y closes and export to quantised ONNX.
Run:  python -m src.ml.train
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pandas as pd
import skl2onnx
from onnxruntime.quantization import quantize_dynamic  # type: ignore
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from src.data.market import get_price_history

TICKER = "AAPL"
WINDOW = 30
OUT_PATH = Path(__file__).resolve().parent / "models" / "price.onnx"
OUT_PATH.parent.mkdir(exist_ok=True)

df = get_price_history(TICKER, start="2020-01-01")["Close"]
X, y = [], []
for i in range(WINDOW, len(df) - 1):
    X.append(df.iloc[i - WINDOW : i].values)
    y.append(df.iloc[i + 1])
X, y = np.array(X), np.array(y)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
model = LinearRegression().fit(X_scaled, y)

# export – first to float ONNX
onnx_model = convert_sklearn(
    model,
    initial_types=[("float_input", FloatTensorType([None, WINDOW]))],
    target_opset=16,
)
onnx.save_model(onnx_model, OUT_PATH.with_suffix(".tmp.onnx"))

# quantise to INT8 → smaller & faster
quantize_dynamic(
    model_input=str(OUT_PATH.with_suffix(".tmp.onnx")),
    model_output=str(OUT_PATH),
    per_channel=False,
    weight_type=ort.quantization.QuantType.QInt8,
)
OUT_PATH.with_suffix(".tmp.onnx").unlink()  # cleanup
print(f"Saved quantised model → {OUT_PATH}")
