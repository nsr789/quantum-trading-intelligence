"""Headline‑level sentiment classifier (XGBoost ➔ ONNX).
Trains once on a small built‑in corpus if no ONNX is present so the test suite
stays offline.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import onnx
import onnxruntime as ort
import pandas as pd
import xgboost as xgb
from joblib import dump, load
from sklearn.feature_extraction.text import TfidfVectorizer
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import StringTensorType

from src.utils.logging import get_logger

log = get_logger(module="xgb")
MODEL_DIR = Path(__file__).resolve().parent / "models"
MODEL_DIR.mkdir(exist_ok=True)

CORPUS = [
    ("Apple reports record profits", 1),
    ("iPhone battery issues spark complaints", -1),
    ("Microsoft launches new AI tools", 1),
    ("Regulatory fine hits Google", -1),
]

class HeadlineSentiment:
    def __init__(self):
        self.onnx_path = MODEL_DIR / "headline_sentiment.onnx"
        if self.onnx_path.exists():
            self._sess = ort.InferenceSession(str(self.onnx_path))
            self.vectorizer = load(self.onnx_path.with_suffix(".vec.joblib"))
        else:
            self._train_fallback()

    def _train_fallback(self):
        texts, labels = zip(*CORPUS)
        self.vectorizer = TfidfVectorizer(min_df=1).fit(texts)
        X = self.vectorizer.transform(texts)
        y = np.array([1 if l > 0 else 0 for l in labels])
        clf = xgb.XGBClassifier(max_depth=3, n_estimators=20).fit(X, y)

        # export ONNX
        initial = [("text_input", StringTensorType([None, 1]))]
        onnx_model = convert_sklearn(clf, initial_types=initial, target_opset=16)
        onnx.save_model(onnx_model, self.onnx_path)
        dump(self.vectorizer, self.onnx_path.with_suffix(".vec.joblib"))
        self._sess = ort.InferenceSession(str(self.onnx_path))
        log.info("XGBoost sentiment model exported → {}", self.onnx_path.name)

    def score(self, titles: List[str]) -> List[float]:
        X = self.vectorizer.transform(titles)
        ort_in = {self._sess.get_inputs()[0].name: X.astype("object")}
        probs = self._sess.run(None, ort_in)[1]  # probability tensor
        return probs[:, 1].tolist()
