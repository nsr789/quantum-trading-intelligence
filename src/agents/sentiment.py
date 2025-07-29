"""Transformer sentiment helper – FinBERT (finance-tuned)."""
from __future__ import annotations
from typing import List

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline,
)

_MODEL_NAME = "ProsusAI/finbert"
_DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

_tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME)
_model     = AutoModelForSequenceClassification.from_pretrained(_MODEL_NAME).to(_DEVICE)
_pipe      = pipeline(
    task="sentiment-analysis",
    model=_model,
    tokenizer=_tokenizer,
    device=0 if _DEVICE == "cuda" else -1,
    return_all_scores=False,
)

def transformer_sentiment(texts: List[str]) -> List[str]:
    """Map list[str] → list[label].  Labels: positive / negative / neutral."""
    out = _pipe(texts, truncation=True, max_length=512, batch_size=8)
    return [o["label"].lower() for o in out]
