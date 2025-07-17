import pandas as pd
from src.ml.predictor import PricePredictor

def test_predictor_fallback():
    pred = PricePredictor("AAPL", window=10)
    val = pred.predict_next_close()
    assert isinstance(val, float) and val > 0

    series = pred.predict_series(horizon=3)
    assert isinstance(series, pd.Series) and len(series) == 3
