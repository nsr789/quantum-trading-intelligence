# tests/test_predictor.py
import pytest
from src.ml.predictor import PricePredictor


@pytest.mark.parametrize("engine", ["linear", "xgb", "lstm"])
def test_predictor_engines(engine):
    """Every back-end returns a positive float."""
    pred = PricePredictor("AAPL", window=10, engine=engine)
    val  = pred.predict_next_close()
    assert isinstance(val, float) and val > 0


def test_predictor_series_shape():
    pred = PricePredictor("AAPL", window=12, engine="linear")
    s    = pred.predict_series(horizon=5)
    assert len(s) == 5 and s.isna().sum() == 0
