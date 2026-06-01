"""Tests for ROASPredictor and compute_metrics."""

import pytest
import numpy as np
import pandas as pd
from roas_prediction.data.generator import DataGenerator
from roas_prediction.features.engineering import FeatureEngineer
from roas_prediction.models.lgbm_model import ROASPredictor
from roas_prediction.evaluation.metrics import compute_metrics


@pytest.fixture(scope="module")
def processed_df():
    raw = DataGenerator(start_date="2024-01-01", end_date="2024-12-31", seed=42).generate()
    return FeatureEngineer().transform(raw)


def test_train_returns_metrics(processed_df):
    predictor = ROASPredictor()
    metrics = predictor.train(processed_df, test_days=30, num_boost_round=50)
    assert set(metrics.keys()) == {"rmse", "mae", "r2", "mape"}


def test_metrics_are_finite(processed_df):
    predictor = ROASPredictor()
    metrics = predictor.train(processed_df, test_days=30, num_boost_round=50)
    for key, val in metrics.items():
        assert np.isfinite(val), f"Metric {key} is not finite: {val}"


def test_rmse_positive(processed_df):
    predictor = ROASPredictor()
    metrics = predictor.train(processed_df, test_days=30, num_boost_round=50)
    assert metrics["rmse"] > 0


def test_r2_reasonable(processed_df):
    predictor = ROASPredictor()
    metrics = predictor.train(processed_df, test_days=30, num_boost_round=50)
    # R² should be at least 0.5 for a well-structured synthetic dataset
    assert metrics["r2"] > 0.5


def test_predict_shape(processed_df):
    predictor = ROASPredictor()
    predictor.train(processed_df, test_days=30, num_boost_round=50)
    X = processed_df.head(10)
    preds = predictor.predict(X)
    assert preds.shape == (10,)


def test_predict_before_train_raises():
    predictor = ROASPredictor()
    with pytest.raises(RuntimeError, match="not trained"):
        predictor.predict(pd.DataFrame())


def test_compute_metrics_perfect():
    y = np.array([1.0, 2.0, 3.0])
    m = compute_metrics(y, y)
    assert m["rmse"] == pytest.approx(0.0)
    assert m["mae"] == pytest.approx(0.0)
    assert m["r2"] == pytest.approx(1.0)
    assert m["mape"] == pytest.approx(0.0)


def test_model_save_load(processed_df, tmp_path):
    predictor = ROASPredictor()
    predictor.train(processed_df, test_days=30, num_boost_round=50)
    path = tmp_path / "model.txt"
    predictor.save(path)
    assert path.exists()

    new_predictor = ROASPredictor()
    new_predictor.load(path)
    preds = new_predictor.predict(processed_df.head(5))
    assert preds.shape == (5,)
