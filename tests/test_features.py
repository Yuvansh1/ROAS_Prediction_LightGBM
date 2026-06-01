"""Tests for FeatureEngineer."""

import pytest
from roas_prediction.data.generator import DataGenerator
from roas_prediction.features.engineering import FeatureEngineer, FEATURE_COLUMNS


@pytest.fixture
def raw_df():
    return DataGenerator(start_date="2024-01-01", end_date="2024-06-30", seed=0).generate()


def test_feature_columns_present(raw_df):
    df = FeatureEngineer().transform(raw_df)
    for col in FEATURE_COLUMNS:
        assert col in df.columns, f"Missing column: {col}"


def test_no_nulls_after_transform(raw_df):
    df = FeatureEngineer().transform(raw_df)
    assert df.isnull().sum().sum() == 0


def test_row_count_reduced(raw_df):
    """Lag-7 + rolling-7 removes at least 7 rows."""
    df = FeatureEngineer().transform(raw_df)
    assert len(df) < len(raw_df)
    assert len(df) == len(raw_df) - 7  # rolling(7) anchored at lag-1 → drops 7 rows


def test_day_of_week_range(raw_df):
    df = FeatureEngineer().transform(raw_df)
    assert df["day_of_week"].between(0, 6).all()


def test_month_range(raw_df):
    df = FeatureEngineer().transform(raw_df)
    assert df["month"].between(1, 12).all()


def test_original_df_unchanged(raw_df):
    original_len = len(raw_df)
    FeatureEngineer().transform(raw_df)
    assert len(raw_df) == original_len  # transform must not mutate input
