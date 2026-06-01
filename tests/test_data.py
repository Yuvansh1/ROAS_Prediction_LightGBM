"""Tests for DataGenerator."""

import pytest
import pandas as pd
from roas_prediction.data.generator import DataGenerator


@pytest.fixture
def generator():
    return DataGenerator(start_date="2024-01-01", end_date="2024-03-31", seed=0)


def test_output_columns(generator):
    df = generator.generate()
    assert set(df.columns) == {"date", "revenue", "marketing_spend", "is_holiday"}


def test_row_count(generator):
    df = generator.generate()
    expected = len(pd.date_range("2024-01-01", "2024-03-31", freq="D"))
    assert len(df) == expected


def test_no_nulls(generator):
    df = generator.generate()
    assert df.isnull().sum().sum() == 0


def test_marketing_spend_bounds(generator):
    df = generator.generate()
    assert df["marketing_spend"].between(5_000, 25_000).all()


def test_holiday_indicator(generator):
    df = generator.generate()
    # Jan–Mar: no holidays (months 11, 12)
    assert (df["is_holiday"] == 0).all()


def test_reproducibility():
    g1 = DataGenerator(seed=99)
    g2 = DataGenerator(seed=99)
    pd.testing.assert_frame_equal(g1.generate(), g2.generate())


def test_different_seeds_differ():
    g1 = DataGenerator(seed=1)
    g2 = DataGenerator(seed=2)
    assert not g1.generate()["revenue"].equals(g2.generate()["revenue"])
