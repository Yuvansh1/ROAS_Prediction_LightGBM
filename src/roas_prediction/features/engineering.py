"""
Feature engineering for the ROAS prediction pipeline.

Transforms raw daily revenue + spend data into a supervised-learning
feature matrix by adding:
  - Calendar encodings (day-of-week, month)
  - Auto-regressive lags (revenue_lag_1, revenue_lag_7)
  - Rolling mean (revenue_roll_mean_7)
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from roas_prediction.utils.audit import AuditLogger

logger = logging.getLogger(__name__)

FEATURE_COLUMNS = [
    "marketing_spend",
    "is_holiday",
    "day_of_week",
    "month",
    "revenue_lag_1",
    "revenue_lag_7",
    "revenue_roll_mean_7",
]


class FeatureEngineer:
    """Adds temporal and lag features to the raw DataFrame."""

    def __init__(
        self,
        lag_days: list[int] | None = None,
        rolling_windows: list[int] | None = None,
    ) -> None:
        self.lag_days = lag_days or [1, 7]
        self.rolling_windows = rolling_windows or [7]

    def transform(
        self,
        df: pd.DataFrame,
        audit_logger: AuditLogger | None = None,
    ) -> pd.DataFrame:
        """
        Apply all feature transformations.

        Args:
            df: Raw DataFrame produced by DataGenerator (must contain 'date' and 'revenue').
            audit_logger: Optional AuditLogger for recording this step.

        Returns:
            Processed DataFrame (NaN rows from lags dropped).
        """
        t0 = time.perf_counter()
        input_rows = len(df)
        logger.info("Engineering features on %d rows...", input_rows)

        out = df.copy()
        features_added: list[str] = []

        # Calendar features
        out["day_of_week"] = out["date"].dt.dayofweek
        out["month"] = out["date"].dt.month
        features_added += ["day_of_week", "month"]

        # Lag features
        for lag in self.lag_days:
            col = f"revenue_lag_{lag}"
            out[col] = out["revenue"].shift(lag)
            features_added.append(col)

        # Rolling mean features (lag-1 anchored to avoid leakage)
        for window in self.rolling_windows:
            col = f"revenue_roll_mean_{window}"
            out[col] = out["revenue"].shift(1).rolling(window=window).mean()
            features_added.append(col)

        out = out.dropna().reset_index(drop=True)
        output_rows = len(out)
        elapsed = time.perf_counter() - t0

        logger.info(
            "Feature engineering done: %d -> %d rows, %.3fs",
            input_rows,
            output_rows,
            elapsed,
        )

        if audit_logger is not None:
            audit_logger.log_feature_engineering(
                input_rows=input_rows,
                output_rows=output_rows,
                rows_dropped=input_rows - output_rows,
                features_added=features_added,
                duration_seconds=elapsed,
            )

        return out
