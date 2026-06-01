"""
Synthetic luxury-retail data generator.

Produces a daily DataFrame with realistic marketing spend, revenue (via a
logarithmic diminishing-returns curve + seasonal holiday peak), and a
binary holiday indicator.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from roas_prediction.utils.audit import AuditLogger

logger = logging.getLogger(__name__)


class DataGenerator:
    """Generates synthetic luxury-retail marketing data."""

    def __init__(
        self,
        start_date: str = "2024-01-01",
        end_date: str = "2025-12-31",
        seed: int = 42,
        marketing_spend_range: tuple[float, float] = (5_000, 25_000),
        baseline_range: tuple[float, float] = (30_000, 45_000),
        noise_std: float = 2_000,
        holiday_months: list[int] | None = None,
    ) -> None:
        self.start_date = start_date
        self.end_date = end_date
        self.seed = seed
        self.marketing_spend_range = marketing_spend_range
        self.baseline_range = baseline_range
        self.noise_std = noise_std
        self.holiday_months = holiday_months or [11, 12]

    def generate(
        self,
        audit_logger: AuditLogger | None = None,
    ) -> pd.DataFrame:
        """
        Generate synthetic data.

        Returns:
            DataFrame with columns: date, revenue, marketing_spend, is_holiday.
        """
        t0 = time.perf_counter()
        logger.info(
            "Generating synthetic data [%s -> %s, seed=%d]",
            self.start_date,
            self.end_date,
            self.seed,
        )

        rng = np.random.default_rng(self.seed)
        dates = pd.date_range(start=self.start_date, end=self.end_date, freq="D")
        n = len(dates)

        marketing_spend = rng.uniform(*self.marketing_spend_range, size=n)
        baseline = np.linspace(*self.baseline_range, num=n)
        diminishing_impact = np.log(marketing_spend) * 12_000
        noise = rng.normal(0, self.noise_std, size=n)

        df = pd.DataFrame(
            {
                "date": dates,
                "revenue": baseline + diminishing_impact + noise,
                "marketing_spend": marketing_spend,
            }
        )
        df["is_holiday"] = df["date"].dt.month.isin(self.holiday_months).astype(int)

        elapsed = time.perf_counter() - t0
        logger.info("Generated %d rows in %.3fs", len(df), elapsed)

        if audit_logger is not None:
            audit_logger.log_data_generation(
                start_date=self.start_date,
                end_date=self.end_date,
                seed=self.seed,
                rows_generated=len(df),
                columns=list(df.columns),
                config={
                    "marketing_spend_range": list(self.marketing_spend_range),
                    "baseline_range": list(self.baseline_range),
                    "noise_std": self.noise_std,
                    "holiday_months": self.holiday_months,
                },
            )

        return df
