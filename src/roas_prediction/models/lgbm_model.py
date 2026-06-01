"""
LightGBM-based ROAS predictor.

Training strategy
-----------------
* Target:  revenue (log-stable, avoids direct ROAS instability)
* ROAS is derived post-hoc: predicted_revenue / marketing_spend
* Chronological train/test split (no shuffle) to prevent leakage
"""

from __future__ import annotations

import logging
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from roas_prediction.evaluation.metrics import compute_metrics
from roas_prediction.features.engineering import FEATURE_COLUMNS

if TYPE_CHECKING:
    from roas_prediction.utils.audit import AuditLogger

logger = logging.getLogger(__name__)

DEFAULT_PARAMS: dict[str, Any] = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 20,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.9,
    "bagging_freq": 5,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "verbose": -1,
}


class ROASPredictor:
    """Trains and serves a LightGBM revenue-forecasting model."""

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        self.params: dict[str, Any] = {**DEFAULT_PARAMS, **(params or {})}
        self.model: lgb.Booster | None = None
        self.run_id: str = str(uuid.uuid4())
        self.feature_columns: list[str] = FEATURE_COLUMNS

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        df: pd.DataFrame,
        test_days: int = 90,
        num_boost_round: int = 500,
        early_stopping_rounds: int = 20,
        audit_logger: AuditLogger | None = None,
        model_save_path: str | None = None,
    ) -> dict[str, float]:
        """
        Chronological train/test split -> fit LightGBM -> return metrics.

        Args:
            df:                   Processed feature DataFrame (from FeatureEngineer).
            test_days:            Number of most-recent days reserved for evaluation.
            num_boost_round:      Maximum boosting rounds.
            early_stopping_rounds: Patience for early stopping.
            audit_logger:         Optional AuditLogger.
            model_save_path:      If provided, serialise the booster here.

        Returns:
            Metrics dict from compute_metrics().
        """
        self.run_id = str(uuid.uuid4())
        logger.info("Training run %s started", self.run_id)
        t0 = time.perf_counter()

        split_date = df["date"].max() - pd.Timedelta(days=test_days)
        train_df = df[df["date"] <= split_date]
        test_df = df[df["date"] > split_date].copy()

        logger.info(
            "Split: %d train rows | %d test rows (split at %s)",
            len(train_df),
            len(test_df),
            split_date.date(),
        )

        X_train = train_df[self.feature_columns]
        y_train = train_df["revenue"]
        X_test = test_df[self.feature_columns]
        y_test = test_df["revenue"]

        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

        self.model = lgb.train(
            self.params,
            train_data,
            num_boost_round=num_boost_round,
            valid_sets=[val_data],
            callbacks=[
                lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False),
                lgb.log_evaluation(period=-1),
            ],
        )

        # Predict + ROAS derivation
        test_df["predicted_revenue"] = self.model.predict(
            X_test, num_iteration=self.model.best_iteration
        )
        test_df["actual_roas"] = test_df["revenue"] / test_df["marketing_spend"]
        test_df["predicted_roas"] = test_df["predicted_revenue"] / test_df["marketing_spend"]

        metrics = compute_metrics(y_test.values, test_df["predicted_revenue"].values)
        elapsed = time.perf_counter() - t0

        logger.info(
            "Run %s | best_iter=%d | RMSE=%.2f | MAE=%.2f | R2=%.4f | MAPE=%.2f%% | %.1fs",
            self.run_id,
            self.model.best_iteration,
            metrics["rmse"],
            metrics["mae"],
            metrics["r2"],
            metrics["mape"],
            elapsed,
        )

        # Feature importance
        importance = dict(
            zip(
                self.feature_columns,
                self.model.feature_importance(importance_type="gain").tolist(),
            )
        )

        # Optionally save model
        saved_path: str | None = None
        if model_save_path is not None:
            Path(model_save_path).parent.mkdir(parents=True, exist_ok=True)
            self.model.save_model(model_save_path)
            saved_path = model_save_path
            logger.info("Model saved to %s", saved_path)

        if audit_logger is not None:
            predictions_log = test_df[
                ["date", "marketing_spend", "revenue", "predicted_revenue",
                 "actual_roas", "predicted_roas"]
            ].rename(columns={"revenue": "actual_revenue"}).to_dict(orient="records")

            audit_logger.log_training_run(
                run_id=self.run_id,
                train_rows=len(train_df),
                test_rows=len(test_df),
                features=self.feature_columns,
                params=self.params,
                best_iteration=self.model.best_iteration,
                metrics=metrics,
                feature_importance=importance,
                duration_seconds=elapsed,
                model_path=saved_path,
            )
            audit_logger.log_prediction(
                run_id=self.run_id,
                prediction_id=str(uuid.uuid4()),
                rows=len(test_df),
                date_range=(
                    str(test_df["date"].min().date()),
                    str(test_df["date"].max().date()),
                ),
                predictions=predictions_log,
                summary={
                    "mean_actual_roas": float(test_df["actual_roas"].mean()),
                    "mean_predicted_roas": float(test_df["predicted_roas"].mean()),
                    "mean_actual_revenue": float(test_df["revenue"].mean()),
                    "mean_predicted_revenue": float(test_df["predicted_revenue"].mean()),
                },
                duration_seconds=elapsed,
            )

        self._test_df = test_df
        return metrics

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return raw revenue predictions for feature matrix X."""
        if self.model is None:
            raise RuntimeError("Model is not trained yet. Call .train() first.")
        return self.model.predict(X[self.feature_columns], num_iteration=self.model.best_iteration)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        if self.model is None:
            raise RuntimeError("Nothing to save — model not trained.")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(path))
        logger.info("Model saved -> %s", path)

    def load(self, path: str | Path) -> None:
        self.model = lgb.Booster(model_file=str(path))
        logger.info("Model loaded ← %s", path)
