"""
ROAS Prediction — main pipeline entry point.

Usage
-----
  python main.py                        # uses configs/model_config.yaml
  python main.py --config path/to.yaml  # custom config
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

from roas_prediction.data.generator import DataGenerator
from roas_prediction.features.engineering import FeatureEngineer
from roas_prediction.models.lgbm_model import ROASPredictor
from roas_prediction.utils.audit import AuditLogger
from roas_prediction.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)


def load_config(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def run_pipeline(config: dict) -> None:
    log_cfg = config.get("logging", {})
    setup_logging(
        level=log_cfg.get("level", "INFO"),
        log_file=log_cfg.get("app_log_path"),
    )

    audit = AuditLogger(log_path=log_cfg.get("audit_log_path", "logs/audit.jsonl"))

    # ── 1. Generate data ────────────────────────────────────────────────
    data_cfg = config.get("data", {})
    generator = DataGenerator(
        start_date=data_cfg.get("start_date", "2024-01-01"),
        end_date=data_cfg.get("end_date", "2025-12-31"),
        seed=data_cfg.get("seed", 42),
        marketing_spend_range=tuple(data_cfg.get("marketing_spend_range", [5000, 25000])),
        baseline_range=tuple(data_cfg.get("baseline_range", [30000, 45000])),
        noise_std=data_cfg.get("noise_std", 2000),
        holiday_months=data_cfg.get("holiday_months", [11, 12]),
    )
    raw_df = generator.generate(audit_logger=audit)

    # ── 2. Feature engineering ──────────────────────────────────────────
    feat_cfg = config.get("features", {})
    engineer = FeatureEngineer(
        lag_days=feat_cfg.get("lag_days", [1, 7]),
        rolling_windows=feat_cfg.get("rolling_windows", [7]),
    )
    processed_df = engineer.transform(raw_df, audit_logger=audit)

    # ── 3. Train model ──────────────────────────────────────────────────
    model_cfg = config.get("model", {})
    train_cfg = config.get("training", {})

    predictor = ROASPredictor(params=model_cfg)

    output_dir = Path(train_cfg.get("output_dir", "outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = str(output_dir / "lgbm_roas_model.txt") if train_cfg.get("save_model", True) else None

    metrics = predictor.train(
        df=processed_df,
        test_days=train_cfg.get("test_days", 90),
        num_boost_round=train_cfg.get("num_boost_round", 500),
        early_stopping_rounds=train_cfg.get("early_stopping_rounds", 20),
        audit_logger=audit,
        model_save_path=model_path,
    )

    # ── 4. Print results ────────────────────────────────────────────────
    test_df = predictor._test_df

    print("\n" + "=" * 57)
    print("          ROAS FORECAST RESULTS (HOLDOUT SAMPLE)        ")
    print("=" * 57)
    print(
        test_df[["date", "marketing_spend", "actual_roas", "predicted_roas"]]
        .head(10)
        .to_string(index=False)
    )
    print("=" * 57)
    print(f"  RMSE  : ${metrics['rmse']:>12,.2f}")
    print(f"  MAE   : ${metrics['mae']:>12,.2f}")
    print(f"  R2    :  {metrics['r2']:>12.4f}")
    print(f"  MAPE  :  {metrics['mape']:>11.2f}%")
    print("=" * 57)
    print(f"\nAudit log -> {audit.log_path.resolve()}")
    if model_path:
        print(f"Model    -> {Path(model_path).resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="ROAS Prediction Pipeline")
    parser.add_argument(
        "--config",
        default="configs/model_config.yaml",
        help="Path to YAML configuration file.",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    config = load_config(config_path)
    run_pipeline(config)


if __name__ == "__main__":
    main()
