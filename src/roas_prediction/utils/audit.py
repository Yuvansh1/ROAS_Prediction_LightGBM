"""
Audit trail logger — every pipeline event is appended as a JSON-Line record.

Schema per record:
  {
    "event_id":   str,       # UUID4
    "timestamp":  str,       # ISO-8601 UTC
    "event_type": str,       # data_generation | feature_engineering | training_run | prediction
    "payload":    dict       # event-specific fields (see method docstrings)
  }
"""

from __future__ import annotations

import json
import logging
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_LOCK = threading.Lock()


class AuditLogger:
    """Append-only, thread-safe JSONL audit logger."""

    def __init__(self, log_path: str | Path = "logs/audit.jsonl") -> None:
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _write(self, event_type: str, payload: dict[str, Any]) -> str:
        record = {
            "event_id": str(uuid.uuid4()),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "payload": payload,
        }
        line = json.dumps(record, default=str)
        with _LOCK:
            with open(self.log_path, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        logger.debug("Audit event written: %s (%s)", event_type, record["event_id"])
        return record["event_id"]

    # ------------------------------------------------------------------
    # Public log methods
    # ------------------------------------------------------------------

    def log_data_generation(
        self,
        *,
        start_date: str,
        end_date: str,
        seed: int,
        rows_generated: int,
        columns: list[str],
        config: dict[str, Any] | None = None,
    ) -> str:
        """Log a synthetic data generation event."""
        return self._write(
            "data_generation",
            {
                "start_date": start_date,
                "end_date": end_date,
                "seed": seed,
                "rows_generated": rows_generated,
                "columns": columns,
                "config": config or {},
            },
        )

    def log_feature_engineering(
        self,
        *,
        input_rows: int,
        output_rows: int,
        rows_dropped: int,
        features_added: list[str],
        duration_seconds: float,
    ) -> str:
        """Log a feature-engineering event."""
        return self._write(
            "feature_engineering",
            {
                "input_rows": input_rows,
                "output_rows": output_rows,
                "rows_dropped": rows_dropped,
                "features_added": features_added,
                "duration_seconds": round(duration_seconds, 4),
            },
        )

    def log_training_run(
        self,
        *,
        run_id: str,
        train_rows: int,
        test_rows: int,
        features: list[str],
        params: dict[str, Any],
        best_iteration: int,
        metrics: dict[str, float],
        feature_importance: dict[str, float],
        duration_seconds: float,
        model_path: str | None = None,
    ) -> str:
        """Log a model training run."""
        return self._write(
            "training_run",
            {
                "run_id": run_id,
                "train_rows": train_rows,
                "test_rows": test_rows,
                "features": features,
                "params": params,
                "best_iteration": best_iteration,
                "metrics": metrics,
                "feature_importance": feature_importance,
                "duration_seconds": round(duration_seconds, 4),
                "model_path": model_path,
            },
        )

    def log_prediction(
        self,
        *,
        run_id: str,
        prediction_id: str,
        rows: int,
        date_range: tuple[str, str],
        predictions: list[dict[str, Any]],
        summary: dict[str, float],
        duration_seconds: float,
    ) -> str:
        """Log a prediction (inference) event."""
        return self._write(
            "prediction",
            {
                "run_id": run_id,
                "prediction_id": prediction_id,
                "rows": rows,
                "date_range": {"start": date_range[0], "end": date_range[1]},
                "predictions": predictions,
                "summary": summary,
                "duration_seconds": round(duration_seconds, 4),
            },
        )

    # ------------------------------------------------------------------
    # Read helpers (used by dashboard)
    # ------------------------------------------------------------------

    def read_all(self) -> list[dict[str, Any]]:
        """Return all audit records as a list of dicts."""
        if not self.log_path.exists():
            return []
        records: list[dict[str, Any]] = []
        with open(self.log_path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        logger.warning("Skipping malformed audit line: %s", line[:80])
        return records

    def read_by_type(self, event_type: str) -> list[dict[str, Any]]:
        """Return all records of a given event_type."""
        return [r for r in self.read_all() if r.get("event_type") == event_type]
