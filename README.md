# ROAS Prediction with LightGBM

![CI](https://github.com/Yuvansh1/ROAS_Prediction_LightGBM/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)
![LightGBM](https://img.shields.io/badge/LightGBM-4.3%2B-green)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

A production-ready ML pipeline that predicts **Revenue** and derives **Return on Ad Spend (ROAS)** for luxury retail using LightGBM — replacing volatile direct-ROAS curve-fitting with a stable two-step framework.

---

## Architecture

```
Raw Data  ──►  Feature Engineering  ──►  LightGBM  ──►  Revenue Forecast
                                                              │
                                                              ▼
                                               ROAS = Predicted Revenue / Planned Spend
```

**Why predict Revenue instead of ROAS directly?**

ROAS is a ratio. Ratios are mathematically unstable when the denominator (spend) varies — a small spend produces extreme ROAS values that destabilise any gradient boosting model. By modelling revenue with spend as a covariate, we get:
- Numerically stable targets
- Better handling of diminishing-returns curvature (log-spend transform)
- Clean post-hoc ROAS derivation for any hypothetical spend scenario

---

## Project Structure

```
ROAS_Prediction_LightGBM/
├── src/
│   └── roas_prediction/
│       ├── data/
│       │   └── generator.py        # Synthetic luxury-retail data generator
│       ├── features/
│       │   └── engineering.py      # Lag, rolling-mean, calendar features
│       ├── models/
│       │   └── lgbm_model.py       # ROASPredictor (train / predict / save / load)
│       ├── evaluation/
│       │   └── metrics.py          # RMSE, MAE, R², MAPE
│       └── utils/
│           ├── audit.py            # Append-only JSONL audit trail
│           └── logging_config.py   # Centralised logging setup
├── dashboard/
│   └── app.py                      # Streamlit audit-trail dashboard
├── tests/
│   ├── test_data.py
│   ├── test_features.py
│   └── test_models.py
├── configs/
│   └── model_config.yaml           # All hyperparameters & paths
├── logs/                           # audit.jsonl written here at runtime
├── outputs/                        # Serialised model written here
├── .github/workflows/ci.yml        # GitHub Actions CI (test + lint)
├── main.py                         # Pipeline entry point
├── pyproject.toml
└── requirements.txt
```

---

## Quick Start

### 1. Clone & install

```bash
git clone https://github.com/yuvansh/ROAS_Prediction_LightGBM.git
cd ROAS_Prediction_LightGBM

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -e ".[dev]"
```

### 2. Train the model

```bash
python main.py
```

Optional — point at a custom config:

```bash
python main.py --config configs/model_config.yaml
```

Sample output:

```
=========================================================
         ROAS FORECAST RESULTS (HOLDOUT SAMPLE)
=========================================================
       date  marketing_spend  actual_roas  predicted_roas
 2025-10-02      17243.21        6.1234          6.0891
 ...
=========================================================
  RMSE  :      $2,341.87
  MAE   :      $1,876.12
  R²    :          0.9421
  MAPE  :          2.31%
=========================================================

Audit log → .../logs/audit.jsonl
Model    → .../outputs/lgbm_roas_model.txt
```

### 3. Launch the dashboard

```bash
streamlit run dashboard/app.py
```

Opens at `http://localhost:8501` with four tabs:

| Tab | Contents |
|-----|----------|
| **Overview** | KPI cards, event breakdown donut, latest run metadata |
| **Training Runs** | RMSE / R² over runs, feature importance bar chart, runs table |
| **ROAS Forecast** | Actual vs predicted ROAS & revenue time series, scatter, distribution |
| **Audit Log** | Filterable event table, timeline, raw JSON expander |

### 4. Run tests

```bash
pytest                       # all tests
pytest --cov=src -q          # with coverage summary
```

---

## Configuration

All knobs live in `configs/model_config.yaml`:

```yaml
data:
  start_date: "2024-01-01"
  end_date:   "2025-12-31"
  seed: 42

model:
  learning_rate: 0.05
  num_leaves: 31
  ...

training:
  test_days: 90              # holdout window
  num_boost_round: 500
  early_stopping_rounds: 20
  save_model: true

logging:
  audit_log_path: logs/audit.jsonl
  app_log_path:   logs/app.log
```

---

## Features Engineered

| Feature | Description |
|---------|-------------|
| `marketing_spend` | Raw daily ad spend ($) |
| `is_holiday` | 1 in November / December (retail peak) |
| `day_of_week` | 0 = Monday … 6 = Sunday |
| `month` | 1 – 12 |
| `revenue_lag_1` | Revenue 1 day ago |
| `revenue_lag_7` | Revenue 7 days ago |
| `revenue_roll_mean_7` | 7-day rolling average (lag-anchored, no leakage) |

---

## Audit Trail

Every pipeline step writes a structured JSON-Line record to `logs/audit.jsonl`:

```jsonc
// data_generation
{"event_id":"...","timestamp":"...","event_type":"data_generation",
 "payload":{"rows_generated":731,"seed":42,...}}

// feature_engineering
{"event_type":"feature_engineering",
 "payload":{"input_rows":731,"output_rows":724,"rows_dropped":7,...}}

// training_run
{"event_type":"training_run",
 "payload":{"run_id":"...","metrics":{"rmse":2341.87,"r2":0.9421},...}}

// prediction
{"event_type":"prediction",
 "payload":{"rows":90,"predictions":[...],"summary":{...}}}
```

---

## License

MIT — see [LICENSE](LICENSE).
