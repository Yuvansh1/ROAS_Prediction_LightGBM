# Luxury Retail ROAS Forecasting with LightGBM

This repository provides a high-performance machine learning alternative to traditional curve-fitting time-series models (like Facebook Prophet). It reformulates the task into a supervised regression problem using **LightGBM**, specifically designed to handle marketing spend, diminishing returns profiles, and promotional holiday calendars natively.

## Project Architecture

Instead of forecasting ratios (ROAS) directly—which introduces extreme volatility and mathematical instability—the model follows a two-step framework:
1. **Target Modeling:** Forecasts total **Revenue** (`y`) using non-linear external covariates (such as logarithmic ad spend transformations and historical rolling windows).
2. **Post-Processing Pipeline:** Dynamically derives predicted **ROAS** by calculating the ratio of predicted revenue against planned marketing budgets (`Predicted Revenue / Planned Spend`).

## Core Features Engineered

* **Lags (`revenue_lag_1`, `revenue_lag_7`):** Provides sequence and auto-regressive context to the gradient booster.
* **Rolling Windows (`revenue_roll_mean_7`):** Captures moving baseline trends over a localized window.
* **Covariates:** Maps structural features like promotional peaks (`is_holiday`), day-of-week trends, and localized monthly demand shifts.

## Getting Started

### Prerequisites
Ensure you have Python installed, then install the required dependencies:
```bash
pip install lightgbm pandas numpy scikit-learn
