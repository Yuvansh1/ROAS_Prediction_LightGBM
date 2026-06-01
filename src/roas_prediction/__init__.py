"""
ROAS Prediction — LightGBM-based Revenue & Return-on-Ad-Spend Forecasting.

Two-step framework:
  1. Predict total revenue using LightGBM with marketing covariates.
  2. Derive ROAS post-hoc: predicted_revenue / planned_spend.
"""

__version__ = "1.0.0"
__author__ = "Yuvansh"
