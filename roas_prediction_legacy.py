import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_error

def generate_mock_data():
    """Generates synthetic luxury retail data spanning 2 years."""
    print("Generating synthetic luxury retail data...")
    np.random.seed(42)
    dates = pd.date_range(start="2024-01-01", end="2025-12-31", freq="D")

    # Features
    marketing_spend = np.random.uniform(5000, 25000, len(dates))
    baseline = np.linspace(30000, 45000, len(dates))
    diminishing_spend_impact = np.log(marketing_spend) * 12000
    noise = np.random.normal(0, 2000, len(dates))

    df = pd.DataFrame({
        'date': dates,
        'revenue': baseline + diminishing_spend_impact + noise,
        'marketing_spend': marketing_spend
    })
    
    # Add high-season holiday indicator (November/December retail peak)
    df['is_holiday'] = 0
    df.loc[df['date'].dt.month.isin([11, 12]), 'is_holiday'] = 1
    return df

def engineer_features(df):
    """Transforms time-series data into supervised machine learning features."""
    df = df.copy()
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month

    # Time-series Lag and Rolling features
    df['revenue_lag_1'] = df['revenue'].shift(1)
    df['revenue_lag_7'] = df['revenue'].shift(7)
    df['revenue_roll_mean_7'] = df['revenue'].shift(1).rolling(window=7).mean()
    
    return df.dropna().reset_index(drop=True)

def train_and_forecast():
    # 1. Prepare Data
    raw_df = generate_mock_data()
    processed_df = engineer_features(raw_df)

    # 2. Chronological Train/Test Split (Avoids Data Leakage)
    split_date = processed_df['date'].max() - pd.Timedelta(days=90)
    train_df = processed_df[processed_df['date'] <= split_date]
    test_df = processed_df[processed_df['date'] > split_date]

    features = ['marketing_spend', 'is_holiday', 'day_of_week', 'month', 
                'revenue_lag_1', 'revenue_lag_7', 'revenue_roll_mean_7']
    target = 'revenue'

    X_train, y_train = train_df[features], train_df[target]
    X_test, y_test = test_df[features], test_df[target]

    # 3. Format datasets for LightGBM
    train_data = lgb.Dataset(X_train, label=y_train)
    test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'verbose': -1
    }

    # 4. Train Model
    print("Training LightGBM model...")
    model = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        valid_sets=[test_data],
        callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)]
    )

    # 5. Predict and Post-Process ROAS
    test_df = test_df.copy()
    test_df['predicted_revenue'] = model.predict(X_test, num_iteration=model.best_iteration)
    
    # ROAS Calculation = Revenue / Spend
    test_df['actual_roas'] = test_df['revenue'] / test_df['marketing_spend']
    test_df['predicted_roas'] = test_df['predicted_revenue'] / test_df['marketing_spend']

    # 6. Evaluation
    rmse = np.sqrt(mean_squared_error(test_df['revenue'], test_df['predicted_revenue']))
    
    print("\n=======================================================")
    print("             Q4 FORECAST RESULTS (SAMPLE)              ")
    print("=======================================================")
    print(test_df[['date', 'marketing_spend', 'actual_roas', 'predicted_roas']].head(10).to_string(index=False))
    print("=======================================================")
    print(f"Out-of-Sample Revenue RMSE: ${rmse:,.2f}")
    print("=======================================================")

if __name__ == "__main__":
    train_and_forecast()
