"""
Simple Model Metrics Display
Run from src/: python display_model_metrics.py
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def calculate_metrics(actual, predicted):
    mask = ~(np.isnan(actual) | np.isnan(predicted))
    actual_clean = actual[mask]
    predicted_clean = predicted[mask]
    
    rmse = np.sqrt(mean_squared_error(actual_clean, predicted_clean))
    mae = mean_absolute_error(actual_clean, predicted_clean)
    r2 = r2_score(actual_clean, predicted_clean)
    mape_mask = actual_clean != 0
    mape = np.mean(np.abs((actual_clean[mape_mask] - predicted_clean[mape_mask]) / actual_clean[mape_mask])) * 100
    
    return rmse, mae, r2, mape

print("\nModel Performance Metrics - Validation Set (2016)")
print("="*60)

all_metrics = {}

# Moving Average
try:
    df = pd.read_csv('../results/moving_average_7day_predictions.csv')
    if 'ma_prediction' in df.columns:
        df['predictions'] = df['ma_prediction']
    if 'unit_sales' in df.columns:
        df['actual'] = df['unit_sales']
    rmse, mae, r2, mape = calculate_metrics(df['actual'].values, df['predictions'].values)
    all_metrics['Moving Average'] = (rmse, mae, r2, mape)
    print("Moving Average loaded")
except:
    print("Moving Average - file not found")

# Linear Regression
try:
    df = pd.read_csv('../results/linear_regression_predictions.csv')
    rmse, mae, r2, mape = calculate_metrics(df['actual'].values, df['predictions'].values)
    all_metrics['Linear Regression'] = (rmse, mae, r2, mape)
    print("Linear Regression loaded")
except:
    print("Linear Regression - file not found")

# Prophet
# Prophet
try:
    df = pd.read_csv('D:/UNIVERSITY/UNI YEAR 3/Final Year Computing Project/Grocery-Retail-Demand-Forecast/results/prophet_predictions.csv')
    if 'predicted' in df.columns:  # CHANGE THIS
        df['predictions'] = df['predicted']  # ADD THIS LINE
    if 'actuals' in df.columns:
        df['actual'] = df['actuals']
    rmse, mae, r2, mape = calculate_metrics(df['actual'].values, df['predictions'].values)
    all_metrics['Prophet'] = (rmse, mae, r2, mape)
    print("Prophet loaded")
except Exception as e:
    print(f"Prophet error: {e}")

# XGBoost
try:
    df = pd.read_csv('../results/xgboost_predictions.csv')
    rmse, mae, r2, mape = calculate_metrics(df['actual'].values, df['predictions'].values)
    all_metrics['XGBoost'] = (rmse, mae, r2, mape)
    print("XGBoost loaded")
except:
    print("XGBoost - file not found")

print("\n")
print(f"{'Model':<25} {'RMSE':>8} {'MAE':>8} {'R2':>8} {'MAPE':>8}")
print("\n")


for model, (rmse, mae, r2, mape) in all_metrics.items():
    print(f"{model:<25} {rmse:>8.2f} {mae:>8.2f} {r2:>8.4f} {mape:>8.2f}")

