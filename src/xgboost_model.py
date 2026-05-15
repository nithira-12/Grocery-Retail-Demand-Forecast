import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

os.makedirs('../results', exist_ok=True)
os.makedirs('../models', exist_ok=True)

train = pd.read_csv('../data/processed/train_env.csv', parse_dates=['date'])
val   = pd.read_csv('../data/processed/val_env.csv',   parse_dates=['date'])

BLEND_PRODUCTS = [564287, 903285, 584028, 1047679]
BLEND_WEIGHT   = 0.80


def create_lag_features(df):
    df_lagged = df.copy()
    for lag in [1, 2, 7, 14, 28]:
        df_lagged[f'sales_lag_{lag}'] = df_lagged.groupby(
            ['store_nbr', 'item_nbr'])['unit_sales'].shift(lag)
    df_lagged['sales_rolling_mean_7'] = df_lagged.groupby(
        ['store_nbr', 'item_nbr'])['unit_sales'].transform(
        lambda x: x.shift(1).rolling(window=7, min_periods=7).mean())
    df_lagged['sales_rolling_mean_28'] = df_lagged.groupby(
        ['store_nbr', 'item_nbr'])['unit_sales'].transform(
        lambda x: x.shift(1).rolling(window=28, min_periods=28).mean())
    df_lagged['sales_rolling_std_7'] = df_lagged.groupby(
        ['store_nbr', 'item_nbr'])['unit_sales'].transform(
        lambda x: x.shift(1).rolling(window=7, min_periods=7).std())
    return df_lagged.dropna()


def apply_blend(predictions, df):
    blended = predictions.copy()
    for product in BLEND_PRODUCTS:
        mask    = df['item_nbr'].values == product
        rolling = df.loc[df['item_nbr'] == product, 'sales_rolling_mean_7'].values
        if mask.sum() == 0:
            continue
        blended[mask] = (
            BLEND_WEIGHT * predictions[mask] +
            (1 - BLEND_WEIGHT) * rolling
        )
    return blended


train_lagged = create_lag_features(train)
val_lagged   = create_lag_features(val)

TARGET = 'unit_sales'

ALL_FEATURES = [
    'sales_lag_1', 'sales_lag_2',
    'sales_lag_7', 'sales_lag_14', 'sales_lag_28',
    'sales_rolling_mean_7', 'sales_rolling_mean_28', 'sales_rolling_std_7',
    'dayofweek', 'month', 'quarter', 'day',
    'is_weekend', 'is_month_start', 'is_month_end',
    'onpromotion', 'holiday_impact', 'family_encoded',
    'temperature_c', 'rainfall_mm', 'windspeed_kmh',
    'cpi_normalized', 'fuel_normalized'
]

ALL_FEATURES = [f for f in ALL_FEATURES if f in train_lagged.columns]
print(f"features used: {len(ALL_FEATURES)}")

X_train = train_lagged[ALL_FEATURES]
y_train = train_lagged[TARGET]
X_val   = val_lagged[ALL_FEATURES]
y_val   = val_lagged[TARGET]

print("trainingg")

model = xgb.XGBRegressor(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.7,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    verbosity=0
)

model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

y_val_pred_raw = np.maximum(model.predict(X_val), 0)
y_val_pred     = apply_blend(y_val_pred_raw, val_lagged)

actual    = pd.Series(y_val.values)
predicted = pd.Series(y_val_pred)

rmse  = np.sqrt(mean_squared_error(actual, predicted))
mae   = mean_absolute_error(actual, predicted)
r2    = r2_score(actual, predicted)
mape  = np.mean(np.abs((actual - predicted) / actual.replace(0, np.nan))) * 100
wmape = (actual - predicted).abs().sum() / actual.sum() * 100

print(f"\nXGBoost validation results:")
print(f"  RMSE: {rmse:.2f}  MAE: {mae:.2f}  R2: {r2*100:.2f}%  WMAPE: {wmape:.2f}%  MAPE: {mape:.2f}%")

results                = val_lagged[['date', 'store_nbr', 'item_nbr']].copy()
results['actual']      = y_val.values
results['predictions'] = y_val_pred
results['error']       = results['actual'] - results['predictions']
results.to_csv('../results/xgboost_predictions.csv', index=False)

pd.DataFrame([{
    'Model': 'XGBoost', 'RMSE': round(rmse, 2), 'MAE': round(mae, 2),
    'R2': round(r2, 4), 'R2_pct': round(r2 * 100, 2),
    'MAPE': round(mape, 2), 'WMAPE': round(wmape, 2)
}]).to_csv('../results/xgboost_metrics.csv', index=False)

with open('../models/xgboost_model_v2.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('../models/feature_list.pkl', 'wb') as f:
    pickle.dump(ALL_FEATURES, f)