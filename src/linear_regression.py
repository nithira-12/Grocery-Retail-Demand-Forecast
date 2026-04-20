import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import warnings
import os
warnings.filterwarnings('ignore')

PROCESSED_DATA_DIR = '../data/processed'
RESULTS_DIR        = '../results'
os.makedirs(RESULTS_DIR, exist_ok=True)

SELECTED_STORES = [44, 51]
STORE_NAMES     = {44: "Store Colombo", 51: "Store Gampaha"}

train = pd.read_csv(f'{PROCESSED_DATA_DIR}/train_processed.csv', parse_dates=['date'])
val   = pd.read_csv(f'{PROCESSED_DATA_DIR}/val_processed.csv',   parse_dates=['date'])
test  = pd.read_csv(f'{PROCESSED_DATA_DIR}/test_processed.csv',  parse_dates=['date'])


def create_lag_features(df, lag_days=[7, 14, 28]):
    df_lagged = df.copy()
    for lag in lag_days:
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

    return df_lagged


train_lagged = create_lag_features(train).dropna()
val_lagged   = create_lag_features(val).dropna()
test_lagged  = create_lag_features(test).dropna()

LAG_FEATURES      = ['sales_lag_7', 'sales_lag_14', 'sales_lag_28',
                      'sales_rolling_mean_7', 'sales_rolling_mean_28', 'sales_rolling_std_7']
TEMPORAL_FEATURES = ['dayofweek', 'month', 'quarter', 'day',
                      'is_weekend', 'is_month_start', 'is_month_end']
EXTERNAL_FEATURES = ['onpromotion', 'is_holiday', 'family_encoded']
ALL_FEATURES      = LAG_FEATURES + TEMPORAL_FEATURES + EXTERNAL_FEATURES
TARGET            = 'unit_sales'

# warn if any expected features are missing
missing_features = [f for f in ALL_FEATURES if f not in train_lagged.columns]
if missing_features:
    print(f"warning: missing features: {missing_features}")

X_train = train_lagged[ALL_FEATURES].copy()
y_train = train_lagged[TARGET].copy()
X_val   = val_lagged[ALL_FEATURES].copy()
y_val   = val_lagged[TARGET].copy()

# scaler fit on training data only — prevents data leakage into validation
scaler         = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled   = scaler.transform(X_val)

model = LinearRegression()
model.fit(X_train_scaled, y_train)

y_val_pred = model.predict(X_val_scaled)

rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
mae  = mean_absolute_error(y_val, y_val_pred)
r2   = r2_score(y_val, y_val_pred)
mape = np.mean(np.abs((y_val - y_val_pred) / y_val)) * 100

# validation results
print(f"linear regression validation results:")
print(f"  RMSE: {rmse:.2f}  MAE: {mae:.2f}  R2: {r2*100:.2f}%  MAPE: {mape:.2f}%")

# warn on suspicious r2 values
if r2 < 0:
    print(f"warning: negative R2 — model performs worse than predicting the mean")
elif r2 > 0.95:
    print(f"warning: very high R2 — check for data leakage")

# feature coefficients — useful for understanding what the model learned
feature_importance = pd.DataFrame({
    'feature':     ALL_FEATURES,
    'coefficient': model.coef_
})
feature_importance['abs_coefficient'] = feature_importance['coefficient'].abs()
feature_importance = feature_importance.sort_values('abs_coefficient', ascending=False)

print(f"\ntop 10 features by coefficient (intercept: {model.intercept_:.2f}):")
for _, row in feature_importance.head(10).iterrows():
    print(f"  {row['feature']:<25} {row['coefficient']:>8.3f}")

# performance by product and store
val_results             = val_lagged.copy()
val_results['predictions'] = y_val_pred
val_results['actual']      = y_val.values
val_results['error']       = val_results['actual'] - val_results['predictions']
val_results['abs_error']   = val_results['error'].abs()
val_results['pct_error']   = (val_results['abs_error'] / val_results['actual']) * 100

product_performance = val_results.groupby('item_nbr').agg(
    avg_actual=('actual', 'mean'),
    avg_predicted=('predictions', 'mean'),
    mae=('abs_error', 'mean'),
    mape=('pct_error', 'mean')
).round(2).sort_values('mae', ascending=False)

store_performance = val_results.groupby('store_nbr').agg(
    avg_actual=('actual', 'mean'),
    avg_predicted=('predictions', 'mean'),
    mae=('abs_error', 'mean'),
    mape=('pct_error', 'mean')
).round(2)
store_performance.index = store_performance.index.map(STORE_NAMES)

print(f"\nperformance by product:")
print(product_performance.to_string())

print(f"\nperformance by store:")
print(store_performance.to_string())

# save results
metrics_df = pd.DataFrame({
    'Model': ['Linear Regression'],
    'RMSE':  [rmse],
    'MAE':   [mae],
    'R2':    [r2],
    'MAPE':  [mape]
})
metrics_df.to_csv(f'{RESULTS_DIR}/linear_regression_metrics.csv', index=False)

feature_importance.to_csv(f'{RESULTS_DIR}/linear_regression_feature_importance.csv', index=False)

val_results[['date', 'store_nbr', 'item_nbr', 'actual', 'predictions', 'error']].to_csv(
    f'{RESULTS_DIR}/linear_regression_predictions.csv', index=False
)