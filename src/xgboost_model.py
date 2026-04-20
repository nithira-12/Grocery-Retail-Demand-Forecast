import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
import os
warnings.filterwarnings('ignore')

PROCESSED_DATA_DIR = '../data/processed'
RESULTS_DIR        = '../results'
MODELS_DIR         = '../models'

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

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

TARGET            = 'unit_sales'
LAG_FEATURES      = ['sales_lag_7', 'sales_lag_14', 'sales_lag_28',
                      'sales_rolling_mean_7', 'sales_rolling_mean_28', 'sales_rolling_std_7']
TEMPORAL_FEATURES = ['dayofweek', 'month', 'quarter', 'day',
                      'is_weekend', 'is_month_start', 'is_month_end']
EXTERNAL_FEATURES = ['onpromotion', 'holiday_impact', 'family_encoded']
ALL_FEATURES      = LAG_FEATURES + TEMPORAL_FEATURES + EXTERNAL_FEATURES

# warn if any expected features are missing
missing = [f for f in ALL_FEATURES if f not in train_lagged.columns]
if missing:
    print(f"warning missing features: {missing}")
else:
    print("All features found")

X_train = train_lagged[ALL_FEATURES].copy()
y_train = train_lagged[TARGET].copy()
X_val   = val_lagged[ALL_FEATURES].copy()
y_val   = val_lagged[TARGET].copy()


model = xgb.XGBRegressor(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    verbosity=0
)

model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

y_val_pred = model.predict(X_val)

rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
mae  = mean_absolute_error(y_val, y_val_pred)
r2   = r2_score(y_val, y_val_pred)
mape = np.mean(np.abs((y_val - y_val_pred) / y_val)) * 100

print(f"\nXGBoost validation results:")
print(f"  RMSE: {rmse:.2f}  MAE: {mae:.2f}  R2: {r2*100:.2f}%  MAPE: {mape:.2f}%")

if r2 > 0.90:
    print(f"  R2 > 90% — excellent performance")
elif r2 > 0.85:
    print(f"  R2 > 85% — good performance")
elif r2 > 0.80:
    print(f"  R2 > 80% — consider hyperparameter tuning")
else:
    print(f"  warning: low R2 — check for issues")

# feature importance by gain
importance_dict = model.get_booster().get_score(importance_type='gain')
importance_df   = pd.DataFrame({
    'feature':    list(importance_dict.keys()),
    'importance': list(importance_dict.values())
}).sort_values('importance', ascending=False)

print(f"\ntop 10 features by gain:")
for _, row in importance_df.head(10).iterrows():
    print(f"  {str(row['feature']):<30} {row['importance']:,.1f}")

# holiday_impact rank — useful to verify it was captured
if 'holiday_impact' in importance_df['feature'].values:
    holiday_rank = importance_df[importance_df['feature'] == 'holiday_impact'].index[0] + 1
    print(f"\nholiday_impact rank: {holiday_rank} of {len(importance_df)}")
else:
    print(f"\nwarning:not found ")

# product and store breakdown
val_results               = val_lagged.copy()
val_results['predictions'] = y_val_pred
val_results['actual']      = y_val.values
val_results['error']       = val_results['actual'] - val_results['predictions']
val_results['abs_error']   = val_results['error'].abs()
val_results['pct_error']   = (val_results['abs_error'] / val_results['actual']) * 100

product_perf = val_results.groupby('item_nbr').agg(
    avg_actual=('actual', 'mean'),
    avg_predicted=('predictions', 'mean'),
    mae=('abs_error', 'mean'),
    mape=('pct_error', 'mean')
).round(2).sort_values('mae', ascending=False)

store_perf = val_results.groupby('store_nbr').agg(
    avg_actual=('actual', 'mean'),
    avg_predicted=('predictions', 'mean'),
    mae=('abs_error', 'mean'),
    mape=('pct_error', 'mean')
).round(2)
store_perf.index = store_perf.index.map(STORE_NAMES)

print(f"\nperformance by product:")
print(product_perf.to_string())

print(f"\nperformance by store:")
print(store_perf.to_string())

# save results
pd.DataFrame({
    'Model': ['XGBoost'],
    'RMSE':  [rmse],
    'MAE':   [mae],
    'R2':    [r2],
    'MAPE':  [mape]
}).to_csv(f'{RESULTS_DIR}/xgboost_metrics.csv', index=False)

importance_df.to_csv(f'{RESULTS_DIR}/xgboost_feature_importance.csv', index=False)

val_results[['date', 'store_nbr', 'item_nbr', 'actual', 'predictions', 'error']].to_csv(
    f'{RESULTS_DIR}/xgboost_predictions.csv', index=False
)

with open(f'{MODELS_DIR}/xgboost_model.pkl', 'wb') as f:
    pickle.dump(model, f)