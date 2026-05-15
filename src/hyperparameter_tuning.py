import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, make_scorer
import warnings
import time
import os
warnings.filterwarnings('ignore')

PROCESSED_DATA_DIR = '../data/processed'
RESULTS_DIR        = '../results'

os.makedirs(RESULTS_DIR, exist_ok=True)

train = pd.read_csv(f'{PROCESSED_DATA_DIR}/train_env.csv', parse_dates=['date'])
val   = pd.read_csv(f'{PROCESSED_DATA_DIR}/val_env.csv',   parse_dates=['date'])


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


train_lagged = create_lag_features(train)
val_lagged   = create_lag_features(val)

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

X_train = train_lagged[ALL_FEATURES].copy()
y_train = train_lagged['unit_sales'].copy()
X_val   = val_lagged[ALL_FEATURES].copy()
y_val   = val_lagged['unit_sales'].copy()

param_grid = {
    'n_estimators':     [100, 200, 300],
    'max_depth':        [6, 8, 10],
    'learning_rate':    [0.05, 0.1, 0.15],
    'subsample':        [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9]
}

total_combinations = 1
for values in param_grid.values():
    total_combinations *= len(values)

print(f"starting grid search: {total_combinations} combinations x 3 folds = {total_combinations * 3} fits")

tscv       = TimeSeriesSplit(n_splits=3)
base_model = xgb.XGBRegressor(random_state=42, n_jobs=-1, verbosity=0)
mae_scorer = make_scorer(mean_absolute_error, greater_is_better=False)

start_time = time.time()

grid_search = GridSearchCV(
    estimator=base_model,
    param_grid=param_grid,
    cv=tscv,
    scoring=mae_scorer,
    verbose=1,
    n_jobs=-1,
    return_train_score=True
)

grid_search.fit(X_train, y_train)
duration_minutes = (time.time() - start_time) / 60
print(f"grid search completed in {duration_minutes:.1f} minutes")

best_params = grid_search.best_params_
best_model  = grid_search.best_estimator_

y_val_pred = np.maximum(best_model.predict(X_val), 0)
rmse  = np.sqrt(mean_squared_error(y_val, y_val_pred))
mae   = mean_absolute_error(y_val, y_val_pred)
r2    = r2_score(y_val, y_val_pred)
wmape = (y_val - y_val_pred).abs().sum() / y_val.sum() * 100
mape  = np.mean(np.abs((y_val - y_val_pred) / y_val.replace(0, np.nan))) * 100

print(f"\nbest parameters from GridSearchCV (23 features):")
for param, value in best_params.items():
    print(f"  {param}: {value}")

print(f"\nvalidation set:")
print(f"  RMSE: {rmse:.2f}  MAE: {mae:.2f}  R2: {r2*100:.2f}%  WMAPE: {wmape:.2f}%  MAPE: {mape:.2f}%")

cv_results = pd.DataFrame(grid_search.cv_results_).sort_values(
    'mean_test_score', ascending=False)
cv_results.to_csv(f'{RESULTS_DIR}/xgboost_cv_results.csv', index=False)

print(f"\ntop 5 parameter combinations:")
for i in range(min(5, len(cv_results))):
    row = cv_results.iloc[i]
    print(f"   MAE {-row['mean_test_score']:.2f} | "
          f"n_est={row['param_n_estimators']} "
          f"depth={row['param_max_depth']} "
          f"lr={row['param_learning_rate']} "
          f"sub={row['param_subsample']} "
          f"col={row['param_colsample_bytree']}")

