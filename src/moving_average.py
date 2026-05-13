import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

os.makedirs('../results', exist_ok=True)

val = pd.read_csv('../data/processed/val_env.csv', parse_dates=['date'])


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


val_lagged = create_lag_features(val)

actual      = pd.Series(val_lagged['unit_sales'].values)
predictions = pd.Series(
    val_lagged['sales_rolling_mean_7'].fillna(actual.mean()).values
)

results                = val_lagged[['date', 'store_nbr', 'item_nbr']].copy()
results['actual']      = actual.values
results['predictions'] = predictions.values
results['error']       = results['actual'] - results['predictions']

rmse  = np.sqrt(mean_squared_error(actual, predictions))
mae   = mean_absolute_error(actual, predictions)
r2    = r2_score(actual, predictions)
mape  = np.mean(np.abs((actual - predictions) / actual.replace(0, np.nan))) * 100
wmape = (actual - predictions).abs().sum() / actual.sum() * 100

print(f"\nMoving Average (7-day) validation results:")
print(f"  RMSE: {rmse:.2f}  MAE: {mae:.2f}  R2: {r2*100:.2f}%  WMAPE: {wmape:.2f}%  MAPE: {mape:.2f}%")

results.to_csv('../results/moving_average_7day_predictions.csv', index=False)

pd.DataFrame([{
    'Model': 'Moving Average', 'RMSE': round(rmse, 2), 'MAE': round(mae, 2),
    'R2': round(r2, 4), 'R2_pct': round(r2 * 100, 2),
    'MAPE': round(mape, 2), 'WMAPE': round(wmape, 2)
}]).to_csv('../results/moving_average_metrics.csv', index=False)