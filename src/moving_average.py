import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
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


def moving_average_forecast(train_df, val_df, window_sizes=[7, 28]):
    results = {}

    for window in window_sizes:
        # combine train and val for continuous history, shift(1) prevents leakage
        combined = pd.concat([train_df, val_df], ignore_index=True)
        combined = combined.sort_values(['store_nbr', 'item_nbr', 'date'])

        combined['ma_prediction'] = combined.groupby(
            ['store_nbr', 'item_nbr'])['unit_sales'].transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=window).mean()
        )

        val_with_pred = combined[combined['date'] >= val_df['date'].min()].copy()
        val_with_pred = val_with_pred.dropna(subset=['ma_prediction'])

        y_true = val_with_pred['unit_sales']
        y_pred = val_with_pred['ma_prediction']

        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae  = mean_absolute_error(y_true, y_pred)
        r2   = r2_score(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

        results[f'MA_{window}'] = {
            'window':      window,
            'rmse':        rmse,
            'mae':         mae,
            'r2':          r2,
            'mape':        mape,
            'predictions': val_with_pred
        }

    return results


results = moving_average_forecast(train, val, window_sizes=[7, 28])

# comparison of both window sizes
comparison_data = []
for model_name, model_data in results.items():
    comparison_data.append({
        'Model': model_name,
        'Window': model_data['window'],
        'RMSE':  model_data['rmse'],
        'MAE':   model_data['mae'],
        'R2':    model_data['r2'],
        'MAPE':  model_data['mape']
    })

comparison_df = pd.DataFrame(comparison_data)

print(f"moving average results:")
print(comparison_df.to_string(index=False))

best_model = comparison_df.loc[comparison_df['RMSE'].idxmin()]
print(f"\nbest variant: {best_model['Model']}  RMSE: {best_model['RMSE']:.2f}  R2: {best_model['R2']:.4f}")

# product and store breakdown for 7-day MA
ma7_data              = results['MA_7']['predictions']
ma7_data['error']     = ma7_data['unit_sales'] - ma7_data['ma_prediction']
ma7_data['abs_error'] = ma7_data['error'].abs()
ma7_data['pct_error'] = (ma7_data['abs_error'] / ma7_data['unit_sales']) * 100

product_perf = ma7_data.groupby('item_nbr').agg(
    avg_actual=('unit_sales', 'mean'),
    avg_predicted=('ma_prediction', 'mean'),
    mae=('abs_error', 'mean'),
    mape=('pct_error', 'mean')
).round(2).sort_values('mae', ascending=False)

store_perf = ma7_data.groupby('store_nbr').agg(
    avg_actual=('unit_sales', 'mean'),
    avg_predicted=('ma_prediction', 'mean'),
    mae=('abs_error', 'mean'),
    mape=('pct_error', 'mean')
).round(2)
store_perf.index = store_perf.index.map(STORE_NAMES)

print(f"\nperformance by product (7-day MA):")
print(product_perf.to_string())

print(f"\nperformance by store (7-day MA):")
print(store_perf.to_string())

# save results
comparison_df.to_csv(f'{RESULTS_DIR}/moving_average_metrics.csv', index=False)

ma7_data[['date', 'store_nbr', 'item_nbr', 'unit_sales', 'ma_prediction', 'error']].to_csv(
    f'{RESULTS_DIR}/moving_average_7day_predictions.csv', index=False
)

ma28_data = results['MA_28']['predictions']
ma28_data[['date', 'store_nbr', 'item_nbr', 'unit_sales', 'ma_prediction']].to_csv(
    f'{RESULTS_DIR}/moving_average_28day_predictions.csv', index=False
)