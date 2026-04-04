
# BASELINE MODEL: MOVING AVERAGE


import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
import os
warnings.filterwarnings('ignore')


print("moving average baseeline")



# CONFIGURATION


PROCESSED_DATA_DIR = '../data/processed'
RESULTS_DIR = '../results'
os.makedirs(RESULTS_DIR, exist_ok=True)

SELECTED_STORES = [44, 51]
STORE_NAMES = {44: "Store Colombo", 51: "Store Gampaha"}

print("\n Configuration loaded")

# LOAD DATA

print("STEP 1: LOADING DATA")


train = pd.read_csv(f'{PROCESSED_DATA_DIR}/train_processed.csv', parse_dates=['date'])
val = pd.read_csv(f'{PROCESSED_DATA_DIR}/val_processed.csv', parse_dates=['date'])

print(f"\n Data loaded:")
print(f"  train: {train.shape[0]:,} rows")
print(f"  validation: {val.shape[0]:,} rows")


# moving average 



print("STEP 2: generatin moving avg forecasts")


def moving_average_forecast(train_df, val_df, window_sizes=[7, 28]):
    
    results = {}
    
    for window in window_sizes:
        print(f"\n Computing {window}-day moving average")
        
        # Combine train + val for continuous history
        combined = pd.concat([train_df, val_df], ignore_index=True)
        combined = combined.sort_values(['store_nbr', 'item_nbr', 'date'])
        
        # Calculate rolling mean
        # shift(1) ensures we don't use today's actual value (no leakage!)
        combined['ma_prediction'] = combined.groupby(['store_nbr', 'item_nbr'])['unit_sales'].transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=window).mean()
        )
        
        # Extract only validation period predictions
        val_with_pred = combined[combined['date'] >= val_df['date'].min()].copy()
        val_with_pred = val_with_pred.dropna(subset=['ma_prediction'])
        
        # Calculate metrics
        y_true = val_with_pred['unit_sales']
        y_pred = val_with_pred['ma_prediction']
        
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        results[f'MA_{window}'] = {
            'window': window,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'mape': mape,
            'predictions': val_with_pred
        }
        
        print(f"   {window}-day MA completed:")
        print(f"    RMSE: {rmse:.2f}")
        print(f"    MAE: {mae:.2f}")
        print(f"    R squared: {r2:.4f}")
        print(f"    MAPE: {mape:.2f}%")
    
    return results

# Generate forecasts
results = moving_average_forecast(train, val, window_sizes=[7, 28])

# DETAILED ANALYSIS

print("STEP 3: comparing variants")


comparison_data = []
for model_name, model_data in results.items():
    comparison_data.append({
        'Model': model_name,
        'Window': model_data['window'],
        'RMSE': model_data['rmse'],
        'MAE': model_data['mae'],
        'R2': model_data['r2'],
        'MAPE': model_data['mape']
    })

comparison_df = pd.DataFrame(comparison_data)
print("\n" + comparison_df.to_string(index=False))

# Find best model
best_model = comparison_df.loc[comparison_df['RMSE'].idxmin()]
print(f"\n best performing variant: {best_model['Model']}")
print(f"  RMSE: {best_model['RMSE']:.2f}")
print(f"  R squared: {best_model['R2']:.4f}")


# ANALYZE BY PRODUCT & STORE


print("STEP 4: PERFORMANCE BY PRODUCT & STORE (7 day )")


ma7_data = results['MA_7']['predictions']
ma7_data['error'] = ma7_data['unit_sales'] - ma7_data['ma_prediction']
ma7_data['abs_error'] = ma7_data['error'].abs()
ma7_data['pct_error'] = (ma7_data['abs_error'] / ma7_data['unit_sales']) * 100

# By product
print("\n→ Performance by Product:")
product_perf = ma7_data.groupby('item_nbr').agg({
    'unit_sales': 'mean',
    'ma_prediction': 'mean',
    'abs_error': 'mean',
    'pct_error': 'mean'
}).round(2)
product_perf.columns = ['Avg Actual', 'Avg Predicted', 'MAE', 'MAPE (%)']
product_perf = product_perf.sort_values('MAE', ascending=False)
print(product_perf)

# By store
print("\n→ Performance by Store:")
store_perf = ma7_data.groupby('store_nbr').agg({
    'unit_sales': 'mean',
    'ma_prediction': 'mean',
    'abs_error': 'mean',
    'pct_error': 'mean'
}).round(2)
store_perf.columns = ['Avg Actual', 'Avg Predicted', 'MAE', 'MAPE (%)']
store_perf.index = store_perf.index.map(STORE_NAMES)
print(store_perf)

# SAVE RESULTS


print("STEP 5: SAVING RESULTS")


# Save comparison metrics
metrics_file = f'{RESULTS_DIR}/moving_average_metrics.csv'
comparison_df.to_csv(metrics_file, index=False)
print(f"\n Metrics saved to: {metrics_file}")

# Save 7-day predictions
pred_file = f'{RESULTS_DIR}/moving_average_7day_predictions.csv'
ma7_data[['date', 'store_nbr', 'item_nbr', 'unit_sales', 'ma_prediction', 'error']].to_csv(
    pred_file, index=False
)
print(f"7 day predictions saved to: {pred_file}")

# Save 28-day predictions
ma28_data = results['MA_28']['predictions']
pred_file_28 = f'{RESULTS_DIR}/moving_average_28day_predictions.csv'
ma28_data[['date', 'store_nbr', 'item_nbr', 'unit_sales', 'ma_prediction']].to_csv(
    pred_file_28, index=False
)
print(f" 28 day predictions saved to: {pred_file_28}")


print("MOVING AVERAGE BASELINE done")


print(f"\n Summary:")
print(f"  7 day MA:  RMSE={results['MA_7']['rmse']:.2f}, MAE={results['MA_7']['mae']:.2f}, R sq={results['MA_7']['r2']:.4f}")
print(f"  28 day MA: RMSE={results['MA_28']['rmse']:.2f}, MAE={results['MA_28']['mae']:.2f}, R sq={results['MA_28']['r2']:.4f}")
print(f"  Results saved to: {RESULTS_DIR}/")
