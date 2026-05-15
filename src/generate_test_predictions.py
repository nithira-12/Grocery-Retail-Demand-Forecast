import pandas as pd
import numpy as np
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

os.makedirs('../results', exist_ok=True)

with open('../models/xgboost_model_v2.pkl', 'rb') as f:
    model = pickle.load(f)

with open('../models/feature_list.pkl', 'rb') as f:
    ALL_FEATURES = pickle.load(f)

test = pd.read_csv('../data/processed/test_env.csv', parse_dates=['date'])

BLEND_PRODUCTS = [564287, 903285, 584028, 1047679]
BLEND_WEIGHT   = 0.80

PRODUCT_NAMES = {
    265559:  "Grocery Staples B",
    364606:  "Grocery Staples A",
    502331:  "Bread A",
    564287:  "Baked Goods B",
    584028:  "Meat Product A",
    903285:  "Poultry A",
    1047679: "Soft Drinks A",
    1427659: "Dairy Product",
    1473474: "Vegetables B",
    1503844: "Vegetables A",
    1695835: "Fruit A Bananas"
}


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


test_lagged = create_lag_features(test)

available = [f for f in ALL_FEATURES if f in test_lagged.columns]
if len(available) != len(ALL_FEATURES):
    missing = set(ALL_FEATURES) - set(available)
    print(f"warning: missing features in test data: {missing}")

X_test = test_lagged[available]
y_test = test_lagged['unit_sales']

test_pred_raw = np.maximum(model.predict(X_test), 0)
test_pred     = apply_blend(test_pred_raw, test_lagged)

actual    = pd.Series(y_test.values)
predicted = pd.Series(test_pred)

rmse  = np.sqrt(mean_squared_error(actual, predicted))
mae   = mean_absolute_error(actual, predicted)
r2    = r2_score(actual, predicted)
mape  = np.mean(np.abs((actual - predicted) / actual.replace(0, np.nan))) * 100
wmape = (actual - predicted).abs().sum() / actual.sum() * 100

print(f"2017 test results :")
print(f"  RMSE:  {rmse:.2f}")
print(f"  MAE:   {mae:.2f}")
print(f"  R2:    {r2*100:.2f}%")
print(f"  WMAPE: {wmape:.2f}%")
print(f"  MAPE:  {mape:.2f}%")

results_df                = test_lagged[['date', 'store_nbr', 'item_nbr']].copy()
results_df['actual']      = y_test.values
results_df['predictions'] = test_pred
results_df['error']       = results_df['actual'] - results_df['predictions']
results_df.to_csv('../results/xgboost_test_predictions.csv', index=False)

results_df['abs_error'] = results_df['error'].abs()

product_wmape = results_df.groupby('item_nbr').apply(
    lambda x: x['abs_error'].sum() / x['actual'].sum() * 100
).reset_index()
product_wmape.columns = ['item_nbr', 'wmape']

product_mae           = results_df.groupby('item_nbr')['abs_error'].mean().reset_index()
product_mae.columns   = ['item_nbr', 'mae']

product_perf          = product_wmape.merge(product_mae, on='item_nbr')
product_perf['name']  = product_perf['item_nbr'].map(PRODUCT_NAMES)
product_perf          = product_perf.sort_values('wmape')

print(f"\nperformance by product (2017 test):")
for _, row in product_perf.iterrows():
    blend_flag = " [blend]" if row['item_nbr'] in BLEND_PRODUCTS else ""
    print(f"  {row['name']:<25} MAE: {row['mae']:>6.2f}  WMAPE: {row['wmape']:>6.2f}%{blend_flag}")