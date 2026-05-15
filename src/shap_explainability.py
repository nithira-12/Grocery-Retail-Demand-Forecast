import pandas as pd
import numpy as np
import pickle
import os
import shap
import time
import warnings
warnings.filterwarnings('ignore')

os.makedirs('../results/shap', exist_ok=True)
os.makedirs('../results/shap/shap_product_importance', exist_ok=True)

with open('../models/xgboost_model_v2.pkl', 'rb') as f:
    model = pickle.load(f)

with open('../models/feature_list.pkl', 'rb') as f:
    ALL_FEATURES = pickle.load(f)

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
X_val      = val_lagged[ALL_FEATURES]
y_val      = val_lagged['unit_sales']
y_val_pred = model.predict(X_val)

print(f"computing SHAP values for {len(X_val):,} predictions...")
start       = time.time()
explainer   = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_val)
base_value  = explainer.expected_value
elapsed     = time.time() - start

print(f"  done in {elapsed:.1f} seconds")
print(f"  base value: {base_value:.4f}")

shap_df = pd.DataFrame(shap_values, columns=[f'shap_{f}' for f in ALL_FEATURES])
shap_df['date']       = val_lagged['date'].values
shap_df['store_nbr']  = val_lagged['store_nbr'].values
shap_df['item_nbr']   = val_lagged['item_nbr'].values
shap_df['actual']     = y_val.values
shap_df['prediction'] = y_val_pred
shap_df['base_value'] = base_value
shap_df.to_csv('../results/shap/complete_shap_values.csv', index=False)

mean_abs_shap = np.abs(shap_values).mean(axis=0)
importance_df = pd.DataFrame({
    'feature':       ALL_FEATURES,
    'mean_abs_shap': mean_abs_shap
}).sort_values('mean_abs_shap', ascending=False).reset_index(drop=True)
importance_df['rank']       = range(1, len(importance_df) + 1)
importance_df['normalized'] = importance_df['mean_abs_shap'] / importance_df['mean_abs_shap'].max() * 100
importance_df.to_csv('../results/shap/shap_feature_importance.csv', index=False)

print(f"\ntop 10 features by SHAP importance:")
print(f"  {'rank':<6} {'feature':<30} {'mean |SHAP|':>12} {'normalized':>10}")
print(f"  {'-'*60}")
for _, row in importance_df.head(10).iterrows():
    print(f"  {int(row['rank']):<6} {row['feature']:<30} "
          f"{row['mean_abs_shap']:>12.4f} {row['normalized']:>9.1f}%")

env_features = ['temperature_c', 'rainfall_mm', 'windspeed_kmh', 'cpi_normalized', 'fuel_normalized']
print(f"\nenvironmental feature rankings:")
for feat in env_features:
    row = importance_df[importance_df['feature'] == feat]
    if not row.empty:
        print(f"  {feat:<25} rank {int(row['rank'].values[0])}/{len(ALL_FEATURES)} "
              f"| mean |SHAP|: {row['mean_abs_shap'].values[0]:.4f}")

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

for item_nbr, product_name in PRODUCT_NAMES.items():
    mask = shap_df['item_nbr'] == item_nbr
    if mask.sum() == 0:
        continue
    product_shap       = shap_values[mask.values]
    product_importance = pd.DataFrame({
        'feature':       ALL_FEATURES,
        'mean_abs_shap': np.abs(product_shap).mean(axis=0)
    }).sort_values('mean_abs_shap', ascending=False).reset_index(drop=True)
    product_importance['rank']       = range(1, len(product_importance) + 1)
    product_importance['normalized'] = (
        product_importance['mean_abs_shap'] /
        product_importance['mean_abs_shap'].max() * 100
    )
    product_importance.to_csv(
        f'../results/shap/shap_product_importance/shap_{item_nbr}_importance.csv',
        index=False
    )

print(f"\nper-product SHAP saved for {len(PRODUCT_NAMES)} products")