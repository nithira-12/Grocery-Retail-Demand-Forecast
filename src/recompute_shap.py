"""
RECOMPUTE SHAP VALUES - ENVIRONMENTAL MODEL
============================================
Recomputes SHAP values for the new XGBoost model
trained with 21 features including environmental data.

HOW TO RUN:
    cd src
    python recompute_shap.py

INPUT:
    ../models/xgboost_model_v2.pkl
    ../models/feature_list.pkl
    ../data/processed/val_env.csv

OUTPUT:
    ../results/shap/complete_shap_values.csv      (updated)
    ../results/shap/shap_feature_importance.csv   (updated)
    ../results/shap/shap_product_importance/      (per product)
"""

import pandas as pd
import numpy as np
import pickle
import os
import shap
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("RECOMPUTE SHAP VALUES - 21 FEATURE MODEL")
print("=" * 60)

os.makedirs('../results/shap', exist_ok=True)
os.makedirs('../results/shap/shap_product_importance', exist_ok=True)

# ── LOAD MODEL AND FEATURES ───────────────────────────────
print("\n Loading XGBoost model v2...")

with open('../models/xgboost_model_v2.pkl', 'rb') as f:
    model = pickle.load(f)

with open('../models/feature_list.pkl', 'rb') as f:
    ALL_FEATURES = pickle.load(f)

print(f"   Model loaded: xgboost_model_v2.pkl")
print(f"   Features: {len(ALL_FEATURES)}")
print(f"   Feature list: {ALL_FEATURES}")

# ── LOAD VALIDATION DATA ──────────────────────────────────
print("\n Loading validation data...")

val = pd.read_csv('../data/processed/val_env.csv', parse_dates=['date'])

# Recreate lag features
print(" Creating lag features...")

def create_lag_features(df):
    df_lagged = df.copy()
    for lag in [7, 14, 28]:
        df_lagged[f'sales_lag_{lag}'] = df_lagged.groupby(
            ['store_nbr', 'item_nbr'])['unit_sales'].shift(lag)

    df_lagged['sales_rolling_mean_7'] = df_lagged.groupby(
        ['store_nbr', 'item_nbr'])['unit_sales'].transform(
        lambda x: x.shift(1).rolling(window=7,  min_periods=7).mean())

    df_lagged['sales_rolling_mean_28'] = df_lagged.groupby(
        ['store_nbr', 'item_nbr'])['unit_sales'].transform(
        lambda x: x.shift(1).rolling(window=28, min_periods=28).mean())

    df_lagged['sales_rolling_std_7'] = df_lagged.groupby(
        ['store_nbr', 'item_nbr'])['unit_sales'].transform(
        lambda x: x.shift(1).rolling(window=7,  min_periods=7).std())

    return df_lagged.dropna()

val_lagged = create_lag_features(val)
print(f"   Validation rows after lag: {len(val_lagged):,}")

X_val     = val_lagged[ALL_FEATURES]
y_val     = val_lagged['unit_sales']
y_val_pred = model.predict(X_val)

print(f"   X_val shape: {X_val.shape}")

# ── COMPUTE SHAP VALUES ───────────────────────────────────
print("\n Computing SHAP values using TreeExplainer...")
print("   This may take 1-3 minutes...")

import time
start = time.time()

explainer   = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_val)
base_value  = explainer.expected_value

elapsed = time.time() - start
print(f"   SHAP computation complete in {elapsed:.1f} seconds")
print(f"   SHAP values shape: {shap_values.shape}")
print(f"   Base value: {base_value:.4f}")

# ── BUILD COMPLETE SHAP DATAFRAME ─────────────────────────
print("\n Building complete SHAP dataframe...")

shap_df = pd.DataFrame(
    shap_values,
    columns=[f'shap_{f}' for f in ALL_FEATURES]
)

# Add metadata
shap_df['date']       = val_lagged['date'].values
shap_df['store_nbr']  = val_lagged['store_nbr'].values
shap_df['item_nbr']   = val_lagged['item_nbr'].values
shap_df['actual']     = y_val.values
shap_df['prediction'] = y_val_pred
shap_df['base_value'] = base_value

# Save complete SHAP values
shap_path = '../results/shap/complete_shap_values.csv'
shap_df.to_csv(shap_path, index=False)
print(f"   Saved: {shap_path}")
print(f"   Rows: {len(shap_df):,}")

# ── GLOBAL FEATURE IMPORTANCE ─────────────────────────────
print("\n Computing global feature importance...")

mean_abs_shap = np.abs(shap_values).mean(axis=0)

importance_df = pd.DataFrame({
    'feature':        ALL_FEATURES,
    'mean_abs_shap':  mean_abs_shap
}).sort_values('mean_abs_shap', ascending=False).reset_index(drop=True)

importance_df['rank']       = range(1, len(importance_df) + 1)
importance_df['normalized'] = importance_df['mean_abs_shap'] / importance_df['mean_abs_shap'].max() * 100

# Save global importance
imp_path = '../results/shap/shap_feature_importance.csv'
importance_df.to_csv(imp_path, index=False)
print(f"   Saved: {imp_path}")

print(f"\n Top 10 Features by SHAP Importance:")
print(f"   {'Rank':<6} {'Feature':<30} {'Mean |SHAP|':>12} {'Normalized':>10}")
print("   " + "-" * 62)
for _, row in importance_df.head(10).iterrows():
    print(f"   {int(row['rank']):<6} {row['feature']:<30} "
          f"{row['mean_abs_shap']:>12.4f} {row['normalized']:>9.1f}%")

# Check where environmental features ranked
print(f"\n Environmental Feature Rankings:")
env_features = ['temperature_c', 'rainfall_mm', 'windspeed_kmh',
                'cpi_normalized', 'fuel_normalized']
for feat in env_features:
    row = importance_df[importance_df['feature'] == feat]
    if not row.empty:
        print(f"   {feat:<25} → Rank {int(row['rank'].values[0])}/{len(ALL_FEATURES)} "
              f"| Mean |SHAP|: {row['mean_abs_shap'].values[0]:.4f}")

# ── PRODUCT LEVEL SHAP ────────────────────────────────────
print("\n Computing product-level SHAP importance...")

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

    product_shap = shap_values[mask.values]
    product_importance = pd.DataFrame({
        'feature':       ALL_FEATURES,
        'mean_abs_shap': np.abs(product_shap).mean(axis=0)
    }).sort_values('mean_abs_shap', ascending=False).reset_index(drop=True)

    product_importance['rank']       = range(1, len(product_importance) + 1)
    product_importance['normalized'] = (
        product_importance['mean_abs_shap'] /
        product_importance['mean_abs_shap'].max() * 100
    )

    save_path = f'../results/shap/shap_product_importance/shap_{item_nbr}_importance.csv'
    product_importance.to_csv(save_path, index=False)

print(f"   Product-level SHAP saved for {len(PRODUCT_NAMES)} products")
print(f"   Location: ../results/shap/shap_product_importance/")

# ── SUMMARY ───────────────────────────────────────────────
print(f"\n" + "=" * 60)
print("SHAP RECOMPUTATION COMPLETE")
print("=" * 60)

print(f"""
 Files updated:
   ../results/shap/complete_shap_values.csv
   ../results/shap/shap_feature_importance.csv
   ../results/shap/shap_product_importance/ (11 files)

 Key stats:
   Total explanations: {len(shap_df):,}
   Features explained: {len(ALL_FEATURES)}
   Base value: {base_value:.2f} units
   Computation time: {elapsed:.1f} seconds

 Top feature: {importance_df.iloc[0]['feature']}
 Environmental features included: {len(env_features)} new features ranked

 NEXT STEP: Run python dashboard.py to view updated results
""")