"""
XGBOOST MODEL - GRADIENT BOOSTING
==================================
XGBoost (Extreme Gradient Boosting) for demand forecasting.
Expected to be the best-performing model due to:
- Non-linear pattern detection
- Feature interactions (e.g., promotion × weekend)
- Robust to outliers
- Handles complex relationships

WHY XGBOOST?
- Industry standard for tabular data (retail, finance, etc.)
- Wins most Kaggle competitions
- Fast training with GPU support
- Built-in feature importance
- Perfect for SHAP explainability (tree-based)
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import warnings
import os
warnings.filterwarnings('ignore')

print("\n" )
print("XGBOOST MODEL - DEMAND FORECASTING")
print("\n" )

# CONFIGURATION


PROCESSED_DATA_DIR = '../data/processed'
RESULTS_DIR = '../results'
MODELS_DIR = '../models'

# Create directories
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

SELECTED_STORES = [44, 51]
STORE_NAMES = {44: "Store Colombo", 51: "Store Gampaha"}

print("\n✓ Configuration loaded")
print(f"  Working with: {len(SELECTED_STORES)} stores")
print(f"  Results will be saved to: {RESULTS_DIR}/")


# STEP 1: LOAD PROCESSED DATA


print("\n" )
print("STEP 1: LOADING PROCESSED DATA")
print("\n" )

train = pd.read_csv(f'{PROCESSED_DATA_DIR}/train_processed.csv', parse_dates=['date'])
val = pd.read_csv(f'{PROCESSED_DATA_DIR}/val_processed.csv', parse_dates=['date'])
test = pd.read_csv(f'{PROCESSED_DATA_DIR}/test_processed.csv', parse_dates=['date'])

print(f"\n Data loaded successfully:")
print(f"  train: {train.shape[0]:,} rows ({train['date'].min()} to {train['date'].max()})")
print(f"  validation: {val.shape[0]:,} rows ({val['date'].min()} to {val['date'].max()})")
print(f"  test: {test.shape[0]:,} rows ({test['date'].min()} to {test['date'].max()})")

print(f"\n Products: {train['item_nbr'].nunique()}")
print(f"  Product IDs: {sorted(train['item_nbr'].unique())}")



# STEP 2: CREATE LAG FEATURES (SAME AS LINEAR REGRESSION)


print("\n")
print("STEP 2: CREATING LAG FEATURES")
print("\n" )

print("\n XGBoost needs the same lag features as Linear Regression")
print("  (We already did this work for LR!)")

def create_lag_features(df, lag_days=[7, 14, 28]):
    """
    Create lag features for XGBoost.
    SAME FUNCTION as Linear Regression!
    
    WHY REUSE?
    XGBoost needs explicit temporal features just like Linear Regression.
    Unlike Prophet, XGBoost doesn't automatically detect seasonality.
    """
    
    print(f"\n Creating lag features for periods: {lag_days}")
    
    df_lagged = df.copy()
    
    # Point lags
    for lag in lag_days:
        print(f"  Creating lag_{lag}...")
        df_lagged[f'sales_lag_{lag}'] = df_lagged.groupby(['store_nbr', 'item_nbr'])['unit_sales'].shift(lag)
    
    # Rolling statistics
    print(f"  Creating rolling features...")
    df_lagged['sales_rolling_mean_7'] = df_lagged.groupby(['store_nbr', 'item_nbr'])['unit_sales'].transform(
        lambda x: x.shift(1).rolling(window=7, min_periods=7).mean()
    )
    
    df_lagged['sales_rolling_mean_28'] = df_lagged.groupby(['store_nbr', 'item_nbr'])['unit_sales'].transform(
        lambda x: x.shift(1).rolling(window=28, min_periods=28).mean()
    )
    
    df_lagged['sales_rolling_std_7'] = df_lagged.groupby(['store_nbr', 'item_nbr'])['unit_sales'].transform(
        lambda x: x.shift(1).rolling(window=7, min_periods=7).std()
    )
    
    print(f" Created 6 lag features")
    
    return df_lagged

# Apply to all datasets
train_lagged = create_lag_features(train)
val_lagged = create_lag_features(val)
test_lagged = create_lag_features(test)

# Drop NaN rows
print("\n Handling missing values from lag features...")
print(f"  Before: Train={train_lagged.shape[0]:,}, Val={val_lagged.shape[0]:,}")

train_lagged = train_lagged.dropna()
val_lagged = val_lagged.dropna()
test_lagged = test_lagged.dropna()

print(f"  After:  Train={train_lagged.shape[0]:,}, Val={val_lagged.shape[0]:,}")
print(f"  Lost {train.shape[0] - train_lagged.shape[0]:,} rows (first 28 days)")


# STEP 3: DEFINE FEATURES FOR XGBOOST


print("\n")
print("STEP 3: DEFINING FEATURES FOR XGBOOST")
print("\n" )

# TARGET
TARGET = 'unit_sales'

# FEATURES (Same as Linear Regression!)
# Feature groups
LAG_FEATURES = [
    'sales_lag_7', 'sales_lag_14', 'sales_lag_28',
    'sales_rolling_mean_7', 'sales_rolling_mean_28', 'sales_rolling_std_7'
]

TEMPORAL_FEATURES = [
    'dayofweek', 'month', 'quarter', 'day',
    'is_weekend', 'is_month_start', 'is_month_end'
]

# ← CHANGED: Added holiday_impact instead of is_holiday
EXTERNAL_FEATURES = ['onpromotion', 'holiday_impact', 'family_encoded']

# COMBINE ALL
ALL_FEATURES = LAG_FEATURES + TEMPORAL_FEATURES + EXTERNAL_FEATURES

print(f"\n✓ Feature groups:")
print(f"  Lag features: {len(LAG_FEATURES)}")
print(f"  Temporal features: {len(TEMPORAL_FEATURES)}")
print(f"  External features: {len(EXTERNAL_FEATURES)}")
print(f"  TOTAL: {len(ALL_FEATURES)} features")

# Verify
missing = [f for f in ALL_FEATURES if f not in train_lagged.columns]
if missing:
    print(f"  missing: {missing}")
else:
    print(f"   All features found")



# STEP 4: PREPARE DATA FOR XGBOOST


print("\n")
print("STEP 4: PREPARING DATA FOR XGBOOST")
print("\n" )

# Extract features and target
X_train = train_lagged[ALL_FEATURES].copy()
y_train = train_lagged[TARGET].copy()

X_val = val_lagged[ALL_FEATURES].copy()
y_val = val_lagged[TARGET].copy()

print(f"\n training set:")
print(f"  X_train: {X_train.shape} ({X_train.shape[0]:,} samples, {X_train.shape[1]} features)")
print(f"  y_train: {y_train.shape}")

print(f"\n validation set:")
print(f"  X_val: {X_val.shape}")
print(f"  y_val: {y_val.shape}")

# XGBoost can handle non-scaled data, but scaling can help
# OPTIONAL: We'll skip scaling for XGBoost (it's tree-based, doesn't need it)
print(f"\n Note: XGBoost doesn't require feature scaling (tree-based model)")


# STEP 5: TRAIN XGBOOST MODEL


print("\n" )
print("STEP 5: TRAINING XGBOOST MODEL")
print("\n" )

print("\n XGBoost Hyperparameters:")
print("  n_estimators: Number of trees (more = better, but slower)")
print("  max_depth: Tree depth (deeper = more complex)")
print("  learning_rate: Step size (smaller = more careful learning)")
print("  subsample: Fraction of data per tree (prevents overfitting)")

# Initialize XGBoost model
model = xgb.XGBRegressor(
    n_estimators=100,        # Number of boosting rounds (trees)
    max_depth=6,             # Maximum tree depth
    learning_rate=0.1,       # Step size for updates
    subsample=0.8,           # Use 80% of data per tree (prevents overfitting)
    colsample_bytree=0.8,    # Use 80% of features per tree
    random_state=42,         # For reproducibility
    n_jobs=-1,               # Use all CPU cores
    verbosity=0              # Suppress training logs
)

print(f"\n→ Training XGBoost model...")
print(f"  Training on {X_train.shape[0]:,} samples with {X_train.shape[1]} features...")
print(f"  This may take 2-3 minutes...")

# Train model
model.fit(
    X_train, 
    y_train,
    eval_set=[(X_val, y_val)],  # Monitor validation performance
    verbose=False                # Don't print each iteration
)

print(f"\n XGBoost model trained successfully!")
print(f"  Total trees: {model.n_estimators}")
print(f"  Training complete")


# STEP 6: MAKE PREDICTIONS & EVALUATE


print("\n")
print("STEP 6: MAKING PREDICTIONS & EVALUATING")
print("\n" )

# Predict on validation set
print(f"\n Generating predictions on validation set...")
y_val_pred = model.predict(X_val)
print(f"  Generated {len(y_val_pred):,} predictions")

# Calculate metrics
print(f"\n→ Calculating performance metrics...")

rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
mae = mean_absolute_error(y_val, y_val_pred)
r2 = r2_score(y_val, y_val_pred)
mape = np.mean(np.abs((y_val - y_val_pred) / y_val)) * 100

print("\n" )
print("VALIDATION SET PERFORMANCE")
print("\n" )
print(f"  RMSE: {rmse:.2f} units")
print(f"  MAE:  {mae:.2f} units")
print(f"  R squared:   {r2:.4f} ({r2*100:.2f}%)")
print(f"  MAPE: {mape:.2f}%")
print("\n" )

print(f"\n Interpretation:")
print(f"  On average, predictions are off by ±{mae:.1f} units")
print(f"  Model explains {r2*100:.1f}% of variance")
print(f"  Typical error is ~{mape:.1f}% of actual sales")

# Performance check
if r2 > 0.90:
    print(f"\n  EXCELLENT R sq > 0.90 - XGBoost is the clear winner")
elif r2 > 0.85:
    print(f"\n   GOOD R sq > 0.85 - Solid performance")
elif r2 > 0.80:
    print(f"\n   OK - May need hyperparameter tuning")
else:
    print(f"\n   POOR - Something went wrong")



# STEP 7: FEATURE IMPORTANCE ANALYSIS


print("\n" + "="*60)
print("STEP 7: ANALYZING FEATURE IMPORTANCE")
print("="*60)

print(f"\n→ XGBoost feature importance (gain):")
print("  'Gain' = Average improvement when feature is used to split")
print("  Higher gain = more important feature")

# Get feature importance (already has proper names!)
importance_dict = model.get_booster().get_score(importance_type='gain')

# Convert to DataFrame
importance_df = pd.DataFrame({
    'feature': list(importance_dict.keys()),
    'importance': list(importance_dict.values())
}).sort_values('importance', ascending=False)

print(f"\n Top 10 Most Important Features:")
for idx, row in importance_df.head(10).iterrows():
    print(f"  {str(row['feature']):30s} → {row['importance']:,.1f}")

print(f"\n Key insights:")
if len(importance_df) > 0:
    top_feature = importance_df.iloc[0]['feature']
    print(f"  Most important: {top_feature}")
    
    if 'sales_lag_7' in importance_df['feature'].values[:3]:
        print(f"  Recent sales (lag_7) highly important")
    if 'sales_rolling_mean_28' in importance_df['feature'].values[:3]:
        print(f"   28-day rolling average crucial")
    if 'onpromotion' in importance_df['feature'].values[:10]:
        onprom_rank = importance_df[importance_df['feature'] == 'onpromotion'].index[0] + 1
        print(f"  Promotions ranked #{onprom_rank}")
    if 'holiday_impact' in importance_df['feature'].values:
        holiday_rank = importance_df[importance_df['feature'] == 'holiday_impact'].index[0] + 1
        print(f"   holiday_impact ranked #{holiday_rank}")
    else:
        print(f"  holiday_impact NOT in model features!")

# STEP 8: ANALYZE BY PRODUCT & STORE


print("\n" )
print("STEP 8: ANALYZING PREDICTIONS BY STORE & PRODUCT")
print("\n" )

# Add predictions to validation dataframe
val_results = val_lagged.copy()
val_results['predictions'] = y_val_pred
val_results['actual'] = y_val.values
val_results['error'] = val_results['actual'] - val_results['predictions']
val_results['abs_error'] = val_results['error'].abs()
val_results['pct_error'] = (val_results['abs_error'] / val_results['actual']) * 100

# By product
print(f"\n Performance by Product:")
product_perf = val_results.groupby('item_nbr').agg({
    'actual': 'mean',
    'predictions': 'mean',
    'abs_error': 'mean',
    'pct_error': 'mean'
}).round(2)
product_perf.columns = ['Avg Actual', 'Avg Predicted', 'MAE', 'MAPE (%)']
product_perf = product_perf.sort_values('MAE', ascending=False)
print(product_perf)

# By store
print(f"\n Performance by Store:")
store_perf = val_results.groupby('store_nbr').agg({
    'actual': 'mean',
    'predictions': 'mean',
    'abs_error': 'mean',
    'pct_error': 'mean'
}).round(2)
store_perf.columns = ['Avg Actual', 'Avg Predicted', 'MAE', 'MAPE (%)']
store_perf.index = store_perf.index.map(STORE_NAMES)
print(store_perf)



# STEP 9: SAVE RESULTS

print("\n" )
print("STEP 9: SAVING RESULTS")
print("\n" )

# Save overall metrics
metrics_df = pd.DataFrame({
    'Model': ['XGBoost'],
    'RMSE': [rmse],
    'MAE': [mae],
    'R2': [r2],
    'MAPE': [mape]
})

metrics_file = f'{RESULTS_DIR}/xgboost_metrics.csv'
metrics_df.to_csv(metrics_file, index=False)
print(f"\n Metrics saved to: {metrics_file}")

# Save feature importance
importance_file = f'{RESULTS_DIR}/xgboost_feature_importance.csv'
importance_df.to_csv(importance_file, index=False)
print(f" Feature importance saved to: {importance_file}")

# Save predictions
predictions_file = f'{RESULTS_DIR}/xgboost_predictions.csv'
val_results[['date', 'store_nbr', 'item_nbr', 'actual', 'predictions', 'error']].to_csv(
    predictions_file, index=False
)
print(f" Predictions saved to: {predictions_file}")

# Save trained model
import pickle
model_file = f'{MODELS_DIR}/xgboost_model.pkl'
with open(model_file, 'wb') as f:
    pickle.dump(model, f)
print(f" Trained model saved to: {model_file}")
print(f"  (We'll use this for SHAP explanations!)")

print("\n" )
print("XGBOOST MODEL COMPLETE!")
print("\n" )

print(f"\n Summary:")
print(f"  Trained on: {X_train.shape[0]:,} samples")
print(f"  Validated on: {X_val.shape[0]:,} samples")
print(f"  RMSE: {rmse:.2f}")
print(f"  MAE: {mae:.2f}")
print(f"  R sq: {r2:.4f}")

