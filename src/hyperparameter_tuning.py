"""
XGBOOST HYPERPARAMETER TUNING
==============================
Optimize XGBoost parameters using GridSearchCV to improve model performance.

CURRENT PERFORMANCE (Before Tuning):
- RMSE: 67.91
- MAE: 35.33
- R²: 0.8882 (88.82%)

TARGET (After Tuning):
- RMSE: ~60-65
- MAE: ~30-33
- R²: ~0.90-0.92 (90-92%)

METHOD: Grid Search with Cross-Validation
- Tests multiple parameter combinations
- Uses 3-fold time series cross-validation
- Finds optimal settings automatically

RUNTIME: ~2-3 hours (tests 81 combinations)
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, make_scorer
import pickle
import warnings
import time
import os
warnings.filterwarnings('ignore')

print("\n" + "="*70)
print("XGBOOST HYPERPARAMETER TUNING")
print("="*70)

# ============================================================================
# CONFIGURATION
# ============================================================================

PROCESSED_DATA_DIR = '../data/processed'
RESULTS_DIR = '../results'
MODELS_DIR = '../models'

# Create directories
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

STORE_NAMES = {44: "Store Colombo", 51: "Store Gampaha"}

print("\n✓ Configuration loaded")
print(f"  Results directory: {RESULTS_DIR}/")
print(f"  Models directory: {MODELS_DIR}/")

# ============================================================================
# STEP 1: LOAD PROCESSED DATA
# ============================================================================

print("\n" + "="*70)
print("STEP 1: LOADING PROCESSED DATA")
print("="*70)

train = pd.read_csv(f'{PROCESSED_DATA_DIR}/train_processed.csv', parse_dates=['date'])
val = pd.read_csv(f'{PROCESSED_DATA_DIR}/val_processed.csv', parse_dates=['date'])

print(f"\n✓ Data loaded:")
print(f"  Train: {train.shape[0]:,} rows")
print(f"  Validation: {val.shape[0]:,} rows")


# ============================================================================
# STEP 2: CREATE LAG FEATURES (SAME AS BEFORE)
# ============================================================================

print("\n" + "="*70)
print("STEP 2: CREATING LAG FEATURES")
print("="*70)

def create_lag_features(df, lag_days=[7, 14, 28]):
    """Same lag features as original XGBoost"""
    
    df_lagged = df.copy()
    
    # Point lags
    for lag in lag_days:
        df_lagged[f'sales_lag_{lag}'] = df_lagged.groupby(['store_nbr', 'item_nbr'])['unit_sales'].shift(lag)
    
    # Rolling statistics
    df_lagged['sales_rolling_mean_7'] = df_lagged.groupby(['store_nbr', 'item_nbr'])['unit_sales'].transform(
        lambda x: x.shift(1).rolling(window=7, min_periods=7).mean()
    )
    
    df_lagged['sales_rolling_mean_28'] = df_lagged.groupby(['store_nbr', 'item_nbr'])['unit_sales'].transform(
        lambda x: x.shift(1).rolling(window=28, min_periods=28).mean()
    )
    
    df_lagged['sales_rolling_std_7'] = df_lagged.groupby(['store_nbr', 'item_nbr'])['unit_sales'].transform(
        lambda x: x.shift(1).rolling(window=7, min_periods=7).std()
    )
    
    return df_lagged

train_lagged = create_lag_features(train)
val_lagged = create_lag_features(val)

train_lagged = train_lagged.dropna()
val_lagged = val_lagged.dropna()

print(f"\n✓ Lag features created")
print(f"  Train: {train_lagged.shape[0]:,} rows")
print(f"  Validation: {val_lagged.shape[0]:,} rows")


# ============================================================================
# STEP 3: PREPARE FEATURES
# ============================================================================

print("\n" + "="*70)
print("STEP 3: PREPARING FEATURES")
print("="*70)

# Same 16 features as before
LAG_FEATURES = [
    'sales_lag_7', 'sales_lag_14', 'sales_lag_28',
    'sales_rolling_mean_7', 'sales_rolling_mean_28', 'sales_rolling_std_7'
]

TEMPORAL_FEATURES = [
    'dayofweek', 'month', 'quarter', 'day',
    'is_weekend', 'is_month_start', 'is_month_end'
]

EXTERNAL_FEATURES = ['onpromotion', 'is_holiday', 'family_encoded']

ALL_FEATURES = LAG_FEATURES + TEMPORAL_FEATURES + EXTERNAL_FEATURES

print(f"\n✓ Using {len(ALL_FEATURES)} features (same as original model)")

# Extract features
X_train = train_lagged[ALL_FEATURES].copy()
y_train = train_lagged['unit_sales'].copy()

X_val = val_lagged[ALL_FEATURES].copy()
y_val = val_lagged['unit_sales'].copy()

print(f"\n✓ Data prepared:")
print(f"  X_train: {X_train.shape}")
print(f"  X_val: {X_val.shape}")

# ============================================================================
# STEP 4: DEFINE HYPERPARAMETER GRID
# ============================================================================

print("\n" + "="*70)
print("STEP 4: DEFINING HYPERPARAMETER GRID")
print("="*70)

print("\n→ Parameters to tune:")

# Parameter grid for GridSearch
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [6, 8, 10],
    'learning_rate': [0.05, 0.1, 0.15],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9]
}

print("\n  n_estimators (number of trees):")
print(f"    Testing: {param_grid['n_estimators']}")
print(f"    Current: 100")
print(f"    More trees = Better learning, but slower")

print("\n  max_depth (tree depth):")
print(f"    Testing: {param_grid['max_depth']}")
print(f"    Current: 6")
print(f"    Deeper = More complex patterns, but overfitting risk")

print("\n  learning_rate (step size):")
print(f"    Testing: {param_grid['learning_rate']}")
print(f"    Current: 0.1")
print(f"    Lower = More careful learning, needs more trees")

print("\n  subsample (data fraction per tree):")
print(f"    Testing: {param_grid['subsample']}")
print(f"    Current: 0.8")
print(f"    Lower = More randomness, prevents overfitting")

print("\n  colsample_bytree (feature fraction per tree):")
print(f"    Testing: {param_grid['colsample_bytree']}")
print(f"    Current: 0.8")
print(f"    Lower = More feature diversity, prevents overfitting")

# Calculate total combinations
total_combinations = 1
for param, values in param_grid.items():
    total_combinations *= len(values)

print(f"\n✓ Parameter grid defined:")
print(f"  Total combinations: {total_combinations}")
print(f"  With 3-fold CV: {total_combinations * 3} models to train")
print(f"  Estimated time: ~2-3 hours")

# ============================================================================
# STEP 5: RUN GRID SEARCH WITH CROSS-VALIDATION
# ============================================================================

print("\n" + "="*70)
print("STEP 5: RUNNING GRID SEARCH")
print("="*70)

print("\n→ Setting up cross-validation...")

# Time series cross-validation (respects temporal order)
tscv = TimeSeriesSplit(n_splits=3)

print(f"  Using TimeSeriesSplit with 3 folds")
print(f"  This respects temporal order (no future data leakage)")

# Initialize base model
base_model = xgb.XGBRegressor(
    random_state=42,
    n_jobs=-1,
    verbosity=0
)

# MAE scorer (negative because GridSearch maximizes)
mae_scorer = make_scorer(mean_absolute_error, greater_is_better=False)

print(f"\n→ Creating GridSearchCV...")
print(f"  Optimization metric: MAE (Mean Absolute Error)")
print(f"  Lower MAE = Better model")

# Create GridSearch
grid_search = GridSearchCV(
    estimator=base_model,
    param_grid=param_grid,
    cv=tscv,
    scoring=mae_scorer,
    verbose=2,  # Show progress
    n_jobs=-1,  # Use all CPU cores
    return_train_score=True
)

print(f"\n" + "="*70)
print(f"🚀 STARTING GRID SEARCH")
print(f"  Testing {total_combinations} parameter combinations")
print(f"  You can monitor progress below:")
print("="*70 + "\n")

# Record start time
start_time = time.time()

# RUN THE SEARCH!
grid_search.fit(X_train, y_train)

# Record end time
end_time = time.time()
duration_minutes = (end_time - start_time) / 60

print(f"\n" + "="*70)
print(f"✓ GRID SEARCH COMPLETE!")
print(f"  Duration: {duration_minutes:.1f} minutes ({duration_minutes/60:.1f} hours)")
print("="*70)

# ============================================================================
# STEP 6: ANALYZE GRID SEARCH RESULTS
# ============================================================================

print("\n" + "="*70)
print("STEP 6: ANALYZING GRID SEARCH RESULTS")
print("="*70)

# Get best parameters
best_params = grid_search.best_params_
best_score = -grid_search.best_score_  # Negative because we minimized negative MAE

print(f"\n🏆 BEST PARAMETERS FOUND:")
print("="*70)
for param, value in best_params.items():
    print(f"  {param:20s} = {value}")
print("="*70)

print(f"\n→ Cross-validation performance:")
print(f"  Best CV MAE: {best_score:.2f} units")
print(f"  (This is the average MAE across 3 folds)")

# Get best model
best_model = grid_search.best_estimator_

print(f"\n✓ Best model extracted from GridSearch")


# ============================================================================
# STEP 7: EVALUATE BEST MODEL ON VALIDATION SET
# ============================================================================

print("\n" + "="*70)
print("STEP 7: EVALUATING BEST MODEL ON VALIDATION SET")
print("="*70)

print(f"\n→ Making predictions on validation set...")

# Predict with best model
y_val_pred_tuned = best_model.predict(X_val)

# Calculate metrics
rmse_tuned = np.sqrt(mean_squared_error(y_val, y_val_pred_tuned))
mae_tuned = mean_absolute_error(y_val, y_val_pred_tuned)
r2_tuned = r2_score(y_val, y_val_pred_tuned)
mape_tuned = np.mean(np.abs((y_val - y_val_pred_tuned) / y_val)) * 100

print(f"\n" + "="*70)
print("TUNED MODEL PERFORMANCE (Validation Set)")
print("="*70)
print(f"  RMSE: {rmse_tuned:.2f} units")
print(f"  MAE:  {mae_tuned:.2f} units")
print(f"  R²:   {r2_tuned:.4f} ({r2_tuned*100:.2f}%)")
print(f"  MAPE: {mape_tuned:.2f}%")
print("="*70)


# ============================================================================
# STEP 8: COMPARE WITH ORIGINAL MODEL
# ============================================================================

print("\n" + "="*70)
print("STEP 8: COMPARING WITH ORIGINAL MODEL")
print("="*70)

# Original model performance (from your previous run)
original_metrics = {
    'RMSE': 67.91,
    'MAE': 35.33,
    'R2': 0.8882,
    'MAPE': 50.64
}

print(f"\n📊 PERFORMANCE COMPARISON:")
print("="*70)
print(f"{'Metric':<10} {'Original':<12} {'Tuned':<12} {'Change':<15}")
print("-"*70)

# RMSE comparison
rmse_change = ((rmse_tuned - original_metrics['RMSE']) / original_metrics['RMSE']) * 100
rmse_status = "IMPROVED" if rmse_tuned < original_metrics['RMSE'] else "WORSE"
print(f"{'RMSE':<10} {original_metrics['RMSE']:<12.2f} {rmse_tuned:<12.2f} {rmse_change:>+6.1f}% {rmse_status}")

# MAE comparison
mae_change = ((mae_tuned - original_metrics['MAE']) / original_metrics['MAE']) * 100
mae_status = "IMPROVED" if mae_tuned < original_metrics['MAE'] else "WORSE"
print(f"{'MAE':<10} {original_metrics['MAE']:<12.2f} {mae_tuned:<12.2f} {mae_change:>+6.1f}% {mae_status}")

# R² comparison
r2_change = ((r2_tuned - original_metrics['R2']) / original_metrics['R2']) * 100
r2_status = "IMPROVED" if r2_tuned > original_metrics['R2'] else "WORSE"
print(f"{'R²':<10} {original_metrics['R2']:<12.4f} {r2_tuned:<12.4f} {r2_change:>+6.1f}% {r2_status}")

# MAPE comparison
mape_change = ((mape_tuned - original_metrics['MAPE']) / original_metrics['MAPE']) * 100
mape_status = "IMPROVED" if mape_tuned < original_metrics['MAPE'] else "WORSE"
print(f"{'MAPE':<10} {original_metrics['MAPE']:<12.2f} {mape_tuned:<12.2f} {mape_change:>+6.1f}% {mape_status}")

print("="*70)

# Overall assessment
if r2_tuned > original_metrics['R2']:
    print(f"\n🎉 SUCCESS! Tuned model is better!")
    print(f"   R² improved by {abs(r2_change):.1f}%")
    print(f"   MAE improved by {abs(mae_change):.1f}%")
else:
    print(f"\n Tuned model did not improve significantly")
    print(f"   Consider: Original parameters were already near-optimal")


# ============================================================================
# STEP 9: DETAILED PARAMETER IMPACT ANALYSIS
# ============================================================================

print("\n" + "="*70)
print("STEP 9: PARAMETER IMPACT ANALYSIS")
print("="*70)

# Get all results
cv_results = pd.DataFrame(grid_search.cv_results_)

# Sort by mean test score (best first)
cv_results = cv_results.sort_values('mean_test_score', ascending=False)

print(f"\n→ Top 5 Parameter Combinations:")
print("="*70)

for i in range(min(5, len(cv_results))):
    rank = i + 1
    row = cv_results.iloc[i]
    
    mae_score = -row['mean_test_score']  # Convert back to positive MAE
    
    print(f"\nRank {rank}:")
    print(f"  MAE: {mae_score:.2f}")
    print(f"  n_estimators: {row['param_n_estimators']}")
    print(f"  max_depth: {row['param_max_depth']}")
    print(f"  learning_rate: {row['param_learning_rate']}")
    print(f"  subsample: {row['param_subsample']}")
    print(f"  colsample_bytree: {row['param_colsample_bytree']}")

print("="*70)


# ============================================================================
# STEP 10: SAVE RESULTS & BEST MODEL
# ============================================================================

print("\n" + "="*70)
print("STEP 10: SAVING RESULTS & BEST MODEL")
print("="*70)

# Save tuned model metrics
tuned_metrics = pd.DataFrame({
    'Model': ['XGBoost_Tuned'],
    'RMSE': [rmse_tuned],
    'MAE': [mae_tuned],
    'R2': [r2_tuned],
    'MAPE': [mape_tuned]
})

metrics_file = f'{RESULTS_DIR}/xgboost_tuned_metrics.csv'
tuned_metrics.to_csv(metrics_file, index=False)
print(f"\n✓ Tuned metrics saved to: {metrics_file}")

# Save best parameters
best_params_df = pd.DataFrame([best_params])
params_file = f'{RESULTS_DIR}/xgboost_best_parameters.csv'
best_params_df.to_csv(params_file, index=False)
print(f"✓ Best parameters saved to: {params_file}")

# Save detailed CV results
cv_results_file = f'{RESULTS_DIR}/xgboost_cv_results.csv'
cv_results.to_csv(cv_results_file, index=False)
print(f"✓ Full CV results saved to: {cv_results_file}")

# Save comparison
comparison_df = pd.DataFrame({
    'Model': ['Original', 'Tuned'],
    'RMSE': [original_metrics['RMSE'], rmse_tuned],
    'MAE': [original_metrics['MAE'], mae_tuned],
    'R2': [original_metrics['R2'], r2_tuned],
    'MAPE': [original_metrics['MAPE'], mape_tuned]
})

comparison_file = f'{RESULTS_DIR}/xgboost_comparison.csv'
comparison_df.to_csv(comparison_file, index=False)
print(f"✓ Comparison saved to: {comparison_file}")

# Save tuned model (OVERWRITE original)
print(f"\n→ Saving tuned model...")

user_choice = input("\nDo you want to REPLACE the original model with the tuned model? (yes/no): ").lower()

if user_choice == 'yes':
    model_file = f'{MODELS_DIR}/xgboost_model.pkl'
    
    # Backup original first
    backup_file = f'{MODELS_DIR}/xgboost_model_original_backup.pkl'
    if os.path.exists(model_file):
        import shutil
        shutil.copy(model_file, backup_file)
        print(f"✓ Original model backed up to: {backup_file}")
    
    # Save new model
    with open(model_file, 'wb') as f:
        pickle.dump(best_model, f)
    
    print(f"✓ Tuned model saved to: {model_file}")
    print(f"  (Original model backed up)")
    
else:
    # Save as separate file
    tuned_model_file = f'{MODELS_DIR}/xgboost_model_tuned.pkl'
    with open(tuned_model_file, 'wb') as f:
        pickle.dump(best_model, f)
    
    print(f"✓ Tuned model saved to: {tuned_model_file}")
    print(f"  (Original model unchanged)")


# ============================================================================
# STEP 11: GENERATE TUNED PREDICTIONS
# ============================================================================

print("\n" + "="*70)
print("STEP 11: GENERATING PREDICTIONS WITH TUNED MODEL")
print("="*70)

# Create predictions dataframe
val_results_tuned = val_lagged.copy()
val_results_tuned['predictions'] = y_val_pred_tuned
val_results_tuned['actual'] = y_val.values
val_results_tuned['error'] = val_results_tuned['actual'] - val_results_tuned['predictions']
val_results_tuned['abs_error'] = val_results_tuned['error'].abs()

# Save predictions
predictions_file = f'{RESULTS_DIR}/xgboost_tuned_predictions.csv'
val_results_tuned[['date', 'store_nbr', 'item_nbr', 'actual', 'predictions', 'error']].to_csv(
    predictions_file, index=False
)
print(f"\n✓ Predictions saved to: {predictions_file}")

# Product-level performance
print(f"\n→ Performance by Product:")
product_perf = val_results_tuned.groupby('item_nbr').agg({
    'actual': 'mean',
    'predictions': 'mean',
    'abs_error': 'mean'
}).round(2)
product_perf.columns = ['Avg Actual', 'Avg Predicted', 'MAE']
product_perf = product_perf.sort_values('MAE')

print(product_perf)

# Save product performance
product_perf_file = f'{RESULTS_DIR}/xgboost_tuned_product_performance.csv'
product_perf.to_csv(product_perf_file)
print(f"\n✓ Product performance saved to: {product_perf_file}")


# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "="*70)
print("HYPERPARAMETER TUNING COMPLETE!")
print("="*70)

print(f"\n✓ Summary:")
print(f"  Duration: {duration_minutes:.1f} minutes")
print(f"  Combinations tested: {total_combinations}")
print(f"  Best MAE: {mae_tuned:.2f} units")
print(f"  Best R²: {r2_tuned:.4f} ({r2_tuned*100:.2f}%)")

print(f"\n✓ Best Parameters:")
for param, value in best_params.items():
    print(f"  {param}: {value}")

print(f"\n✓ Improvement over original:")
print(f"  RMSE: {rmse_change:+.1f}%")
print(f"  MAE: {mae_change:+.1f}%")
print(f"  R²: {r2_change:+.1f}%")

print(f"\n✓ Files saved:")
print(f"  1. {metrics_file}")
print(f"  2. {params_file}")
print(f"  3. {cv_results_file}")
print(f"  4. {comparison_file}")
print(f"  5. {predictions_file}")
print(f"  6. {product_perf_file}")
if user_choice == 'yes':
    print(f"  7. {model_file} (tuned model)")
    print(f"  8. {backup_file} (original backup)")
else:
    print(f"  7. {tuned_model_file} (tuned model)")

print(f"\n✓ Next steps:")
print(f"  1. Replace Ecuador holidays with Sri Lankan holidays")
print(f"  2. Re-run data_processing.py")
print(f"  3. Re-run all models with new data")
print(f"  4. Build Streamlit dashboard")
print(f"  5. Add replenishment extension (optional)")

print("="*70 + "\n")