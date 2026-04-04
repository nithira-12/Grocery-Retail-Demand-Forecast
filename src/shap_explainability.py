"""
SHAP EXPLAINABILITY - EXPLAINABLE AI (XAI)
==========================================
SHAP (SHapley Additive exPlanations) makes XGBoost predictions interpretable.

WHY SHAP?
- Shows which features drive each prediction
- Provides local explanations (per-product, per-day)
- Provides global explanations (overall feature importance)
- Industry standard for model explainability
- Perfect for tree-based models like XGBoost

WHAT WE'LL CREATE:
1. SHAP values for all validation predictions
2. Feature importance rankings
3. Visualizations (bar charts, waterfall plots)
4. Per-product explanations
5. Saved results for dashboard integration
"""

import pandas as pd
import numpy as np
import shap
import pickle
import matplotlib.pyplot as plt
import warnings
import os
warnings.filterwarnings('ignore')

print("\n")
print("SHAP EXPLAINABILITY - MAKING PREDICTIONS INTERPRETABLE")
print("\n")


# CONFIGURATION

PROCESSED_DATA_DIR = '../data/processed'
RESULTS_DIR = '../results'
MODELS_DIR = '../models'
SHAP_DIR = '../results/shap'  # New directory for SHAP outputs

# Create SHAP directory
os.makedirs(SHAP_DIR, exist_ok=True)

STORE_NAMES = {44: "Store Colombo", 51: "Store Gampaha"}

print("\n Configuration loaded")
print(f"  Results will be saved to: {SHAP_DIR}/")


# STEP 1: LOAD TRAINED XGBOOST MODEL


print("\n")
print("STEP 1: LOADING TRAINED XGBOOST MODEL")
print("\n")

# Load the saved XGBoost model
model_file = f'{MODELS_DIR}/xgboost_model.pkl'

print(f"\n Loading model from: {model_file}")

with open(model_file, 'rb') as f:
    xgb_model = pickle.load(f)

print(f" XGBoost model loaded successfully!")
print(f"  Model type: {type(xgb_model)}")
print(f"  Number of trees: {xgb_model.n_estimators}")
print(f"  Number of features: {xgb_model.n_features_in_}")



# STEP 2: LOAD PROCESSED DATA WITH LAG FEATURES


print("\n")
print("STEP 2: LOADING VALIDATION DATA")
print("\n")

# Load validation data
val = pd.read_csv(f'{PROCESSED_DATA_DIR}/val_processed.csv', parse_dates=['date'])

print(f"\n Validation data loaded: {val.shape[0]:,} rows")

# Recreate lag features (same as XGBoost training)
print("\n Recreating lag features...")

def create_lag_features(df, lag_days=[7, 14, 28]):
    """
    Create same lag features as XGBoost training.
    IMPORTANT: Must match XGBoost feature engineering exactly!
    """
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

val_lagged = create_lag_features(val)
val_lagged = val_lagged.dropna()  # Drop rows with NaN from lag features

print(f" Lag features created")
print(f"  Data after lag features: {val_lagged.shape[0]:,} rows")



# STEP 3: PREPARE FEATURES (SAME AS XGBOOST)

print("\n")
print("STEP 3: PREPARING FEATURES FOR SHAP")
print("\n")

# Define features (MUST match XGBoost training exactly!)
LAG_FEATURES = [
    'sales_lag_7',
    'sales_lag_14',
    'sales_lag_28',
    'sales_rolling_mean_7',
    'sales_rolling_mean_28',
    'sales_rolling_std_7'
]

TEMPORAL_FEATURES = [
    'dayofweek',
    'month',
    'quarter',
    'day',
    'is_weekend',
    'is_month_start',
    'is_month_end'
]

EXTERNAL_FEATURES = [
    'onpromotion',
    'holiday_impact',  # this is the sri lankan holiday impact that i chagd to 
    'family_encoded'
]

ALL_FEATURES = LAG_FEATURES + TEMPORAL_FEATURES + EXTERNAL_FEATURES

print(f"\n Feature list defined:")
print(f"  Total features: {len(ALL_FEATURES)}")

# Extract features
X_val = val_lagged[ALL_FEATURES].copy()
y_val = val_lagged['unit_sales'].copy()

print(f"\n Features prepared:")
print(f"  X_val shape: {X_val.shape}")
print(f"  y_val shape: {y_val.shape}")

# Verify features match model
if X_val.shape[1] != xgb_model.n_features_in_:
    print(f"\n WARNING: Feature count mismatch!")
    print(f"  Model expects: {xgb_model.n_features_in_}")
    print(f"  Data has: {X_val.shape[1]}")
else:
    print(f"\n Feature count matches model: {xgb_model.n_features_in_} features")



# STEP 4: CREATE SHAP EXPLAINER

print("\n")
print("STEP 4: CREATING SHAP EXPLAINER")
print("\n")

print("\n→ Initializing SHAP TreeExplainer...")
print("  (This uses XGBoost's tree structure for fast computation)")

# Create SHAP explainer
explainer = shap.TreeExplainer(xgb_model)

# Extract base value (handle array case)
if isinstance(explainer.expected_value, np.ndarray):
    base_value = explainer.expected_value[0]
else:
    base_value = explainer.expected_value

print(f"✓ SHAP explainer created!")
print(f"  Explainer type: {type(explainer)}")
print(f"  Base value (average prediction): {base_value:.2f}")

print(f"\n→ Note on base value:")
print(f"  This is the average prediction if we had no features")
print(f"  Each feature's SHAP value shows deviation from this baseline")

# STEP 5: CALCULATE SHAP VALUES


print("\n")
print("STEP 5: CALCULATING SHAP VALUES")
print("\n")

print("\n Computing SHAP values for validation set...")
print(f"  Processing {X_val.shape[0]:,} predictions...")
print(f"  This may take 2-5 minutes...")

import time
start_time = time.time()

# Calculate SHAP values
# This explains EVERY prediction in the validation set
shap_values = explainer.shap_values(X_val)

end_time = time.time()
duration = end_time - start_time

print(f"\n SHAP values calculated!")
print(f"  Computation time: {duration:.1f} seconds ({duration/60:.1f} minutes)")
print(f"  SHAP values shape: {shap_values.shape}")
print(f"   {shap_values.shape[0]:,} predictions × {shap_values.shape[1]} features")

# Verify shape
if shap_values.shape[0] != X_val.shape[0]:
    print("\n ERROR: SHAP values don't match data!")
else:
    print(f"\n SHAP values verified: {shap_values.shape[0]:,} explanations generated")


# ============================================================================
# STEP 6: GLOBAL FEATURE IMPORTANCE
# ============================================================================

print("\n" + "="*70)
print("STEP 6: ANALYZING GLOBAL FEATURE IMPORTANCE")
print("="*70)

print("\n→ Computing mean absolute SHAP values...")
print("  (Shows which features are most important overall)")

# Calculate mean absolute SHAP value for each feature
# |SHAP| = importance regardless of direction (positive or negative)
mean_abs_shap = np.abs(shap_values).mean(axis=0)

# Create feature importance dataframe
feature_importance_df = pd.DataFrame({
    'feature': ALL_FEATURES,
    'mean_abs_shap': mean_abs_shap
}).sort_values('mean_abs_shap', ascending=False)

print(f"\n✓ Feature importance calculated!")
print(f"\n📊 TOP 10 MOST IMPORTANT FEATURES (by mean |SHAP|):")
print("="*70)

for idx, row in feature_importance_df.head(10).iterrows():
    bar_length = int(row['mean_abs_shap'] / feature_importance_df['mean_abs_shap'].max() * 50)
    bar = "█" * bar_length
    print(f"  {row['feature']:25s} │ {bar} {row['mean_abs_shap']:.2f}")

print("="*70)

# Insights
top_feature = feature_importance_df.iloc[0]
print(f"\n→ Key Insight:")
print(f"  Most important feature: {top_feature['feature']}")
print(f"  This feature dominates prediction decisions")

if 'sales_lag_7' in feature_importance_df.head(3)['feature'].values:
    print(f"  ✓ Recent sales history (lag_7) is highly influential - expected!")

if 'onpromotion' in feature_importance_df.head(5)['feature'].values:
    print(f"  ✓ Promotions significantly impact predictions")


# ============================================================================
# STEP 7: FEATURE IMPORTANCE COMPARISON (SHAP vs XGBoost)
# ============================================================================

print("\n" + "="*70)
print("STEP 7: COMPARING SHAP vs XGBOOST FEATURE IMPORTANCE")
print("="*70)

print("\n→ Extracting XGBoost's built-in feature importance...")

# Get XGBoost feature importance (gain)
xgb_importance = xgb_model.get_booster().get_score(importance_type='gain')

# Map feature names (f0, f1, f2... → actual names)
feature_name_map = {f'f{i}': ALL_FEATURES[i] for i in range(len(ALL_FEATURES))}

xgb_importance_df = pd.DataFrame({
    'feature': [feature_name_map.get(k, k) for k in xgb_importance.keys()],
    'xgb_gain': list(xgb_importance.values())
})

# Normalize both for comparison (0-100 scale)
feature_importance_df['shap_normalized'] = (
    feature_importance_df['mean_abs_shap'] / 
    feature_importance_df['mean_abs_shap'].max() * 100
)

xgb_importance_df['xgb_normalized'] = (
    xgb_importance_df['xgb_gain'] / 
    xgb_importance_df['xgb_gain'].max() * 100
)

# Merge for comparison
comparison_df = feature_importance_df.merge(
    xgb_importance_df, 
    on='feature', 
    how='left'
).fillna(0)

comparison_df = comparison_df.sort_values('shap_normalized', ascending=False)

print(f"\n📊 SHAP vs XGBOOST IMPORTANCE (Top 10):")
print("="*70)
print(f"{'Feature':<25} {'SHAP Score':<12} {'XGBoost Gain':<15}")
print("-"*70)

for idx, row in comparison_df.head(10).iterrows():
    print(f"{row['feature']:<25} {row['shap_normalized']:>10.1f}  {row['xgb_normalized']:>13.1f}")

print("="*70)

print(f"\n→ Note:")
print(f"  SHAP and XGBoost importance often differ slightly")
print(f"  SHAP = average impact on predictions (model-agnostic)")
print(f"  XGBoost Gain = improvement when feature used for splits (model-specific)")


# ============================================================================
# STEP 8: SAVE FEATURE IMPORTANCE
# ============================================================================

print("\n" + "="*70)
print("STEP 8: SAVING FEATURE IMPORTANCE RESULTS")
print("="*70)

# Save SHAP feature importance
shap_importance_file = f'{SHAP_DIR}/shap_feature_importance.csv'
feature_importance_df.to_csv(shap_importance_file, index=False)
print(f"\n✓ SHAP importance saved to: {shap_importance_file}")

# Save comparison
comparison_file = f'{SHAP_DIR}/shap_vs_xgboost_importance.csv'
comparison_df.to_csv(comparison_file, index=False)
print(f"✓ Comparison saved to: {comparison_file}")


# ============================================================================
# STEP 9: CREATE SUMMARY PLOT (VISUALIZATION)
# ============================================================================

print("\n" + "="*70)
print("STEP 9: CREATING SHAP VISUALIZATIONS")
print("="*70)

print("\n→ Generating SHAP summary plot...")
print("  (Shows feature importance + impact direction)")

# Create summary plot
plt.figure(figsize=(10, 8))

shap.summary_plot(
    shap_values, 
    X_val, 
    feature_names=ALL_FEATURES,
    show=False,  # Don't display, just save
    max_display=10  # Show top 10 features
)

plt.tight_layout()
summary_plot_file = f'{SHAP_DIR}/shap_summary_plot.png'
plt.savefig(summary_plot_file, dpi=150, bbox_inches='tight')
plt.close()

print(f"✓ Summary plot saved to: {summary_plot_file}")
print(f"  This plot shows:")
print(f"  - Feature importance (top to bottom)")
print(f"  - Impact direction (red = increases prediction, blue = decreases)")
print(f"  - Feature value (color intensity)")


# ============================================================================
# STEP 10: CREATE BAR PLOT (SIMPLE IMPORTANCE)
# ============================================================================

print("\n→ Generating feature importance bar chart...")

plt.figure(figsize=(10, 6))

# Top 10 features
top_10 = feature_importance_df.head(10)

plt.barh(range(len(top_10)), top_10['mean_abs_shap'], color='steelblue')
plt.yticks(range(len(top_10)), top_10['feature'])
plt.xlabel('Mean |SHAP value| (average impact on prediction)', fontsize=12)
plt.title('Top 10 Most Important Features (SHAP Analysis)', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()  # Highest at top
plt.grid(axis='x', alpha=0.3)

plt.tight_layout()
bar_plot_file = f'{SHAP_DIR}/shap_importance_bar.png'
plt.savefig(bar_plot_file, dpi=150, bbox_inches='tight')
plt.close()

print(f"✓ Bar chart saved to: {bar_plot_file}")


# ============================================================================
# STEP 11: PER-PRODUCT SHAP ANALYSIS
# ============================================================================

print("\n" + "="*70)
print("STEP 11: ANALYZING SHAP VALUES BY PRODUCT")
print("="*70)

print("\n→ Calculating feature importance for each product...")

# Add product info to SHAP analysis
shap_df = pd.DataFrame(shap_values, columns=ALL_FEATURES)
shap_df['item_nbr'] = val_lagged['item_nbr'].values
shap_df['actual'] = y_val.values
shap_df['prediction'] = xgb_model.predict(X_val)

# Calculate per-product feature importance
product_importance = {}

unique_products = val_lagged['item_nbr'].unique()

for product in unique_products:
    product_mask = shap_df['item_nbr'] == product
    product_shap = shap_values[product_mask]
    
    # Mean absolute SHAP for this product
    product_mean_shap = np.abs(product_shap).mean(axis=0)
    
    # Top 5 features for this product
    product_top_features = pd.DataFrame({
        'feature': ALL_FEATURES,
        'importance': product_mean_shap
    }).sort_values('importance', ascending=False).head(5)
    
    product_importance[product] = product_top_features

print(f"\n✓ Per-product analysis complete!")
print(f"  Analyzed {len(unique_products)} products")

# Show example for one product
example_product = unique_products[0]
print(f"\n📊 Example - Product {example_product} Top 5 Features:")
print(product_importance[example_product].to_string(index=False))

# Save per-product importance
print(f"\n→ Saving per-product SHAP importance...")

for product, importance_df in product_importance.items():
    product_file = f'{SHAP_DIR}/shap_product_{product}_importance.csv'
    importance_df.to_csv(product_file, index=False)

print(f"✓ Saved importance for {len(product_importance)} products to: {SHAP_DIR}/")


# ============================================================================
# STEP 12: EXAMPLE PREDICTION EXPLANATIONS
# ============================================================================

print("\n" + "="*70)
print("STEP 12: CREATING EXAMPLE PREDICTION EXPLANATIONS")
print("="*70)

print("\n→ Selecting interesting predictions to explain...")

# Find diverse examples:
# 1. Best prediction (lowest error)
# 2. Worst prediction (highest error)
# 3. High sales prediction
# 4. Low sales prediction

shap_df['error'] = shap_df['actual'] - shap_df['prediction']
shap_df['abs_error'] = shap_df['error'].abs()

best_pred_idx = shap_df['abs_error'].idxmin()
worst_pred_idx = shap_df['abs_error'].idxmax()
high_sales_idx = shap_df['prediction'].idxmax()
low_sales_idx = shap_df['prediction'].idxmin()

examples = {
    'best_prediction': best_pred_idx,
    'worst_prediction': worst_pred_idx,
    'highest_sales': high_sales_idx,
    'lowest_sales': low_sales_idx
}

print(f"\n✓ Selected {len(examples)} example predictions")

# Create detailed explanations
example_explanations = []

for example_name, idx in examples.items():
    # Get SHAP values for this prediction
    shap_vals_this = shap_values[idx]
    features_this = X_val.iloc[idx]
    
    # Create explanation dataframe
    explanation = pd.DataFrame({
        'feature': ALL_FEATURES,
        'feature_value': features_this.values,
        'shap_value': shap_vals_this
    })
    
    # Sort by absolute SHAP value
    explanation['abs_shap'] = explanation['shap_value'].abs()
    explanation = explanation.sort_values('abs_shap', ascending=False)
    
    # Add prediction details
    actual = shap_df.iloc[idx]['actual']
    prediction = shap_df.iloc[idx]['prediction']
    error = shap_df.iloc[idx]['error']
    
    print(f"\n📋 {example_name.upper().replace('_', ' ')}:")
    print(f"  Actual: {actual:.1f} units")
    print(f"  Predicted: {prediction:.1f} units")
    print(f"  Error: {error:+.1f} units")
    print(f"\n  Top 5 Contributing Features:")
    
    for _, row in explanation.head(5).iterrows():
        direction = "increases" if row['shap_value'] > 0 else "decreases"
        print(f"    • {row['feature']:20s} = {row['feature_value']:6.1f}  →  {direction:9s} prediction by {abs(row['shap_value']):5.1f} units")
    
    # Save this explanation
    explanation_file = f'{SHAP_DIR}/explanation_{example_name}.csv'
    explanation.to_csv(explanation_file, index=False)
    
    example_explanations.append({
        'example': example_name,
        'actual': actual,
        'prediction': prediction,
        'error': error,
        'top_features': explanation.head(5)['feature'].tolist()
    })

# Save examples summary
examples_summary = pd.DataFrame(example_explanations)
examples_file = f'{SHAP_DIR}/example_predictions_summary.csv'
examples_summary.to_csv(examples_file, index=False)

print(f"\n✓ Example explanations saved to: {SHAP_DIR}/")


# ============================================================================
# STEP 13: SAVE COMPLETE SHAP VALUES
# ============================================================================

print("\n" + "="*70)
print("STEP 13: SAVING COMPLETE SHAP VALUES")
print("="*70)

print("\n→ Saving all SHAP values for dashboard integration...")

# Create comprehensive SHAP dataframe
complete_shap_df = pd.DataFrame(shap_values, columns=[f'shap_{feat}' for feat in ALL_FEATURES])

# Add metadata
complete_shap_df['date'] = val_lagged['date'].values
complete_shap_df['store_nbr'] = val_lagged['store_nbr'].values
complete_shap_df['item_nbr'] = val_lagged['item_nbr'].values
complete_shap_df['actual'] = y_val.values
complete_shap_df['prediction'] = xgb_model.predict(X_val)
complete_shap_df['base_value'] = base_value  # Use the extracted value from earlier

# Save
complete_shap_file = f'{SHAP_DIR}/complete_shap_values.csv'
complete_shap_df.to_csv(complete_shap_file, index=False)

print(f"✓ Complete SHAP values saved to: {complete_shap_file}")
print(f"  Size: {complete_shap_df.shape[0]:,} predictions × {complete_shap_df.shape[1]} columns")
print(f"  This file contains SHAP values for every prediction")


# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "="*70)
print("SHAP EXPLAINABILITY - COMPLETE!")
print("="*70)

print(f"\n✓ Summary:")
print(f"  SHAP values calculated: {shap_values.shape[0]:,} predictions")
print(f"  Features analyzed: {shap_values.shape[1]}")
print(f"  Top feature: {feature_importance_df.iloc[0]['feature']}")
print(f"  Products analyzed: {len(unique_products)}")

print(f"\n✓ Files created in {SHAP_DIR}/:")
print(f"  1. shap_feature_importance.csv - Global importance rankings")
print(f"  2. shap_vs_xgboost_importance.csv - SHAP vs XGBoost comparison")
print(f"  3. shap_summary_plot.png - Visual summary")
print(f"  4. shap_importance_bar.png - Bar chart")
print(f"  5. complete_shap_values.csv - All SHAP values for dashboard")
print(f"  6. Per-product importance files ({len(unique_products)} products)")
print(f"  7. Example prediction explanations (4 examples)")

print(f"\n✓ Next steps:")
print(f"  1. Review SHAP visualizations in {SHAP_DIR}/")
print(f"  2. Integrate SHAP into Streamlit dashboard")
print(f"  3. (Optional) Hyperparameter tuning")
print(f"  4. (Optional) Add extensions (replenishment, etc.)")
print(f"  5. Build complete dashboard")

print("="*70 + "\n")

