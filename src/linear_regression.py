import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import warnings
import os
warnings.filterwarnings('ignore')


print("liner regression baseline model in demand forecasting ")


# configurtion


# File paths
PROCESSED_DATA_DIR = '../data/processed'
RESULTS_DIR = '../results'
os.makedirs(RESULTS_DIR, exist_ok=True)

# Selected stores and items (same as data processing)
SELECTED_STORES = [44, 51]
STORE_NAMES = {44: "Store Colombo", 51: "Store Gampaha"}


print("\n Configuration loaded")
print(f"  now Working with: {len(SELECTED_STORES)} stores")
print(f"  now results will be saved to: {RESULTS_DIR}/")


# STEP 1: LOAD PROCESSED DATA

print("data loading processed")


# loading the datasets that has created in data_processing.py
train = pd.read_csv(f'{PROCESSED_DATA_DIR}/train_processed.csv', parse_dates=['date'])
val = pd.read_csv(f'{PROCESSED_DATA_DIR}/val_processed.csv', parse_dates=['date'])
test = pd.read_csv(f'{PROCESSED_DATA_DIR}/test_processed.csv', parse_dates=['date'])

print(f"\n data loaded successfully:")
print(f"   train: {train.shape[0]:,} rows ({train['date'].min()} to {train['date'].max()})")
print(f"  validation: {val.shape[0]:,} rows ({val['date'].min()} to {val['date'].max()})")
print(f"  test: {test.shape[0]:,} rows ({test['date'].min()} to {test['date'].max()})")

# Check what products we have
unique_products = train['item_nbr'].unique()
print(f"\n Found {len(unique_products)} unique products")
print(f"  Product IDs: {sorted(unique_products)}")


# STEP 2: creating lag features


print("STEP 2: creating lag features for time series ")


def create_lag_features(df, lag_days=[7, 14, 28]):
 
    
    print(f"\n Creating lag features for periods: {lag_days}")
    
    df_lagged = df.copy()
    

    for lag in lag_days:
        print(f"  Creating lag_{lag} (sales {lag} days ago)...")
        
       
        df_lagged[f'sales_lag_{lag}'] = df_lagged.groupby(['store_nbr', 'item_nbr'])['unit_sales'].shift(lag)
    

    print(f"  Creating rolling mean features...")
    df_lagged['sales_rolling_mean_7'] = df_lagged.groupby(['store_nbr', 'item_nbr'])['unit_sales'].transform(
        lambda x: x.shift(1).rolling(window=7, min_periods=7).mean()
    )
    
    df_lagged['sales_rolling_mean_28'] = df_lagged.groupby(['store_nbr', 'item_nbr'])['unit_sales'].transform(
        lambda x: x.shift(1).rolling(window=28, min_periods=28).mean()
    )
    

    print(f"  Creating rolling std features...")
    df_lagged['sales_rolling_std_7'] = df_lagged.groupby(['store_nbr', 'item_nbr'])['unit_sales'].transform(
        lambda x: x.shift(1).rolling(window=7, min_periods=7).std()
    )
    
    print(f"\n Created lag features:")
    print(f"  Point lags: sales_lag_7, sales_lag_14, sales_lag_28")
    print(f"   Rolling means: sales_rolling_mean_7, sales_rolling_mean_28")
    print(f" - Rolling std: sales_rolling_std_7")
    print(f"  Total new features: 6")
    
    return df_lagged

# Apply lag feature creation to all datasets
train_lagged = create_lag_features(train)
val_lagged = create_lag_features(val)
test_lagged = create_lag_features(test)

#dropping nan valuse
print("\n Handling missing values from lag features...")
print(f"  Before dropping NaNs:")
print(f"    Train: {train_lagged.shape[0]:,} rows")
print(f"    Val: {val_lagged.shape[0]:,} rows")
print(f"    Test: {test_lagged.shape[0]:,} rows")

train_lagged = train_lagged.dropna()
val_lagged = val_lagged.dropna()
test_lagged = test_lagged.dropna()

print(f"  After dropping NaNs:")
print(f"    train: {train_lagged.shape[0]:,} rows (lost {train.shape[0] - train_lagged.shape[0]:,})")
print(f"    val: {val_lagged.shape[0]:,} rows (lost {val.shape[0] - val_lagged.shape[0]:,})")
print(f"    test: {test_lagged.shape[0]:,} rows (lost {test.shape[0] - test_lagged.shape[0]:,})")


#defining the feature set


print("\n STEP 3: linear regression feature definition")


TARGET = 'unit_sales'


# 1. LAG FEATURES ( in Step 2)
LAG_FEATURES = [
    'sales_lag_7',           # Sales 7 days ago (last week same day)
    'sales_lag_14',          # Sales 14 days ago  
    'sales_lag_28',          # Sales 28 days ago (last month same day)
    'sales_rolling_mean_7',  # 7-day average (recent trend)
    'sales_rolling_mean_28', # 28-day average (longer trend)
    'sales_rolling_std_7'    # 7-day volatility (how stable sales are)
]

# 2. TEMPORAL FEATURES (already in processed data)
TEMPORAL_FEATURES = [
    'dayofweek',      # Monday=0, Sunday=6 (captures weekly patterns)
    'month',          # 1-12 (captures seasonal patterns)
    'quarter',        # 1-4 (captures quarterly patterns)
    'day',            # 1-31 (day of month)
    'is_weekend',     # 0 or 1 (weekend effect)
    'is_month_start', # 0 or 1 (beginning of month effect)
    'is_month_end'    # 0 or 1 (end of month effect)
]

# 3. EXTERNAL FEATURES (already in processed data)
EXTERNAL_FEATURES = [
    'onpromotion',    # 0 or 1 (is product on promotion?)
    'is_holiday',     # 0 or 1 (is it a national holiday?)
    'family_encoded'  # Numeric code for product category
]

# COMBINE ALL FEATURES
ALL_FEATURES = LAG_FEATURES + TEMPORAL_FEATURES + EXTERNAL_FEATURES

print(f"\n Feature groups defined:")
print(f"  lag features: {len(LAG_FEATURES)} features")
print(f"  temporal features {len(TEMPORAL_FEATURES)} features")
print(f"  external features: {len(EXTERNAL_FEATURES)} features")
print(f"  total fe: {len(ALL_FEATURES)}")

print(f"\n Complete feature list:")
for i, feat in enumerate(ALL_FEATURES, 1):
    print(f"  {i:2d}. {feat}")

# Verify all features exist in our data
print(f"\n Verifying features exist in data...")
missing_features = [f for f in ALL_FEATURES if f not in train_lagged.columns]
if missing_features:
    print(f"  warning missing features: {missing_features}")
else:
    print(f"   all {len(ALL_FEATURES)} features found in data")


# STEP 4: PREPARE DATA FOR TRAINING


print("STEP 4: PREPARING DATA FOR TRAINING")


def prepare_train_test(train_df, test_df, features, target):
   
    X_train = train_df[features].copy()
    y_train = train_df[target].copy()
    
    X_test = test_df[features].copy()
    y_test = test_df[target].copy()
    
    return X_train, y_train, X_test, y_test

# Prepare train/validation sets
print("\n Preparing training and validation sets")
X_train, y_train, X_val, y_val = prepare_train_test(
    train_lagged, val_lagged, ALL_FEATURES, TARGET
)

print(f"  Training set:")
print(f"    X_train shape: {X_train.shape} ({X_train.shape[0]:,} samples, {X_train.shape[1]} features)")
print(f"    y_train shape: {y_train.shape} ({y_train.shape[0]:,} samples)")

print(f"  Validation set:")
print(f"    X_val shape: {X_val.shape} ({X_val.shape[0]:,} samples, {X_val.shape[1]} features)")
print(f"    y_val shape: {y_val.shape} ({y_val.shape[0]:,} samples)")



# STEP 5: FEATURE SCALING 


print("STEP 5: SCALING FEATURES")


print("\n Why scale features?")
print("  Linear Regression is sensitive to feature magnitude.")
print("  'sales_lag_28' might be 0-500, but 'dayofweek' is 0-6.")
print("  Without scaling, the model treats large numbers as more important!")
print("  StandardScaler transforms all features to mean=0, std=1")

# Create scaler
scaler = StandardScaler()

# FIT scaler on training data ONLY (avoid data leakage)
#  If we fit on validation data, we're using future info
print("\n Fitting scaler on training data...")
X_train_scaled = scaler.fit_transform(X_train)
print(f"   Training data scaled")

# TRANSFORM validation data using training scaler
print(" Transforming validation data with training scaler...")
X_val_scaled = scaler.transform(X_val)
print(f"   Validation data scaled")

print(f"\n Scaling verification:")
print(f"  Before scaling - X_train mean: {X_train.mean().mean():.2f}, std: {X_train.std().mean():.2f}")
print(f"  After scaling - X_train mean: {X_train_scaled.mean():.4f}, std: {X_train_scaled.std():.2f}")
print(f"  (Should be close to mean=0, std=1)")



# STEP 6: TRAIN LINEAR REGRESSION MODEL



print("STEP 6: TRAINING LINEAR REGRESSION MODEL")


print("\n Initializing Linear Regression model...")
print("  Model: Ordinary Least Squares (OLS) regression")
print("  Optimization: Minimizes sum of squared errors")

# Create model
model = LinearRegression()

# Train model
print("\n→ Training on full training set...")
print(f"  Fitting model with {X_train_scaled.shape[0]:,} samples, {X_train_scaled.shape[1]} features...")

model.fit(X_train_scaled, y_train)

print(f"   Model trained successfully!")

# Show learned coefficients (how important each feature is)
print(f"\n Model learned {len(model.coef_)} coefficients:")
print(f"  Intercept (baseline): {model.intercept_:.2f}")
print(f"\n  Top 10 most influential features (by absolute coefficient):")

# Create feature importance dataframe
feature_importance = pd.DataFrame({
    'feature': ALL_FEATURES,
    'coefficient': model.coef_
})
feature_importance['abs_coefficient'] = feature_importance['coefficient'].abs()
feature_importance = feature_importance.sort_values('abs_coefficient', ascending=False)

for idx, row in feature_importance.head(10).iterrows():
    print(f"    {row['feature']:25s} → {row['coefficient']:7.3f}")

print("\n   interpretation:")
print("    Positive coefficient = higher feature value → higher sales")
print("    Negative coefficient = higher feature value → lower sales")
print("    Larger absolute value = stronger influence")



print("STEP 7: making predictions and evaluating model")


# Predict on validation set
print("\n Generating predictions on validation set...")
y_val_pred = model.predict(X_val_scaled)
print(f"  Generated {len(y_val_pred):,} predictions")

# Calculate metrics
print("\n Calculating performance metrics...")

rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
mae = mean_absolute_error(y_val, y_val_pred)
r2 = r2_score(y_val, y_val_pred)

# MAPE (Mean Absolute Percentage Error) - useful for business interpretation
# Tells us "on average, predictions are off by X%"
mape = np.mean(np.abs((y_val - y_val_pred) / y_val)) * 100


print("VALIDATION SET PERFORMANCE")

print(f"  RMSE (Root Mean Squared Error): {rmse:.2f} units")
print(f"  MAE  (Mean Absolute Error):     {mae:.2f} units")
print(f"  R squared  :               {r2:.4f} ({r2*100:.2f}%)")
print(f"  MAPE (Mean Abs Percentage Err): {mape:.2f}%")
print("\n")

print("\n Interpretation:")
print(f"   On average, predictions are off by ±{mae:.1f} units")
print(f"   Model explains {r2*100:.1f}% of variance in sales")
print(f"   Typical prediction error is {mape:.1f} percent of actual sales")

# Check for potential issues
if r2 < 0:
    print("\n   WARNING: Negative r squared - model performs worse than predicting the mean!")
elif r2 < 0.3:
    print("\n   WARNING: Low  r squared- model has weak predictive power")
elif r2 > 0.95:
    print("\n   WARNING: Very high r squared - check for data leakage")
else:
    print(f"\n  r squared  looks reasonable for a baseline model")


# STEP 8: ANALYZE PREDICTIONS BY STORE AND PRODUCT

print("STEP 8: ANALYZING PREDICTIONS BY STORE & PRODUCT")


# Add predictions to validation dataframe for analysis
val_results = val_lagged.copy()
val_results['predictions'] = y_val_pred
val_results['actual'] = y_val.values
val_results['error'] = val_results['actual'] - val_results['predictions']
val_results['abs_error'] = val_results['error'].abs()
val_results['pct_error'] = (val_results['abs_error'] / val_results['actual']) * 100

# Analyze by product
print("\n Performance by Product:")
product_performance = val_results.groupby('item_nbr').agg({
    'actual': 'mean',
    'predictions': 'mean',
    'abs_error': 'mean',
    'pct_error': 'mean'
}).round(2)
product_performance.columns = ['Avg Actual Sales', 'Avg Predicted Sales', 'MAE', 'MAPE (%)']
product_performance = product_performance.sort_values('MAE', ascending=False)

print(product_performance)

# Analyze by store
print("\n Performance by Store:")
store_performance = val_results.groupby('store_nbr').agg({
    'actual': 'mean',
    'predictions': 'mean',
    'abs_error': 'mean',
    'pct_error': 'mean'
}).round(2)
store_performance.columns = ['Avg Actual Sales', 'Avg Predicted Sales', 'MAE', 'MAPE (%)']
store_performance.index = store_performance.index.map(STORE_NAMES)

print(store_performance)



# STEP 9: SAVE RESULTS


print("\n")
print("STEP 9: SAVING RESULTS")
print("\n")

# Save overall metrics
metrics_df = pd.DataFrame({
    'Model': ['Linear Regression'],
    'RMSE': [rmse],
    'MAE': [mae],
    'R2': [r2],
    'MAPE': [mape]
})

metrics_file = f'{RESULTS_DIR}/linear_regression_metrics.csv'
metrics_df.to_csv(metrics_file, index=False)
print(f"\n Overall metrics saved to: {metrics_file}")

# Save feature importance
importance_file = f'{RESULTS_DIR}/linear_regression_feature_importance.csv'
feature_importance.to_csv(importance_file, index=False)
print(f" feature importance saved to {importance_file}")

# Save detailed predictions (for later analysis/dashboard)
predictions_file = f'{RESULTS_DIR}/linear_regression_predictions.csv'
val_results[['date', 'store_nbr', 'item_nbr', 'actual', 'predictions', 'error']].to_csv(
    predictions_file, index=False
)
print(f" Detailed predictions saved to: {predictions_file}")

print("\n")
print("LINEAR REGRESSION BASELINE - COMPLETE!")
print("\n")
print(f"\nSummary:")
print(f"  model trained on {X_train.shape[0]:,} samples")
print(f"  validated on {X_val.shape[0]:,} samples")
print(f"  validation RMSE: {rmse:.2f}")
print(f"  validation MAE: {mae:.2f}")
print(f"  validation R squared: {r2:.4f}")
print(f"  results saved to: {RESULTS_DIR}/")





