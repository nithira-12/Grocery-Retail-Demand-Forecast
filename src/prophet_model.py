import pandas as pd
import numpy as np
import prophet
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
import os
import json
warnings.filterwarnings('ignore')

print("\n" )
print("PROPHET MODEL ")
print("\n" )

# CONFIGURATION


PROCESSED_DATA_DIR = '../data/processed'
RESULTS_DIR = '../results'
MODELS_DIR = '../models'  # Will save trained Prophet models here

# Create directories
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

SELECTED_STORES = [44, 51]
STORE_NAMES = {44: "Store Colombo", 51: "Store Gampaha"}

print("\n Configuration loaded")
print(f"  Working with: {len(SELECTED_STORES)} stores")
print(f"  Results will be saved to: {RESULTS_DIR}/")
print(f"  Models will be saved to: {MODELS_DIR}/")

# STEP 1: LOAD PROCESSED DATA

print("\n")
print("STEP 1: LOADING PROCESSED DATA")
print("\n" )

# Load datasets
train = pd.read_csv(f'{PROCESSED_DATA_DIR}/train_processed.csv', parse_dates=['date'])
val = pd.read_csv(f'{PROCESSED_DATA_DIR}/val_processed.csv', parse_dates=['date'])
test = pd.read_csv(f'{PROCESSED_DATA_DIR}/test_processed.csv', parse_dates=['date'])

print(f"\n Data loaded successfully:")
print(f"  train: {train.shape[0]:,} rows ({train['date'].min()} to {train['date'].max()})")
print(f"  validation: {val.shape[0]:,} rows ({val['date'].min()} to {val['date'].max()})")
print(f"  test: {test.shape[0]:,} rows ({test['date'].min()} to {test['date'].max()})")

# Get unique store-product combinations
unique_combinations = train.groupby(['store_nbr', 'item_nbr']).size().reset_index(name='count')
print(f"\n found {len(unique_combinations)} unique store-product combinations")
print(f"  (2 stores × {train['item_nbr'].nunique()} products = {len(unique_combinations)} models to train)")


# STEP 2: PREPARE DATA FOR PROPHET


print("\n")
print("STEP 2: PREPARING DATA FOR PROPHET FORMAT")
print("\n" )

def prepare_prophet_data(df, store, item):
    """
    Convert our data to Prophet's required format.
    
    PROPHET REQUIRES:
    - Column 'ds' (datestamp) → dates
    - Column 'y' (target) → values to predict
    
    WHY THIS FORMAT?
    Prophet is opinionated - it wants simple date + value pairs.
    We can add extra regressors (promotions, holidays) separately.
    """
    
    # Filter to specific store-product
    df_filtered = df[
        (df['store_nbr'] == store) & 
        (df['item_nbr'] == item)
    ].copy()
    
    # Sort by date (Prophet needs chronological order)
    df_filtered = df_filtered.sort_values('date')
    
    # Rename columns to Prophet format
    prophet_df = pd.DataFrame({
        'ds': df_filtered['date'],  # ds = datestamp
        'y': df_filtered['unit_sales']  # y = target variable
    })
    
    # Add extra regressors (features Prophet can use)
    # IMPORTANT: These are optional but can improve accuracy
    prophet_df['onpromotion'] = df_filtered['onpromotion'].values
    prophet_df['is_holiday'] = df_filtered['is_holiday'].values
    
    # Reset index
    prophet_df = prophet_df.reset_index(drop=True)
    
    return prophet_df

# Example: Show what Prophet data looks like
print("\n Example of Prophet-formatted data:")
print("  (Store 44, Item 265559)")
example_data = prepare_prophet_data(train, 44, 265559)
print(example_data.head(10))
print(f"\n  total rows: {len(example_data)}")
print(f"  columns: {list(example_data.columns)}")

# ============================================================================
# STEP 3: TRAIN PROPHET MODELS
# ============================================================================

print("\n")
print("STEP 3: TRAINING PROPHET MODELS")
print("\n" )

print("\n Prophet will train one model per store-product combination")
print("  This captures unique patterns for each product in each store")
print("  (e.g., Bread in Colombo may have different patterns than in Gampaha)")

def train_prophet_model(train_df, store, item, product_name="Product"):
    """
    Train a Prophet model for a specific store-product combination.
    
    PROPHET COMPONENTS:
    - Trend: Overall up/down movement over time
    - Yearly seasonality: Annual patterns (holidays, seasons)
    - Weekly seasonality: Day-of-week patterns (weekends vs weekdays)
    - Holidays: Special events that affect sales
    
    HYPERPARAMETERS:
    - changepoint_prior_scale: How flexible the trend is (default 0.05)
    - seasonality_prior_scale: How much emphasis on seasonality (default 10)
    - We're using defaults - could tune these later for better performance
    """
    
    # Prepare data in Prophet format
    prophet_data = prepare_prophet_data(train_df, store, item)
    
    # Initialize Prophet model
    model = Prophet(
        yearly_seasonality=True,  # Capture annual patterns
        weekly_seasonality=True,  # Capture weekly patterns (weekends, etc)
        daily_seasonality=False,  # Don't need daily (we predict weekly/monthly)
        changepoint_prior_scale=0.05,  # Default trend flexibility
        seasonality_mode='multiplicative'  # Sales scale with trend (not additive)
    )
    
    # Add extra regressors (features)
    # WHY? Promotions and holidays significantly affect retail sales
    model.add_regressor('onpromotion')
    model.add_regressor('is_holiday')
    
    # Train model
    # Prophet uses Stan (Bayesian inference) under the hood - can be slow!
    model.fit(prophet_data)
    
    return model

# Train models for all store-product combinations
print(f"\n Training {len(unique_combinations)} Prophet models...")


trained_models = {}
model_count = 0

for idx, row in unique_combinations.iterrows():
    store = row['store_nbr']
    item = row['item_nbr']
    
    model_count += 1
    store_name = STORE_NAMES.get(store, f"Store {store}")
    
    print(f"  [{model_count}/{len(unique_combinations)}] Training: {store_name}, Item {item}...", end='')
    
    try:
        # Train model
        model = train_prophet_model(train, store, item)
        
        # Store model with unique key
        model_key = f"store_{store}_item_{item}"
        trained_models[model_key] = model
        
        print(" okay")
        
    except Exception as e:
        print(f"  FAILED: {str(e)}")
        continue

print(f"\n have trained {len(trained_models)}/{len(unique_combinations)} models")

if len(trained_models) < len(unique_combinations):
    print(f" warning: {len(unique_combinations) - len(trained_models)} models failed to train")


# STEP 4: MAKE PREDICTIONS ON VALIDATION SET


print("\n")
print("STEP 4: GENERATING PREDICTIONS ON VALIDATION SET")
print("\n")

def make_prophet_predictions(model, val_df, store, item):
    """
    Use trained Prophet model to predict validation set.
    
    PROPHET PREDICTION PROCESS:
    1. Create future dataframe with dates to predict
    2. Add regressor values (promotion, holiday) for those dates
    3. Call model.predict()
    4. Extract 'yhat' column (predicted values)
    
    WHY THIS WAY?
    Prophet needs to know the regressor values for future dates.
    We have actual promotion/holiday data for validation period.
    """
    
    # Prepare validation data in Prophet format
    val_prophet = prepare_prophet_data(val_df, store, item)
    
    # Prophet predicts using the validation dates
    forecast = model.predict(val_prophet)
    
    # Extract predictions and actuals
    predictions = forecast['yhat'].values  # yhat = predicted y
    actuals = val_prophet['y'].values
    
    return predictions, actuals, val_prophet['ds'].values

# Generate predictions for all models
print(f"\n Generating predictions for {len(trained_models)} models...")

all_predictions = []
model_count = 0

for model_key, model in trained_models.items():
    # Extract store and item from key
    # Format: "store_44_item_265559"
    parts = model_key.split('_')
    store = int(parts[1])
    item = int(parts[3])
    
    model_count += 1
    store_name = STORE_NAMES.get(store, f"Store {store}")
    
    print(f"  [{model_count}/{len(trained_models)}] Predicting: {store_name}, Item {item}...", end='')
    
    try:
        # Make predictions
        predictions, actuals, dates = make_prophet_predictions(model, val, store, item)
        
        # Store results
        for i in range(len(predictions)):
            all_predictions.append({
                'date': dates[i],
                'store_nbr': store,
                'item_nbr': item,
                'actual': actuals[i],
                'predicted': predictions[i],
                'error': actuals[i] - predictions[i]
            })
        
        print(" okay")
        
    except Exception as e:
        print(f"  FAILED: {str(e)}")
        continue

# Convert to DataFrame
predictions_df = pd.DataFrame(all_predictions)
print(f"\n Generated {len(predictions_df):,} predictions across all products")


# STEP 5: EVALUATE MODEL PERFORMANCE


print("\n")
print("STEP 5: EVALUATING PROPHET MODEL PERFORMANCE")
print("\n")

# Overall metrics
y_true = predictions_df['actual'].values
y_pred = predictions_df['predicted'].values

rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)
mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

print("\n" )
print("\n" )
print("VALIDATION SET PERFORMANCE")
print("\n" )

print(f"  RMSE (Root Mean Squared Error): {rmse:.2f} units")
print(f"  MAE  (Mean Absolute Error):     {mae:.2f} units")
print(f"  R squared:                      {r2:.4f} ({r2*100:.2f}%)")
print(f"  MAPE (Mean Abs Percentage Err): {mape:.2f}%")
print("\n" )
print("\n" )

print("\n interpretation:")
print(f"  on average, predictions are off by ±{mae:.1f} units")
print(f"   model explains {r2*100:.1f}% of variance in sales")
print(f"  typical prediction error is ~{mape:.1f}% of actual sales")

# Check performance
if r2 < 0:
    print("\n warning: Negative R squared - model performs worse than predicting the mean!")
elif r2 < 0.3:
    print("\n warning: Low R squared - model has weak predictive power")
elif r2 > 0.95:
    print("\n   warn: Very high R sq - check for data leakage")
else:
    print(f"\n  r squared looks reasonable for this")


# STEP 6: ANALYZE BY PRODUCT & STORE


print("\n")
print("STEP 6: ANALYZING PREDICTIONS BY STORE & PRODUCT")
print("\n")

# Add absolute error and percentage error
predictions_df['abs_error'] = predictions_df['error'].abs()
predictions_df['pct_error'] = (predictions_df['abs_error'] / predictions_df['actual']) * 100

# By product
print("\n performance by Product:")
product_perf = predictions_df.groupby('item_nbr').agg({
    'actual': 'mean',
    'predicted': 'mean',
    'abs_error': 'mean',
    'pct_error': 'mean'
}).round(2)
product_perf.columns = ['Avg Actual Sales', 'Avg Predicted Sales', 'MAE', 'MAPE (%)']
product_perf = product_perf.sort_values('MAE', ascending=False)
print(product_perf)

# By store
print("\n performance by Store:")
store_perf = predictions_df.groupby('store_nbr').agg({
    'actual': 'mean',
    'predicted': 'mean',
    'abs_error': 'mean',
    'pct_error': 'mean'
}).round(2)
store_perf.columns = ['Avg Actual Sales', 'Avg Predicted Sales', 'MAE', 'MAPE (%)']
store_perf.index = store_perf.index.map(STORE_NAMES)
print(store_perf)

# Find best and worst performing products
best_product = product_perf.nsmallest(3, 'MAE')
worst_product = product_perf.nlargest(3, 'MAE')

print("\n best Performing Products (Lowest MAE):")
print(best_product[['MAE', 'MAPE (%)']])

print("\n most Challenging Products (Highest MAE):")
print(worst_product[['MAE', 'MAPE (%)']])

# ============================================================================
# STEP 7: SAVE RESULTS & MODELS
# ============================================================================

print("\n")
print("STEP 7: SAVING RESULTS & MODELS")
print("\n")

# Save overall metrics
metrics_df = pd.DataFrame({
    'Model': ['Prophet'],
    'RMSE': [rmse],
    'MAE': [mae],
    'R2': [r2],
    'MAPE': [mape]
})

metrics_file = f'{RESULTS_DIR}/prophet_metrics.csv'
metrics_df.to_csv(metrics_file, index=False)
print(f"\n overall metrics saved to: {metrics_file}")

# Save detailed predictions
predictions_file = f'{RESULTS_DIR}/prophet_predictions.csv'
predictions_df.to_csv(predictions_file, index=False)
print(f" detailed predictions saved to: {predictions_file}")

# Save product-level performance
product_perf_file = f'{RESULTS_DIR}/prophet_product_performance.csv'
product_perf.to_csv(product_perf_file)
print(f" product performance saved to: {product_perf_file}")

# Save store-level performance
store_perf_file = f'{RESULTS_DIR}/prophet_store_performance.csv'
store_perf.to_csv(store_perf_file)
print(f" store performance saved to: {store_perf_file}")

# Save trained models (optional - they're large files!)
# Uncomment if you want to save models for later use
# import pickle
# for model_key, model in trained_models.items():
#     model_file = f'{MODELS_DIR}/{model_key}_prophet.pkl'
#     with open(model_file, 'wb') as f:
#         pickle.dump(model, f)
# print(f"\n✓ Saved {len(trained_models)} trained models to: {MODELS_DIR}/")

print("\n")
print("PROPHET MODEL COMPLETE!")
print("\n")

print(f"\nsummary:")
print(f"  models trained: {len(trained_models)}")
print(f"  predictions generated: {len(predictions_df):,}")
print(f"  validation RMSE: {rmse:.2f}")
print(f"  validation MAE: {mae:.2f}")
print(f"  validation R squared: {r2:.4f}")
print(f"  results saved to: {RESULTS_DIR}/")




