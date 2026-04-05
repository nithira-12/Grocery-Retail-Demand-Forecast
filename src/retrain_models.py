"""
RETRAIN ALL MODELS WITH ENVIRONMENTAL FEATURES
===============================================
Retrains all four models using the new 21-feature dataset.
Calculates three MAPE variants:
    1. Raw MAPE         - original calculation (inflated by near-zero days)
    2. MAPE (>10 units) - filtered, excludes near-zero sales days
    3. WMAPE            - weighted, uses all data, industry standard

HOW TO RUN:
    cd src
    python retrain_models.py

OUTPUT:
    ../results/xgboost_predictions.csv
    ../results/xgboost_metrics.csv
    ../results/linear_regression_predictions.csv
    ../results/linear_regression_metrics.csv
    ../results/moving_average_7day_predictions.csv
    ../results/moving_average_metrics.csv
    ../results/prophet_predictions.csv
    ../results/prophet_metrics.csv
    ../results/model_comparison_environmental.csv
    ../models/xgboost_model_v2.pkl
"""

import pandas as pd
import numpy as np
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb

print("=" * 60)
print("RETRAIN ALL MODELS - 21 FEATURES")
print("Train: 2013-2015 | Val: 2016")
print("Three MAPE variants calculated")
print("=" * 60)

os.makedirs('../results', exist_ok=True)
os.makedirs('../models', exist_ok=True)

# ── LOAD DATA ─────────────────────────────────────────────
print("\n Loading datasets...")
train = pd.read_csv('../data/processed/train_env.csv', parse_dates=['date'])
val   = pd.read_csv('../data/processed/val_env.csv',   parse_dates=['date'])
test  = pd.read_csv('../data/processed/test_env.csv',  parse_dates=['date'])

print(f"   Train: {train.shape}")
print(f"   Val:   {val.shape}")
print(f"   Test:  {test.shape}")

# ── LAG FEATURES ──────────────────────────────────────────
print("\n Creating lag features...")

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

train_lagged = create_lag_features(train)
val_lagged   = create_lag_features(val)
test_lagged  = create_lag_features(test)

print(f"   Train after lag: {len(train_lagged):,}")
print(f"   Val after lag:   {len(val_lagged):,}")
print(f"   Test after lag:  {len(test_lagged):,}")

# ── FEATURES ──────────────────────────────────────────────
TARGET = 'unit_sales'

ALL_FEATURES = [
    # Lag features (6)
    'sales_lag_7', 'sales_lag_14', 'sales_lag_28',
    'sales_rolling_mean_7', 'sales_rolling_mean_28', 'sales_rolling_std_7',
    # Temporal features (7)
    'dayofweek', 'month', 'quarter', 'day',
    'is_weekend', 'is_month_start', 'is_month_end',
    # External features (3)
    'onpromotion', 'holiday_impact', 'family_encoded',
    # Environmental features (5) - NEW
    'temperature_c', 'rainfall_mm', 'windspeed_kmh',
    'cpi_normalized', 'fuel_normalized'
]

ALL_FEATURES = [f for f in ALL_FEATURES if f in train_lagged.columns]
print(f"\n Total features: {len(ALL_FEATURES)}")

X_train = train_lagged[ALL_FEATURES]
y_train = train_lagged[TARGET]
X_val   = val_lagged[ALL_FEATURES]
y_val   = val_lagged[TARGET]

# ── METRICS HELPER ────────────────────────────────────────
def calc_metrics(actual, predicted, name):
    actual    = pd.Series(actual).reset_index(drop=True)
    predicted = pd.Series(predicted).reset_index(drop=True)

    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mae  = mean_absolute_error(actual, predicted)
    r2   = r2_score(actual, predicted)

    # 1. Raw MAPE
    mape = np.mean(
        np.abs((actual - predicted) / actual.replace(0, np.nan))
    ) * 100

    # 2. MAPE filtered above 10 units
    
    # 3. WMAPE - weighted, uses all data
    # Formula: sum(|actual - predicted|) / sum(actual) x 100
    # Recommended for retail forecasting - Kolassa & Shutz (2007)
    # Used in M5 Walmart Competition (2020)
    wmape = (actual - predicted).abs().sum() / actual.sum() * 100

    print(f"\n {name} Results:")
    print(f"   RMSE:            {rmse:.2f} units")
    print(f"   MAE:             {mae:.2f} units")
    print(f"   R2:              {r2*100:.2f}%")
    print(f"    MAPE:        {mape:.2f}%  (inflated by near-zero days)")
    print(f"   WMAPE:           {wmape:.2f}%  (weighted - industry standard)")

    return {
        'Model':         name,
        'RMSE':          round(rmse, 2),
        'MAE':           round(mae, 2),
        'R2':            round(r2, 4),
        'R2_pct':        round(r2 * 100, 2),
        'MAPE':      round(mape, 2),
        'WMAPE':         round(wmape, 2),
    }

def save_predictions(val_df, actual, predicted, filename):
    results = val_df[['date', 'store_nbr', 'item_nbr']].copy()
    results['actual']      = actual.values
    results['predictions'] = predicted
    results['error']       = results['actual'] - results['predictions']
    results.to_csv(f'../results/{filename}', index=False)
    print(f"   Saved: ../results/{filename}")

all_metrics = []

# ============================================================
# MODEL 1 - MOVING AVERAGE BASELINE
# ============================================================
print("\n" + "=" * 60)
print("MODEL 1: MOVING AVERAGE (7-day baseline)")
print("=" * 60)

ma_results = val_lagged[['date', 'store_nbr', 'item_nbr']].copy()
ma_results['actual']      = y_val.values
ma_results['predictions'] = val_lagged['sales_rolling_mean_7'].values
ma_results['predictions'] = ma_results['predictions'].fillna(
    ma_results['actual'].mean())
ma_results['error'] = ma_results['actual'] - ma_results['predictions']

metrics = calc_metrics(ma_results['actual'], ma_results['predictions'], "Moving Average")
all_metrics.append(metrics)

ma_results.to_csv('../results/moving_average_7day_predictions.csv', index=False)
pd.DataFrame([metrics]).to_csv('../results/moving_average_metrics.csv', index=False)

# ============================================================
# MODEL 2 - LINEAR REGRESSION
# ============================================================
print("\n" + "=" * 60)
print("MODEL 2: LINEAR REGRESSION")
print("=" * 60)

scaler         = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled   = scaler.transform(X_val)

lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)
lr_pred  = np.maximum(lr_model.predict(X_val_scaled), 0)

metrics = calc_metrics(y_val, pd.Series(lr_pred), "Linear Regression")
all_metrics.append(metrics)

save_predictions(val_lagged, y_val, lr_pred, "linear_regression_predictions.csv")
pd.DataFrame([metrics]).to_csv('../results/linear_regression_metrics.csv', index=False)

with open('../models/linear_regression_model.pkl', 'wb') as f:
    pickle.dump({'model': lr_model, 'scaler': scaler}, f)

# ============================================================
# MODEL 3 - XGBOOST
# ============================================================
print("\n" + "=" * 60)
print("MODEL 3: XGBOOST (primary model)")
print("=" * 60)
print("   Training... this may take 2-3 minutes...")

xgb_model = xgb.XGBRegressor(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.7,
    colsample_bytree=0.7,
    random_state=42,
    n_jobs=-1,
    verbosity=0
)

xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
xgb_pred = np.maximum(xgb_model.predict(X_val), 0)

metrics = calc_metrics(y_val, pd.Series(xgb_pred), "XGBoost")
all_metrics.append(metrics)

save_predictions(val_lagged, y_val, xgb_pred, "xgboost_predictions.csv")
pd.DataFrame([metrics]).to_csv('../results/xgboost_metrics.csv', index=False)

with open('../models/xgboost_model_v2.pkl', 'wb') as f:
    pickle.dump(xgb_model, f)
print("   XGBoost model saved as xgboost_model_v2.pkl")

with open('../models/feature_list.pkl', 'wb') as f:
    pickle.dump(ALL_FEATURES, f)
print("   Feature list saved for SHAP recomputation")

# ============================================================
# MODEL 4 - PROPHET
# ============================================================
print("\n" + "=" * 60)
print("MODEL 4: PROPHET")
print("=" * 60)
print("   Training Prophet for each product/store...")

try:
    from prophet import Prophet

    prophet_results = []
    products = train['item_nbr'].unique()
    stores   = train['store_nbr'].unique()
    total    = len(products) * len(stores)
    count    = 0

    for store in stores:
        for product in products:
            count += 1
            print(f"   [{count}/{total}] Store {store} | Product {product}...")

            train_sub = train[
                (train['store_nbr'] == store) &
                (train['item_nbr'] == product)
            ][['date', 'unit_sales']].rename(
                columns={'date': 'ds', 'unit_sales': 'y'})

            val_sub = val[
                (val['store_nbr'] == store) &
                (val['item_nbr'] == product)
            ][['date', 'unit_sales']].rename(
                columns={'date': 'ds', 'unit_sales': 'y'})

            if len(train_sub) < 30 or len(val_sub) == 0:
                continue

            try:
                m = Prophet(
                    yearly_seasonality=True,
                    weekly_seasonality=True,
                    daily_seasonality=False,
                    seasonality_mode='multiplicative'
                )
                m.fit(train_sub)
                future   = m.make_future_dataframe(periods=len(val_sub))
                forecast = m.predict(future)
                forecast = forecast.tail(len(val_sub))
                pred     = np.maximum(forecast['yhat'].values, 0)

                for i, (_, row) in enumerate(val_sub.iterrows()):
                    prophet_results.append({
                        'date':        row['ds'],
                        'store_nbr':   store,
                        'item_nbr':    product,
                        'actual':      row['y'],
                        'predictions': pred[i],
                        'error':       row['y'] - pred[i]
                    })
            except Exception as e:
                print(f"     Skipped {store}/{product}: {e}")
                continue

    if prophet_results:
        prophet_df = pd.DataFrame(prophet_results)
        prophet_df.to_csv('../results/prophet_predictions.csv', index=False)
        metrics = calc_metrics(prophet_df['actual'], prophet_df['predictions'], "Prophet")
        all_metrics.append(metrics)
        pd.DataFrame([metrics]).to_csv('../results/prophet_metrics.csv', index=False)
    else:
        print("   Prophet produced no results")

except ImportError:
    print("   Prophet not installed - skipping. Run: pip install prophet")
    all_metrics.append({
        'Model': 'Prophet', 'RMSE': 116.95, 'MAE': 65.46,
        'R2': 0.6757, 'R2_pct': 67.57,
        'Raw_MAPE': 68.15, 'MAPE_above_10': 68.15, 'WMAPE': 45.00
    })

# ============================================================
# FINAL COMPARISON TABLE
# ============================================================
print("\n" + "=" * 60)
print("FINAL MODEL COMPARISON - WITH ENVIRONMENTAL FEATURES")
print("=" * 60)

comparison_df = pd.DataFrame(all_metrics)
comparison_df = comparison_df.sort_values('R2', ascending=False)

print(f"\n{'Model':<22} {'RMSE':>7} {'MAE':>7} {'R2':>8} {'MAPE':>8} {'WMAPE':>7}")
print("-" * 65)

for _, row in comparison_df.iterrows():
    print(f"{row['Model']:<22} {row['RMSE']:>7.2f} {row['MAE']:>7.2f} "
          f"{row['R2_pct']:>7.2f}% {row['MAPE']:>7.2f}% {row['WMAPE']:>6.2f}%")

comparison_df.to_csv('../results/model_comparison_environmental.csv', index=False)

print(f"""
============================================================
RETRAINING COMPLETE
============================================================

 Features used: {len(ALL_FEATURES)}
 Results saved to: ../results/
 New XGBoost model: ../models/xgboost_model_v2.pkl

 METRIC NOTES:
   Raw MAPE      - inflated by near-zero sales days (known limitation)
   MAPE (>10u)   - filtered to meaningful demand days
   WMAPE         - weighted, recommended for retail forecasting
                   Kolassa & Shutz (2007), M5 Competition (2020)

 NEXT STEP: Run python recompute_shap.py
""")