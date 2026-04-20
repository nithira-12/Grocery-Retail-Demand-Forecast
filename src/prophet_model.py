import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
import os
warnings.filterwarnings('ignore')

PROCESSED_DATA_DIR = '../data/processed'
RESULTS_DIR        = '../results'
MODELS_DIR         = '../models'

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

SELECTED_STORES = [44, 51]
STORE_NAMES     = {44: "Store Colombo", 51: "Store Gampaha"}

train = pd.read_csv(f'{PROCESSED_DATA_DIR}/train_processed.csv', parse_dates=['date'])
val   = pd.read_csv(f'{PROCESSED_DATA_DIR}/val_processed.csv',   parse_dates=['date'])
test  = pd.read_csv(f'{PROCESSED_DATA_DIR}/test_processed.csv',  parse_dates=['date'])

unique_combinations = train.groupby(['store_nbr', 'item_nbr']).size().reset_index(name='count')


def prepare_prophet_data(df, store, item):
    df_filtered = df[
        (df['store_nbr'] == store) &
        (df['item_nbr']  == item)
    ].copy().sort_values('date')

    prophet_df = pd.DataFrame({
        'ds': df_filtered['date'],
        'y':  df_filtered['unit_sales']
    })
    prophet_df['onpromotion'] = df_filtered['onpromotion'].values
    prophet_df['is_holiday']  = df_filtered['is_holiday'].values

    return prophet_df.reset_index(drop=True)


def train_prophet_model(train_df, store, item):
    prophet_data = prepare_prophet_data(train_df, store, item)

    # multiplicative seasonality so sales scale with trend rather than adding to it
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        changepoint_prior_scale=0.05,
        seasonality_mode='multiplicative'
    )
    model.add_regressor('onpromotion')
    model.add_regressor('is_holiday')
    model.fit(prophet_data)

    return model


def make_prophet_predictions(model, val_df, store, item):
    val_prophet = prepare_prophet_data(val_df, store, item)
    forecast    = model.predict(val_prophet)
    predictions = forecast['yhat'].values
    actuals     = val_prophet['y'].values
    return predictions, actuals, val_prophet['ds'].values


# train one model per store-product combination
print(f"training {len(unique_combinations)} prophet models...")

trained_models = {}
model_count    = 0

for idx, row in unique_combinations.iterrows():
    store = row['store_nbr']
    item  = row['item_nbr']
    model_count += 1

    print(f"  [{model_count}/{len(unique_combinations)}] {STORE_NAMES.get(store)} item {item}...", end='')

    try:
        model     = train_prophet_model(train, store, item)
        model_key = f"store_{store}_item_{item}"
        trained_models[model_key] = model
        print(" done")
    except Exception as e:
        print(f" failed: {e}")
        continue

if len(trained_models) < len(unique_combinations):
    print(f"warning: {len(unique_combinations) - len(trained_models)} models failed to train")

# generate validation predictions
print(f"\ngenerating predictions for {len(trained_models)} models...")

all_predictions = []
model_count     = 0

for model_key, model in trained_models.items():
    parts = model_key.split('_')
    store = int(parts[1])
    item  = int(parts[3])
    model_count += 1

    print(f"  [{model_count}/{len(trained_models)}] {STORE_NAMES.get(store)} item {item}...", end='')

    try:
        predictions, actuals, dates = make_prophet_predictions(model, val, store, item)
        for i in range(len(predictions)):
            all_predictions.append({
                'date':      dates[i],
                'store_nbr': store,
                'item_nbr':  item,
                'actual':    actuals[i],
                'predicted': predictions[i],
                'error':     actuals[i] - predictions[i]
            })
        print(" done")
    except Exception as e:
        print(f" failed: {e}")
        continue

predictions_df = pd.DataFrame(all_predictions)

# overall metrics
y_true = predictions_df['actual'].values
y_pred = predictions_df['predicted'].values

rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mae  = mean_absolute_error(y_true, y_pred)
r2   = r2_score(y_true, y_pred)
mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

print(f"\nprophet validation results:")
print(f"  RMSE: {rmse:.2f}  MAE: {mae:.2f}  R2: {r2*100:.2f}%  MAPE: {mape:.2f}%")

if r2 < 0:
    print(f"warning: negative R2 — model performs worse than predicting the mean")
elif r2 > 0.95:
    print(f"warning: very high R2 — check for data leakage")

# product and store breakdown
predictions_df['abs_error'] = predictions_df['error'].abs()
predictions_df['pct_error'] = (predictions_df['abs_error'] / predictions_df['actual']) * 100

product_perf = predictions_df.groupby('item_nbr').agg(
    avg_actual=('actual', 'mean'),
    avg_predicted=('predicted', 'mean'),
    mae=('abs_error', 'mean'),
    mape=('pct_error', 'mean')
).round(2).sort_values('mae', ascending=False)

store_perf = predictions_df.groupby('store_nbr').agg(
    avg_actual=('actual', 'mean'),
    avg_predicted=('predicted', 'mean'),
    mae=('abs_error', 'mean'),
    mape=('pct_error', 'mean')
).round(2)
store_perf.index = store_perf.index.map(STORE_NAMES)

print(f"\nperformance by product:")
print(product_perf.to_string())

print(f"\nperformance by store:")
print(store_perf.to_string())

print(f"\nbest 3 products (lowest MAE):")
print(product_perf.nsmallest(3, 'mae')[['mae', 'mape']].to_string())

print(f"\nmost challenging 3 products (highest MAE):")
print(product_perf.nlargest(3, 'mae')[['mae', 'mape']].to_string())

# save results
pd.DataFrame({
    'Model': ['Prophet'],
    'RMSE':  [rmse],
    'MAE':   [mae],
    'R2':    [r2],
    'MAPE':  [mape]
}).to_csv(f'{RESULTS_DIR}/prophet_metrics.csv', index=False)

predictions_df.to_csv(f'{RESULTS_DIR}/prophet_predictions.csv', index=False)
product_perf.to_csv(f'{RESULTS_DIR}/prophet_product_performance.csv')
store_perf.to_csv(f'{RESULTS_DIR}/prophet_store_performance.csv')