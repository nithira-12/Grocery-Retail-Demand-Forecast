import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
import os
warnings.filterwarnings('ignore')

os.makedirs('../results', exist_ok=True)

STORE_NAMES = {44: "Store Colombo", 51: "Store Gampaha"}

# using env CSVs — same data as other models for consistency
train = pd.read_csv('../data/processed/train_env.csv', parse_dates=['date'])
val   = pd.read_csv('../data/processed/val_env.csv',   parse_dates=['date'])

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
    prophet_df['onpromotion']  = df_filtered['onpromotion'].values
    prophet_df['holiday_impact'] = df_filtered['holiday_impact'].values

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
    model.add_regressor('holiday_impact')
    model.fit(prophet_data)

    return model


def make_prophet_predictions(model, val_df, store, item):
    val_prophet = prepare_prophet_data(val_df, store, item)
    forecast    = model.predict(val_prophet)
    predictions = forecast['yhat'].values
    actuals     = val_prophet['y'].values
    return predictions, actuals, val_prophet['ds'].values


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
                'date':        dates[i],
                'store_nbr':   store,
                'item_nbr':    item,
                'actual':      actuals[i],
                'predictions': predictions[i],
                'error':       actuals[i] - predictions[i]
            })
        print(" done")
    except Exception as e:
        print(f" failed: {e}")
        continue

predictions_df = pd.DataFrame(all_predictions)

actual    = pd.Series(predictions_df['actual'].values)
predicted = pd.Series(predictions_df['predictions'].values)

rmse  = np.sqrt(mean_squared_error(actual, predicted))
mae   = mean_absolute_error(actual, predicted)
r2    = r2_score(actual, predicted)
mape  = np.mean(np.abs((actual - predicted) / actual.replace(0, np.nan))) * 100
wmape = (actual - predicted).abs().sum() / actual.sum() * 100

print(f"\nProphet validation results:")
print(f"  RMSE: {rmse:.2f}  MAE: {mae:.2f}  R2: {r2*100:.2f}%  WMAPE: {wmape:.2f}%  MAPE: {mape:.2f}%")

predictions_df.to_csv('../results/prophet_predictions.csv', index=False)

pd.DataFrame([{
    'Model': 'Prophet', 'RMSE': round(rmse, 2), 'MAE': round(mae, 2),
    'R2': round(r2, 4), 'R2_pct': round(r2 * 100, 2),
    'MAPE': round(mape, 2), 'WMAPE': round(wmape, 2)
}]).to_csv('../results/prophet_metrics.csv', index=False)