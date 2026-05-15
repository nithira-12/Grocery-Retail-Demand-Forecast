import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

os.makedirs('../results', exist_ok=True)
os.makedirs('../models', exist_ok=True)

train = pd.read_csv('../data/processed/train_env.csv', parse_dates=['date'])
val   = pd.read_csv('../data/processed/val_env.csv',   parse_dates=['date'])

ALL_FEATURES = [
    'sales_lag_1', 'sales_lag_2',
    'sales_lag_7', 'sales_lag_14', 'sales_lag_28',
    'sales_rolling_mean_7', 'sales_rolling_mean_28', 'sales_rolling_std_7',
    'dayofweek', 'month', 'quarter', 'day',
    'is_weekend', 'is_month_start', 'is_month_end',
    'onpromotion', 'holiday_impact', 'family_encoded',
    'temperature_c', 'rainfall_mm', 'windspeed_kmh',
    'cpi_normalized', 'fuel_normalized'
]

TARGET = 'unit_sales'


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


train_lagged = create_lag_features(train)
val_lagged   = create_lag_features(val)

available_features = [f for f in ALL_FEATURES if f in train_lagged.columns]
print(f"features used: {len(available_features)}")

X_train = train_lagged[available_features].values
y_train = train_lagged[TARGET].values
X_val   = val_lagged[available_features].values
y_val   = val_lagged[TARGET].values

# scale features to 0-1 range — LSTM is sensitive to input scale
# unlike XGBoost which handles raw values natively
feature_scaler = MinMaxScaler()
target_scaler  = MinMaxScaler()

X_train_scaled = feature_scaler.fit_transform(X_train)
X_val_scaled   = feature_scaler.transform(X_val)

y_train_scaled = target_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
y_val_scaled   = target_scaler.transform(y_val.reshape(-1, 1)).flatten()

# reshape to (samples, timesteps, features) — LSTM requires 3D input
# timesteps=1 because our lag features already encode the temporal context
X_train_lstm = X_train_scaled.reshape(X_train_scaled.shape[0], 1, X_train_scaled.shape[1])
X_val_lstm   = X_val_scaled.reshape(X_val_scaled.shape[0], 1, X_val_scaled.shape[1])

print("training LSTM — this will take several minutes...")

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping

    tf.random.set_seed(42)

    model = Sequential([
        LSTM(64, input_shape=(1, len(available_features)), return_sequences=True),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse')

    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )

    history = model.fit(
        X_train_lstm, y_train_scaled,
        validation_data=(X_val_lstm, y_val_scaled),
        epochs=100,
        batch_size=256,
        callbacks=[early_stop],
        verbose=1
    )

    lstm_pred_scaled = model.predict(X_val_lstm, verbose=0).flatten()
    lstm_pred = target_scaler.inverse_transform(
        lstm_pred_scaled.reshape(-1, 1)
    ).flatten()
    lstm_pred = np.maximum(lstm_pred, 0)

    actual    = pd.Series(y_val)
    predicted = pd.Series(lstm_pred)

    rmse  = np.sqrt(mean_squared_error(actual, predicted))
    mae   = mean_absolute_error(actual, predicted)
    r2    = r2_score(actual, predicted)
    mape  = np.mean(np.abs((actual - predicted) / actual.replace(0, np.nan))) * 100
    wmape = (actual - predicted).abs().sum() / actual.sum() * 100

    print(f"\nLSTM validation results:")
    print(f"  RMSE:  {rmse:.2f}")
    print(f"  MAE:   {mae:.2f}")
    print(f"  R2:    {r2*100:.2f}%")
    print(f"  WMAPE: {wmape:.2f}%  (primary metric)")
    print(f"  MAPE:  {mape:.2f}%  (inflated by near-zero days)")

    results_df = val_lagged[['date', 'store_nbr', 'item_nbr']].copy()
    results_df['actual']      = y_val
    results_df['predictions'] = lstm_pred
    results_df['error']       = results_df['actual'] - results_df['predictions']
    results_df.to_csv('../results/lstm_predictions.csv', index=False)

    metrics_df = pd.DataFrame([{
        'Model': 'LSTM',
        'RMSE':  round(rmse, 2),
        'MAE':   round(mae, 2),
        'R2':    round(r2, 4),
        'R2_pct': round(r2 * 100, 2),
        'MAPE':  round(mape, 2),
        'WMAPE': round(wmape, 2),
    }])
    metrics_df.to_csv('../results/lstm_metrics.csv', index=False)

    stopped_epoch = early_stop.stopped_epoch if early_stop.stopped_epoch > 0 else 100
    print(f"\ntraining stopped at epoch {stopped_epoch}")
    print(f"best val_loss: {min(history.history['val_loss']):.6f}")

except ImportError:
    print("tensorflow not installed — run: pip install tensorflow")
    print("then re-run this script")