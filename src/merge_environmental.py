import pandas as pd
import numpy as np
import os

weather  = pd.read_csv('../data/external/weather_colombo_gampaha.csv', parse_dates=['date'])
economic = pd.read_csv('../data/external/economic_indicators.csv',      parse_dates=['date'])

weather_cols  = ['date', 'store_nbr', 'temperature_c', 'rainfall_mm', 'windspeed_kmh']
economic_cols = ['date', 'cpi_normalized', 'fuel_normalized']

weather  = weather[weather_cols]
economic = economic[economic_cols]

splits = {
    'train': '../data/processed/train_processed.csv',
    'val':   '../data/processed/val_processed.csv',
    'test':  '../data/processed/test_processed.csv',
}

for split_name, input_path in splits.items():
    df = pd.read_csv(input_path, parse_dates=['date'])

    # merge weather by date and store — economic by date only (same for both stores)
    df = df.merge(weather,   on=['date', 'store_nbr'], how='left')
    df = df.merge(economic,  on='date',                how='left')

    env_cols = ['temperature_c', 'rainfall_mm', 'windspeed_kmh',
                'cpi_normalized', 'fuel_normalized']

    missing = df[env_cols].isnull().sum()
    if missing.sum() > 0:
        print(f"warning: missing values in {split_name} environmental features — filling")
        for col in env_cols:
            if df[col].isnull().sum() > 0:
                df[col] = df[col].ffill().bfill()

    output_path = input_path.replace('_processed.csv', '_env.csv')
    df.to_csv(output_path, index=False)

print("merge complete  train_env.csv, val_env.csv, test_env.csv saved")