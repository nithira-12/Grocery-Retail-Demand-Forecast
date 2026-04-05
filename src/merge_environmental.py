"""
MERGE ENVIRONMENTAL DATA
=========================
Merges weather and economic data into existing processed CSVs.
Adds 4 new features to each dataset:
    - temperature_c
    - rainfall_mm
    - cpi_normalized
    - fuel_normalized

Takes your existing 16 features and creates 20 features.

HOW TO RUN:
    cd src
    python merge_environmental.py

INPUT:
    ../data/processed/train_processed.csv
    ../data/processed/val_processed.csv
    ../data/processed/test_processed.csv
    ../data/external/weather_colombo_gampaha.csv
    ../data/external/economic_indicators.csv

OUTPUT:
    ../data/processed/train_env.csv
    ../data/processed/val_env.csv
    ../data/processed/test_env.csv
"""

import pandas as pd
import numpy as np
import os

print("=" * 60)
print("MERGE ENVIRONMENTAL DATA INTO PROCESSED CSVs")
print("=" * 60)

# ── LOAD ENVIRONMENTAL DATA ───────────────────────────────
print("\n Loading environmental data...")

weather = pd.read_csv('../data/external/weather_colombo_gampaha.csv',
                      parse_dates=['date'])
economic = pd.read_csv('../data/external/economic_indicators.csv',
                       parse_dates=['date'])

print(f"   Weather rows: {len(weather):,}")
print(f"   Economic rows: {len(economic):,}")

# Economic data has no store_nbr - same for both stores
# We'll merge by date only for economic
weather_cols   = ['date', 'store_nbr', 'temperature_c', 'rainfall_mm', 'windspeed_kmh']
economic_cols  = ['date', 'cpi_normalized', 'fuel_normalized']

weather  = weather[weather_cols]
economic = economic[economic_cols]

# ── PROCESS EACH SPLIT ───────────────────────────────────
splits = {
    'train': '../data/processed/train_processed.csv',
    'val':   '../data/processed/val_processed.csv',
    'test':  '../data/processed/test_processed.csv',
}

for split_name, input_path in splits.items():
    print(f"\n Processing {split_name} split...")

    # Load original processed CSV
    df = pd.read_csv(input_path, parse_dates=['date'])
    original_shape = df.shape
    print(f"   Original shape: {original_shape}")
    print(f"   Original columns: {list(df.columns)}")

    # Merge weather by date AND store_nbr
    df = df.merge(weather, on=['date', 'store_nbr'], how='left')

    # Merge economic by date only
    df = df.merge(economic, on='date', how='left')

    # Check for missing values after merge
    env_cols = ['temperature_c', 'rainfall_mm', 'windspeed_kmh',
                'cpi_normalized', 'fuel_normalized']

    missing = df[env_cols].isnull().sum()
    if missing.sum() > 0:
        print(f"   Missing values found - filling...")
        for col in env_cols:
            if df[col].isnull().sum() > 0:
                df[col] = df[col].ffill().bfill()
                print(f"     {col}: filled {missing[col]} missing values")
    else:
        print(f"   No missing values in environmental features")

    # Save new version
    output_path = input_path.replace('_processed.csv', '_env.csv')
    df.to_csv(output_path, index=False)

    new_shape = df.shape
    print(f"   New shape: {new_shape}")
    print(f"   New columns added: {new_shape[1] - original_shape[1]}")
    print(f"   Saved to: {output_path}")

    # Verify
    print(f"\n   Environmental feature sample:")
    print(df[['date', 'store_nbr'] + env_cols].head(3).to_string())

# ── FINAL SUMMARY ─────────────────────────────────────────
print(f"\n" + "=" * 60)
print("MERGE COMPLETE")
print("=" * 60)

print(f"""
 Files created:
   ../data/processed/train_env.csv
   ../data/processed/val_env.csv
   ../data/processed/test_env.csv

 Features went from 16 → 20:
   Added: temperature_c, rainfall_mm, windspeed_kmh
          cpi_normalized, fuel_normalized

 NEXT STEP: Run python retrain_models.py
""")