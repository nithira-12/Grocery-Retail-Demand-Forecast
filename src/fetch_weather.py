"""
FETCH WEATHER DATA - Sri Lanka Retail Forecasting
===================================================
Fetches historical daily weather for Colombo and Gampaha
from Open-Meteo API (free, no API key needed).

Covers 2013-01-01 to 2017-12-31 to match dataset.

HOW TO RUN:
    cd src
    python fetch_weather.py

OUTPUT:
    ../data/external/weather_colombo_gampaha.csv
"""

import requests
import pandas as pd
import numpy as np
import os
import time

print("=" * 60)
print("FETCH WEATHER DATA - Open-Meteo API")
print("Colombo + Gampaha | 2013-2017")
print("=" * 60)

# ── CREATE OUTPUT FOLDER ──────────────────────────────────
output_dir = '../data/external'
os.makedirs(output_dir, exist_ok=True)
print(f"\n Output folder ready: {output_dir}")

# ── STORE COORDINATES ─────────────────────────────────────
# Store 44 = Colombo, Store 51 = Gampaha
LOCATIONS = {
    44: {
        'name': 'Colombo',
        'latitude': 6.9271,
        'longitude': 79.8612
    },
    51: {
        'name': 'Gampaha',
        'latitude': 7.0840,
        'longitude': 80.0098
    }
}

DATE_START = "2013-01-01"
DATE_END   = "2017-12-31"

# ── FETCH FUNCTION ────────────────────────────────────────
def fetch_weather_for_location(store_nbr, info):
    """
    Fetch daily weather from Open-Meteo for one location.
    Returns a DataFrame with date, temperature, rainfall, humidity.
    """
    print(f"\n Fetching weather for Store {store_nbr} - {info['name']}...")
    print(f"   Coordinates: {info['latitude']}N, {info['longitude']}E")
    print(f"   Period: {DATE_START} to {DATE_END}")

    url = "https://archive-api.open-meteo.com/v1/archive"

    params = {
        "latitude":           info['latitude'],
        "longitude":          info['longitude'],
        "start_date":         DATE_START,
        "end_date":           DATE_END,
        "daily": [
            "temperature_2m_mean",    # Mean daily temperature (°C)
            "precipitation_sum",       # Total daily rainfall (mm)
            "relative_humidity_2m_mean" if False else "windspeed_10m_max",  # placeholder
        ],
        "timezone":           "Asia/Colombo"
    }

    # Use simpler parameter set that Open-Meteo archive supports reliably
    params = {
        "latitude":   info['latitude'],
        "longitude":  info['longitude'],
        "start_date": DATE_START,
        "end_date":   DATE_END,
        "daily":      "temperature_2m_mean,precipitation_sum,windspeed_10m_max",
        "timezone":   "Asia/Colombo"
    }

    try:
        response = requests.get(url, params=params, timeout=60)
        response.raise_for_status()
        data = response.json()

        if 'daily' not in data:
            print(f"   ERROR: No daily data in response")
            print(f"   Response: {data}")
            return None

        daily = data['daily']

        df = pd.DataFrame({
            'date':            pd.to_datetime(daily['time']),
            'store_nbr':       store_nbr,
            'temperature_c':   daily['temperature_2m_mean'],
            'rainfall_mm':     daily['precipitation_sum'],
            'windspeed_kmh':   daily['windspeed_10m_max'],
        })

        # Fill any missing values with forward fill then backward fill
        df['temperature_c'] = pd.to_numeric(df['temperature_c'], errors='coerce')
        df['rainfall_mm']   = pd.to_numeric(df['rainfall_mm'],   errors='coerce')
        df['windspeed_kmh'] = pd.to_numeric(df['windspeed_kmh'], errors='coerce')

        df['temperature_c'] = df['temperature_c'].fillna(method='ffill').fillna(method='bfill')
        df['rainfall_mm']   = df['rainfall_mm'].fillna(0)
        df['windspeed_kmh'] = df['windspeed_kmh'].fillna(method='ffill').fillna(method='bfill')

        print(f"   Fetched {len(df):,} daily records")
        print(f"   Temperature range: {df['temperature_c'].min():.1f}°C to {df['temperature_c'].max():.1f}°C")
        print(f"   Max rainfall day: {df['rainfall_mm'].max():.1f}mm")
        print(f"   Missing values: {df.isnull().sum().sum()}")

        return df

    except requests.exceptions.RequestException as e:
        print(f"   NETWORK ERROR: {e}")
        return None
    except Exception as e:
        print(f"   ERROR: {e}")
        return None

# ── FETCH FOR BOTH STORES ─────────────────────────────────
all_weather = []

for store_nbr, info in LOCATIONS.items():
    df = fetch_weather_for_location(store_nbr, info)

    if df is not None:
        all_weather.append(df)
        print(f"   Store {store_nbr} done.")
    else:
        print(f"   Store {store_nbr} FAILED - will use fallback values")
        # Fallback: use Sri Lanka climate averages if API fails
        dates = pd.date_range(start=DATE_START, end=DATE_END, freq='D')
        fallback_df = pd.DataFrame({'date': dates})
        fallback_df['store_nbr']    = store_nbr
        # Sri Lanka average climate values
        fallback_df['temperature_c'] = 28.0 + 2.0 * np.sin(
            2 * np.pi * (fallback_df['date'].dt.dayofyear / 365)
        )
        fallback_df['rainfall_mm']   = 5.0
        fallback_df['windspeed_kmh'] = 15.0
        all_weather.append(fallback_df)
        print(f"   Used climate averages as fallback for Store {store_nbr}")

    # Small delay between API calls to be respectful
    time.sleep(2)

# ── COMBINE AND SAVE ──────────────────────────────────────
if all_weather:
    combined = pd.concat(all_weather, ignore_index=True)
    combined = combined.sort_values(['store_nbr', 'date']).reset_index(drop=True)

    output_path = f'{output_dir}/weather_colombo_gampaha.csv'
    combined.to_csv(output_path, index=False)

    print(f"\n" + "=" * 60)
    print("WEATHER DATA SAVED SUCCESSFULLY")
    print("=" * 60)
    print(f"\n File: {output_path}")
    print(f" Total rows: {len(combined):,}")
    print(f" Stores: {combined['store_nbr'].unique()}")
    print(f" Date range: {combined['date'].min()} to {combined['date'].max()}")
    print(f"\n Columns saved:")
    for col in combined.columns:
        print(f"   - {col}")

    print(f"\n Sample data (first 5 rows):")
    print(combined.head())

    print(f"\n NEXT STEP: Run python fetch_economic.py")

else:
    print("\n ERROR: No weather data collected. Check internet connection.")