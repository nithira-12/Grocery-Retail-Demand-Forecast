import requests
import pandas as pd
import numpy as np
import os
import time

output_dir = '../data/external'
os.makedirs(output_dir, exist_ok=True)

# store 44 = colombo, store 51 = gampaha
LOCATIONS = {
    44: {'name': 'Colombo', 'latitude': 6.9271, 'longitude': 79.8612},
    51: {'name': 'Gampaha', 'latitude': 7.0840, 'longitude': 80.0098}
}

DATE_START = "2013-01-01"
DATE_END   = "2017-12-31"


def fetch_weather_for_location(store_nbr, info):
    url = "https://archive-api.open-meteo.com/v1/archive"

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
            print(f"warning: no daily data in response for store {store_nbr}")
            return None

        daily = data['daily']

        df = pd.DataFrame({
            'date':          pd.to_datetime(daily['time']),
            'store_nbr':     store_nbr,
            'temperature_c': daily['temperature_2m_mean'],
            'rainfall_mm':   daily['precipitation_sum'],
            'windspeed_kmh': daily['windspeed_10m_max'],
        })

        df['temperature_c'] = pd.to_numeric(df['temperature_c'], errors='coerce')
        df['rainfall_mm']   = pd.to_numeric(df['rainfall_mm'],   errors='coerce')
        df['windspeed_kmh'] = pd.to_numeric(df['windspeed_kmh'], errors='coerce')

        df['temperature_c'] = df['temperature_c'].ffill().bfill()
        df['rainfall_mm']   = df['rainfall_mm'].fillna(0)
        df['windspeed_kmh'] = df['windspeed_kmh'].ffill().bfill()

        return df

    except requests.exceptions.RequestException as e:
        print(f"warning: network error fetching weather for store {store_nbr}: {e}")
        return None
    except Exception as e:
        print(f"warning: error fetching weather for store {store_nbr}: {e}")
        return None


all_weather = []

for store_nbr, info in LOCATIONS.items():
    df = fetch_weather_for_location(store_nbr, info)

    if df is not None:
        all_weather.append(df)
    else:
        # api failed — use sri lanka climate averages as fallback
        dates = pd.date_range(start=DATE_START, end=DATE_END, freq='D')
        fallback_df = pd.DataFrame({'date': dates})
        fallback_df['store_nbr']     = store_nbr
        fallback_df['temperature_c'] = 28.0 + 2.0 * np.sin(
            2 * np.pi * (fallback_df['date'].dt.dayofyear / 365)
        )
        fallback_df['rainfall_mm']   = 5.0
        fallback_df['windspeed_kmh'] = 15.0
        all_weather.append(fallback_df)
        print(f"warning: used climate averages as fallback for store {store_nbr}")

    time.sleep(2)

if all_weather:
    combined = pd.concat(all_weather, ignore_index=True)
    combined = combined.sort_values(['store_nbr', 'date']).reset_index(drop=True)

    output_path = f'{output_dir}/weather_colombo_gampaha.csv'
    combined.to_csv(output_path, index=False)
else:
    print("error ,  no weather data collected ")