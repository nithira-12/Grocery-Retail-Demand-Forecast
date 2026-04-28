import wbgapi
import pandas as pd
import numpy as np
import os
import requests

output_dir = '../data/external'
os.makedirs(output_dir, exist_ok=True)

COUNTRY_CODE = 'LKA'
DATE_START   = '2013-01-01'
DATE_END     = '2017-12-31'

# fetch CPI from world bank, fall back to historical approximation if api fails
try:
    cpi_data = wbgapi.data.DataFrame(
        'FP.CPI.TOTL',
        economy=COUNTRY_CODE,
        time=range(2012, 2018)
    )

    cpi_data = cpi_data.T.reset_index()
    cpi_data.columns = ['year', 'cpi_annual']
    cpi_data['year'] = cpi_data['year'].astype(str).str.extract(r'(\d{4})').astype(int)
    cpi_data = cpi_data.dropna()
    cpi_data = cpi_data.sort_values('year').reset_index(drop=True)

    monthly_dates = pd.date_range(start='2012-01-01', end='2017-12-31', freq='MS')
    monthly_df = pd.DataFrame({'date': monthly_dates})
    monthly_df['year'] = monthly_df['date'].dt.year
    monthly_df = monthly_df.merge(cpi_data, on='year', how='left')
    monthly_df['cpi_index'] = monthly_df['cpi_annual'].interpolate(method='linear')

    daily_dates = pd.date_range(start=DATE_START, end=DATE_END, freq='D')
    daily_df = pd.DataFrame({'date': daily_dates})
    daily_df = daily_df.merge(
        monthly_df[['date', 'cpi_index']],
        on='date', how='left'
    )
    daily_df['cpi_index'] = daily_df['cpi_index'].interpolate(method='linear')
    daily_df['cpi_index'] = daily_df['cpi_index'].ffill().bfill()

except Exception as e:
    print(f"warning: world bank CPI fetch failed ({e}), using historical approximation")

    # sri lanka annual CPI approximation (base year 2010 = 100)
    annual_cpi = {
        2013: 143.5,
        2014: 149.8,
        2015: 151.2,
        2016: 156.9,
        2017: 164.3
    }

    daily_dates = pd.date_range(start=DATE_START, end=DATE_END, freq='D')
    daily_df = pd.DataFrame({'date': daily_dates})
    daily_df['year'] = daily_df['date'].dt.year
    daily_df['cpi_index'] = daily_df['year'].map(annual_cpi)
    daily_df['month_factor'] = 1 + 0.002 * (daily_df['date'].dt.month - 6)
    daily_df['cpi_index'] = daily_df['cpi_index'] * daily_df['month_factor']
    daily_df = daily_df[['date', 'cpi_index']]


# sri lanka petrol prices (LKR per litre) from CPC historical records
# only one price change in 2013-2017: 117 -> 106 in feb 2015
monthly_fuel = {
    '2013-01': 117, '2013-02': 117, '2013-03': 117, '2013-04': 117,
    '2013-05': 117, '2013-06': 117, '2013-07': 117, '2013-08': 117,
    '2013-09': 117, '2013-10': 117, '2013-11': 117, '2013-12': 117,
    '2014-01': 117, '2014-02': 117, '2014-03': 117, '2014-04': 117,
    '2014-05': 117, '2014-06': 117, '2014-07': 117, '2014-08': 117,
    '2014-09': 117, '2014-10': 117, '2014-11': 117, '2014-12': 117,
    '2015-01': 117, '2015-02': 106, '2015-03': 106, '2015-04': 106,
    '2015-05': 106, '2015-06': 106, '2015-07': 106, '2015-08': 106,
    '2015-09': 106, '2015-10': 106, '2015-11': 106, '2015-12': 106,
    '2016-01': 106, '2016-02': 106, '2016-03': 106, '2016-04': 106,
    '2016-05': 106, '2016-06': 106, '2016-07': 106, '2016-08': 106,
    '2016-09': 106, '2016-10': 106, '2016-11': 106, '2016-12': 106,
    '2017-01': 106, '2017-02': 106, '2017-03': 106, '2017-04': 106,
    '2017-05': 106, '2017-06': 106, '2017-07': 106, '2017-08': 106,
    '2017-09': 106, '2017-10': 106, '2017-11': 106, '2017-12': 106,
}

fuel_monthly = pd.DataFrame([
    {'date': pd.to_datetime(k + '-01'), 'fuel_price_lkr': v}
    for k, v in monthly_fuel.items()
])

daily_dates_full = pd.date_range(start=DATE_START, end=DATE_END, freq='D')
fuel_daily = pd.DataFrame({'date': daily_dates_full})
fuel_daily = fuel_daily.merge(fuel_monthly, on='date', how='left')
fuel_daily['fuel_price_lkr'] = fuel_daily['fuel_price_lkr'].ffill().bfill()

economic_df = daily_df.merge(fuel_daily, on='date', how='inner')
economic_df = economic_df[
    (economic_df['date'] >= DATE_START) &
    (economic_df['date'] <= DATE_END)
].reset_index(drop=True)

# normalise both indicators to zero mean unit variance for model input
cpi_mean = economic_df['cpi_index'].mean()
cpi_std  = economic_df['cpi_index'].std()
economic_df['cpi_normalized'] = (economic_df['cpi_index'] - cpi_mean) / cpi_std

fuel_mean = economic_df['fuel_price_lkr'].mean()
fuel_std  = economic_df['fuel_price_lkr'].std()
economic_df['fuel_normalized'] = (economic_df['fuel_price_lkr'] - fuel_mean) / fuel_std

print(f"   Combined rows: {len(economic_df):,}")
print(f"   Columns: {list(economic_df.columns)}")

output_path = f'{output_dir}/economic_indicators.csv'
economic_df.to_csv(output_path, index=False)