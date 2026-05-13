import pandas as pd
import numpy as np
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')


SELECTED_STORES = [44, 51]
STORE_NAMES = {
    44: "Store Colombo",
    51: "Store Gampaha"
}

PERISHABLE_ITEMS = [
    1503844,  # vegetables A
    1473474,  # vegetables B
    1695835,  # fruit A (bananas)
    502331,   # bread A
    564287,   # baked goods B
    584028,   # meat product A
    903285,   # poultry A
    1167614,  # eggs
    1427659,  # dairy
]

NON_PERISHABLE_ITEMS = [
    1047679,  # soft drinks A
    364606,   # staples A (rice/flour)
    265559,   # staples B (rice/flour)
]

SELECTED_ITEMS = PERISHABLE_ITEMS + NON_PERISHABLE_ITEMS

TRAIN_START = '2013-01-01'
TRAIN_END   = '2015-12-31'
VAL_START   = '2016-01-01'
VAL_END     = '2016-12-31'
TEST_START  = '2017-01-01'
TEST_END    = '2017-08-15'

RAW_DATA_DIR       = '../data/raw'
PROCESSED_DATA_DIR = '../data/processed'
HOLIDAYS_DIR       = '../data/holidays'

os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)


def load_raw_data():
    train = pd.read_csv(
        f'{RAW_DATA_DIR}/train.csv',
        parse_dates=['date'],
        dtype={
            'id': 'int32',
            'store_nbr': 'int8',
            'item_nbr': 'int32',
            'unit_sales': 'float32'
        }
    )
    items    = pd.read_csv(f'{RAW_DATA_DIR}/items.csv')
    stores   = pd.read_csv(f'{RAW_DATA_DIR}/stores.csv')
    holidays = pd.read_csv(f'{HOLIDAYS_DIR}/sri_lanka_holidays.csv', parse_dates=['date'])
    return train, items, stores, holidays


def filter_data(train):
    filtered = train[
        (train['store_nbr'].isin(SELECTED_STORES)) &
        (train['item_nbr'].isin(SELECTED_ITEMS))
    ].copy()

    unique_items = filtered['item_nbr'].nunique()
    if unique_items != len(SELECTED_ITEMS):
        print(f"warning: missing {len(SELECTED_ITEMS) - unique_items} products in filtered data")

    return filtered


def correct_stockout_zeros(df):
    """
    Identifies probable stockout days and replaces zero sales with a
    locally-smoothed demand estimate.

    Threshold lowered to 0.30: products selling on more than 30% of days
    are treated as consistently-selling. This captures perishable products
    (meat, poultry, baked goods) which may sell on 35-45% of days but are
    not genuinely intermittent — their zeros are more likely stockouts.

    Methodology follows Andrade & Cunha (2023).
    """
    df = df.copy().sort_values(['store_nbr', 'item_nbr', 'date'])
    corrected_count = 0

    for (store, item), group in df.groupby(['store_nbr', 'item_nbr']):
        positive_rate = (group['unit_sales'] > 0).mean()

        # threshold of 0.30 captures perishable products that sell on
        # 30%+ of days — low enough to include meat, poultry, baked goods
        if positive_rate <= 0.30:
            continue

        idx = group.index

        rolling_avg = (
            df.loc[idx, 'unit_sales']
            .replace(0, np.nan)
            .rolling(window=7, min_periods=3, center=True)
            .mean()
        )

        zero_mask    = df.loc[idx, 'unit_sales'] == 0
        product_mean = df.loc[idx, 'unit_sales'].replace(0, np.nan).mean()
        fill_values  = rolling_avg.fillna(product_mean)

        corrections = zero_mask.sum()
        if corrections > 0:
            df.loc[idx[zero_mask], 'unit_sales'] = fill_values[zero_mask].values
            corrected_count += corrections

    print(f"stockout correction: {corrected_count} zero-sales days imputed")
    return df


def add_temporal_features(df):
    df['year']           = df['date'].dt.year
    df['month']          = df['date'].dt.month
    df['day']            = df['date'].dt.day
    df['dayofweek']      = df['date'].dt.dayofweek
    df['dayofyear']      = df['date'].dt.dayofyear
    df['week']           = df['date'].dt.isocalendar().week.astype('int32')
    df['quarter']        = df['date'].dt.quarter
    df['time_idx']       = (df['date'] - df['date'].min()).dt.days
    df['is_weekend']     = (df['dayofweek'] >= 5).astype('int8')
    df['is_month_start'] = df['date'].dt.is_month_start.astype('int8')
    df['is_month_end']   = df['date'].dt.is_month_end.astype('int8')
    return df


def add_sri_lankan_holiday_features(df, holidays):
    holidays['is_holiday'] = 1

    df = df.merge(
        holidays[['date', 'is_holiday', 'general_impact', 'meat_restriction']],
        on='date',
        how='left'
    )

    df['is_holiday']       = df['is_holiday'].fillna(0).astype('int8')
    df['general_impact']   = df['general_impact'].fillna(0).astype('int8')
    df['meat_restriction'] = df['meat_restriction'].fillna(0).astype('int8')

    df['holiday_impact'] = df['general_impact'].copy()

    meat_poultry_products = [584028, 903285]
    for product_id in meat_poultry_products:
        mask = (df['item_nbr'] == product_id) & (df['meat_restriction'] == 1)
        df.loc[mask, 'holiday_impact'] = -1

    return df


def clean_promotions(df):
    df['onpromotion'] = df['onpromotion'].fillna(False)
    df['onpromotion'] = df['onpromotion'].astype(str).str.lower()
    df['onpromotion'] = (df['onpromotion'] == 'true').astype('int8')
    return df


def add_product_metadata(df, items):
    df = df.merge(
        items[['item_nbr', 'family', 'perishable']],
        on='item_nbr',
        how='left'
    )
    df['perishable']     = df['perishable'].astype('int8')
    df['family_encoded'] = df['family'].astype('category').cat.codes
    return df


def split_data(df):
    train = df[(df['date'] >= TRAIN_START) & (df['date'] <= TRAIN_END)].copy()
    val   = df[(df['date'] >= VAL_START)   & (df['date'] <= VAL_END)].copy()
    test  = df[(df['date'] >= TEST_START)  & (df['date'] <= TEST_END)].copy()
    return train, val, test


def save_processed_data(train, val, test, full_df):
    train.to_csv(f'{PROCESSED_DATA_DIR}/train_processed.csv', index=False)
    val.to_csv(f'{PROCESSED_DATA_DIR}/val_processed.csv',     index=False)
    test.to_csv(f'{PROCESSED_DATA_DIR}/test_processed.csv',   index=False)
    full_df.to_csv(f'{PROCESSED_DATA_DIR}/full_processed.csv', index=False)


def main():
    start_time = datetime.now()

    try:
        train, items, stores, holidays = load_raw_data()
        filtered = filter_data(train)
        filtered = add_temporal_features(filtered)
        filtered = add_sri_lankan_holiday_features(filtered, holidays)
        filtered = clean_promotions(filtered)
        filtered = add_product_metadata(filtered, items)
        filtered = correct_stockout_zeros(filtered)

        train_df, val_df, test_df = split_data(filtered)
        save_processed_data(train_df, val_df, test_df, filtered)

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        print(f"done in {duration/60:.1f} minutes")

    except Exception as e:
        print(f"error: {str(e)}")
        raise


if __name__ == "__main__":
    main()