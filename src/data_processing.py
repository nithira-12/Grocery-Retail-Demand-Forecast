import pandas as pd
import numpy as np
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')


# CONFIGURATION

print("\n")
print("DEMAND FORECASTING - DATA PROCESSING")
print("\n")

# Selected stores (Ecuador → Sri Lanka mapping)
SELECTED_STORES = [44, 51]  # Store 44 (Quito), Store 51 (Guayaquil)
STORE_NAMES = {
    44: "Store Colombo",
    51: "Store Gampaha"
}

# Selected items - separated by perishability
PERISHABLE_ITEMS = [
    1503844,  # PRODUCE - Vegetables A (6.3M sales, 100% activity)
    1473474,  # PRODUCE - Vegetables B (5.0M sales, 99.99% activity)
    1695835,  # PRODUCE - Fruit A (Bananas) (2.5M sales, 100% activity)  # ← CHANGED: Renamed from vege C
    502331,   # BREAD/BAKERY -  Bread A (3.7M sales, 100% activity)
    564287,   # BREAD/BAKERY - Baked Goods B (1.7M sales, 99.99% activity)
    584028,   # MEATS - Product A (3.3M sales, 99.99% activity)
    903285,   # POULTRY - Product A (2.6M sales, 99.99% activity)
    1167614,  # EGGS - Standard (3.2M sales, 99.99% activity)
    1427659,  # DAIRY - (1.9M sales, 99.99% activity)
]

NON_PERISHABLE_ITEMS = [
    1047679,  # BEVERAGES - Soft Drinks A (5.5M sales, 99.99% activity)  # ← CHANGED: Clarified as soft drinks
    364606,   # GROCERY I - Staples A (Rice/Flour) (4.4M sales, 100% activity)  # ← CHANGED: Added clarification
    265559,   # GROCERY I - Staples B (Rice/Flour) (4.1M sales, 99.99% activity)  # ← CHANGED: Added clarification
]

# Combine all selected items
SELECTED_ITEMS = PERISHABLE_ITEMS + NON_PERISHABLE_ITEMS

# Date ranges for splits
TRAIN_START = '2013-01-01'
TRAIN_END = '2015-12-31'
VAL_START = '2016-01-01'
VAL_END = '2016-12-31'
TEST_START = '2017-01-01'
TEST_END = '2017-08-15'  # Dataset ends here

# File paths
RAW_DATA_DIR = '../data/raw'
PROCESSED_DATA_DIR = '../data/processed'
HOLIDAYS_DIR = '../data/holidays'  # ← CHANGED: Added holidays directory

# Create output directory if it doesn't exist
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

print(f"\n Configuration loaded")
print(f"   stores: {len(SELECTED_STORES)} ({STORE_NAMES[44]}, {STORE_NAMES[51]})")
print(f"   products: {len(SELECTED_ITEMS)} ({len(PERISHABLE_ITEMS)} perishable + {len(NON_PERISHABLE_ITEMS)} non-perishable)")
print(f"  train period: {TRAIN_START} to {TRAIN_END}")
print(f"  validation period: {VAL_START} to {VAL_END}")
print(f"  test period: {TEST_START} to {TEST_END}")




# FUNCTION 1: LOAD RAW DATA


def load_raw_data():
    """Load raw Favorita datasets with optimized data types"""
   
    print("STEP 1: LOADING RAW DATA")
   
    
    # Load train data with optimized dtypes to save memory
    print("\n Loading train.csv ...")
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
    print(f"   loaded train.csv: {train.shape[0]:,} rows, {train.shape[1]} columns")
    
    # Load items
    print("\n Loading items.csv...")
    items = pd.read_csv(f'{RAW_DATA_DIR}/items.csv')
    print(f"   Loaded items.csv: {items.shape[0]:,} rows")
    
    # Load stores
    print("\n Loading stores.csv...")
    stores = pd.read_csv(f'{RAW_DATA_DIR}/stores.csv')
    print(f"   Loaded stores.csv: {stores.shape[0]:,} rows")
    
    # ← CHANGED: Load Sri Lankan holidays instead of Ecuador holidays
    print("\n Loading Sri Lankan holidays...")
    holidays = pd.read_csv(
        f'{HOLIDAYS_DIR}/sri_lanka_holidays.csv',
        parse_dates=['date']
    )
    print(f"   Loaded sri_lanka_holidays.csv: {holidays.shape[0]:,} entries")
    
    print(f"\n all data loaded successfully!")
    print(f"  total memory usage: ~{train.memory_usage(deep=True).sum() / 1024**3:.2f} GB")
    
    return train, items, stores, holidays



# FUNCTION 2: FILTER DATA


def filter_data(train):
    """Filter to selected stores and items"""

    print("STEP 2: FILTERING DATA")
    
    print(f"\n Original dataset size: {train.shape[0]:,} rows")
    
    # Filter to selected stores and items
    filtered = train[
        (train['store_nbr'].isin(SELECTED_STORES)) &
        (train['item_nbr'].isin(SELECTED_ITEMS))
    ].copy()
    
    print(f"  filtered to {len(SELECTED_STORES)} stores and {len(SELECTED_ITEMS)} products")
    print(f"   new size: {filtered.shape[0]:,} rows")
    print(f"  reduction: {(1 - len(filtered)/len(train))*100:.1f}%")
    print(f"    memory saved: ~{(train.memory_usage(deep=True).sum() - filtered.memory_usage(deep=True).sum()) / 1024**3:.2f} GB")
    
    # verify we have all products
    unique_items = filtered['item_nbr'].nunique()
    unique_stores = filtered['store_nbr'].nunique()
    
    print(f"\n Data verification:")
    print(f"  unique stores: {unique_stores}/{len(SELECTED_STORES)}")
    print(f"  unique products: {unique_items}/{len(SELECTED_ITEMS)}")
    
    if unique_items != len(SELECTED_ITEMS):
        print(f"  warning: Missing {len(SELECTED_ITEMS) - unique_items} products!")
    
    return filtered



# FUNCTION 3: ADD TEMPORAL FEATURES


def add_temporal_features(df):
    """Add date-based features for model training"""
    print("\n")
    print("STEP 3: ADDING TEMPORAL FEATURES")
    print("\n")
    print("\n")
    
    print("\n Creating temporal features...")
    
    # Basic date components
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['dayofweek'] = df['date'].dt.dayofweek  # Monday=0, Sunday=6
    df['dayofyear'] = df['date'].dt.dayofyear
    df['week'] = df['date'].dt.isocalendar().week.astype('int32')
    df['quarter'] = df['date'].dt.quarter
    df['time_idx'] = (df['date'] - df['date'].min()).dt.days
    
    # Weekend flag
    df['is_weekend'] = (df['dayofweek'] >= 5).astype('int8')
    
    # Month start/end flags
    df['is_month_start'] = df['date'].dt.is_month_start.astype('int8')
    df['is_month_end'] = df['date'].dt.is_month_end.astype('int8')
    
    print(f"   Added 11 temporal features:")
    print(f"     year, month, day, dayofweek, dayofyear")
    print(f"   week, quarter, time_idx")
    print(f"    is_weekend, is_month_start, is_month_end")
    
    return df




# ============================================================================
# CHANGED: COMPLETELY NEW FUNCTION FOR SRI LANKAN HOLIDAYS
# ============================================================================

def add_sri_lankan_holiday_features(df, holidays):
    """
    Add Sri Lankan holiday features with sophisticated impact modeling
    
    Features added:
    - is_holiday: Binary flag (0/1)
    - general_impact: -1 (closure), 0 (neutral), +1 (shopping spike)
    - meat_restriction: 0/1 flag for Poya days and Independence Day
    - holiday_impact: Final feature used by model (combines general + meat restriction)
    """
    print("\n")
    print("STEP 4: ADDING SRI LANKAN HOLIDAY FEATURES")
    print("\n")
    print("\n")
    
    print("\n Processing Sri Lankan holiday calendar...")
    print(f"   loaded {len(holidays)} holiday entries (2013-2017)")
    
    # Show breakdown
    spike_days = len(holidays[holidays['general_impact'] == 1])
    closure_days = len(holidays[holidays['general_impact'] == -1])
    meat_restriction_days = len(holidays[holidays['meat_restriction'] == 1])
    
    print(f"\n Holiday breakdown:")
    print(f"   Shopping spike days (+1): {spike_days}")
    print(f"   Shop closure days (-1): {closure_days}")
    print(f"   Meat restriction days: {meat_restriction_days}")
    
    # Create holiday flag
    holidays['is_holiday'] = 1
    
    # Merge with main data
    print(f"\n Merging holiday features with sales data...")
    df = df.merge(
        holidays[['date', 'is_holiday', 'general_impact', 'meat_restriction']], 
        on='date', 
        how='left'
    )
    
    # Fill missing (non-holiday days) with 0
    df['is_holiday'] = df['is_holiday'].fillna(0).astype('int8')
    df['general_impact'] = df['general_impact'].fillna(0).astype('int8')
    df['meat_restriction'] = df['meat_restriction'].fillna(0).astype('int8')
    
    # Create holiday_impact feature (what model actually uses)
    df['holiday_impact'] = df['general_impact'].copy()
    
    # For meat/poultry products, override with meat restriction on Poya days
    meat_poultry_products = [584028, 903285]  # Meat & Poultry product IDs
    
    print(f"\n Applying meat/poultry restriction logic...")
    for product_id in meat_poultry_products:
        mask = (df['item_nbr'] == product_id) & (df['meat_restriction'] == 1)
        affected_rows = mask.sum()
        df.loc[mask, 'holiday_impact'] = -1  # Force restriction impact
        print(f"   Product {product_id}: {affected_rows:,} Poya restriction days applied")
    
    print(f"\n Holiday features successfully added:")
    print(f"  is_holiday: Binary flag (0/1)")
    print(f"  general_impact: Shop impact (-1/0/+1)")
    print(f"    meat_restriction: Poya restriction flag (0/1)")
    print(f"    holiday_impact: Final model feature (-1/0/+1)")
    
    # Show distribution
    num_holiday_rows = df['is_holiday'].sum()
    spike_rows = (df['holiday_impact'] == 1).sum()
    closure_rows = (df['holiday_impact'] == -1).sum()
    neutral_rows = (df['holiday_impact'] == 0).sum()
    
    print(f"\n Holiday impact distribution in dataset:")
    print(f"  Total holiday records: {num_holiday_rows:,}")
    print(f"   Shopping spike records (+1): {spike_rows:,}")
    print(f"   Closure records (-1): {closure_rows:,}")
    print(f"   Neutral records (0): {neutral_rows:,}")
    
    print(f"\n Key holiday patterns captured:")
    print(f"   New Year shopping surge (7 days)")
    print(f"  Monthly Poya days (meat restrictions)")
    print(f"   Vesak & Poson (shop closures)")
    print(f"   Eid & Deepavali (multi-day patterns)")
    print(f"   Christmas & Thai Pongal shopping spikes")
    
    return df


# FUNCTION 5: CLEAN PROMOTION DATA

def clean_promotions(df):
    """Clean and convert promotion column to binary"""
    print("\n")
    print("STEP 5: CLEANING PROMOTION DATA")
    print("\n")
    print("\n")
    
    print(f"\n Original onpromotion column:")
    print(f"   Data type: {df['onpromotion'].dtype}")
    print(f"   Missing values: {df['onpromotion'].isnull().sum():,}")
    
    # Convert to numeric, treating NaN and False as 0, True as 1
    df['onpromotion'] = df['onpromotion'].fillna(False)
    df['onpromotion'] = df['onpromotion'].astype(str).str.lower()
    df['onpromotion'] = (df['onpromotion'] == 'true').astype('int8')
    
    print(f"\n Cleaned onpromotion column:")
    print(f"  Converted to binary (0/1)")
    print(f"  Items on promotion: {df['onpromotion'].sum():,} ({df['onpromotion'].mean()*100:.1f}%)")
    print(f"  No missing values")
    
    return df


# FUNCTION 6: ADD PRODUCT METADATA


def add_product_metadata(df, items):
    """Merge product family and perishability info"""
    print("\n")
    print("STEP 6: ADDING PRODUCT METADATA")
    print("\n")
    print("\n")
    
    print("\n Merging item metadata...")
    
    # Merge family and perishable info
    df = df.merge(
        items[['item_nbr', 'family', 'perishable']],
        on='item_nbr',
        how='left'
    )
    
    # Convert perishable to int8
    df['perishable'] = df['perishable'].astype('int8')
    
    # Encode family as numeric (for XGBoost)
    print("\n Encoding product family...")
    df['family_encoded'] = df['family'].astype('category').cat.codes
    
    print(f"   Added family (product category)")
    print(f"   Added family_encoded (numeric version for models)")
    print(f"  Added perishable flag")
    
    print(f"\n Product family distribution:")
    family_counts = df.groupby(['family', 'family_encoded']).size().reset_index(name='count')
    print(family_counts.to_string(index=False))
    
    print(f"\n Perishability:")
    print(f"   Perishable: {(df['perishable']==1).sum():,} rows")
    print(f"   Non-perishable: {(df['perishable']==0).sum():,} rows")
    
    return df



# FUNCTION 7: SPLIT DATA INTO TRAIN/VAL/TEST


def split_data(df):
    """Split data by date ranges into train, validation, and test sets"""
    print("\n")
    print("STEP 7: SPLITTING DATA")
    print("\n")
    print("\n")
    
    print("\n Splitting by date ranges...")
    
    # Train set (2013-2015)
    train = df[(df['date'] >= TRAIN_START) & (df['date'] <= TRAIN_END)].copy()
    
    # Validation set (2016)
    val = df[(df['date'] >= VAL_START) & (df['date'] <= VAL_END)].copy()
    
    # Test set (2017)
    test = df[(df['date'] >= TEST_START) & (df['date'] <= TEST_END)].copy()
    
    print(f"\n Train set ({TRAIN_START} to {TRAIN_END}):")
    print(f"   {train.shape[0]:,} rows")
    print(f"   {train.shape[1]} columns")
    print(f"   {train['date'].min()} to {train['date'].max()}")
    
    print(f"\n Validation set ({VAL_START} to {VAL_END}):")
    print(f"   {val.shape[0]:,} rows")
    print(f"  {val.shape[1]} columns")
    print(f"   {val['date'].min()} to {val['date'].max()}")
    
    print(f"\n Test set ({TEST_START} to {TEST_END}):")
    print(f"   {test.shape[0]:,} rows")
    print(f"   {test.shape[1]} columns")
    print(f"   {test['date'].min()} to {test['date'].max()}")
    
    # Calculate split percentages
    total_rows = len(df)
    train_pct = len(train) / total_rows * 100
    val_pct = len(val) / total_rows * 100
    test_pct = len(test) / total_rows * 100
    
    print(f"\n Split distribution:")
    print(f"   Train: {train_pct:.1f}%")
    print(f"   Validation: {val_pct:.1f}%")
    print(f"   Test: {test_pct:.1f}%")
    
    return train, val, test


# FUNCTION 8: SAVE PROCESSED DATA


def save_processed_data(train, val, test, full_df):
    """Save processed datasets to CSV files"""
    print("\n" )
    print("STEP 8: SAVING PROCESSED DATA")
    print("\n")
    print("\n")
    
    print("\n Saving datasets to CSV files...")
    
    # Save train set
    train_path = f'{PROCESSED_DATA_DIR}/train_processed.csv'
    train.to_csv(train_path, index=False)
    print(f"   Saved train set: {train_path}")
    print(f"    Size: {train.shape[0]:,} rows, {train.shape[1]} columns")
    
    # Save validation set
    val_path = f'{PROCESSED_DATA_DIR}/val_processed.csv'
    val.to_csv(val_path, index=False)
    print(f"  ✓ Saved validation set: {val_path}")
    print(f"    Size: {val.shape[0]:,} rows, {val.shape[1]} columns")
    
    # Save test set
    test_path = f'{PROCESSED_DATA_DIR}/test_processed.csv'
    test.to_csv(test_path, index=False)
    print(f"   Saved test set: {test_path}")
    print(f"    Size: {test.shape[0]:,} rows, {test.shape[1]} columns")
    
    # Save full processed data
    full_path = f'{PROCESSED_DATA_DIR}/full_processed.csv'
    full_df.to_csv(full_path, index=False)
    print(f"   Saved full dataset: {full_path}")
    print(f"    Size: {full_df.shape[0]:,} rows, {full_df.shape[1]} columns")
    
    print(f"\n All files saved to: {PROCESSED_DATA_DIR}/")
    
    # Show column list
    print(f"\n Final columns in processed data ({full_df.shape[1]} total):")
    for i, col in enumerate(full_df.columns, 1):
        print(f"    {i:2d}. {col}")



# MAIN EXECUTION FUNCTION


def main():
    """Main execution pipeline"""
    print("\n")
    print("STARTING DATA PROCESSING PIPELINE")
    print("WITH SRI LANKAN HOLIDAYS")  # ← CHANGED: Updated header
    print("\n")
    
    start_time = datetime.now()
    
    try:
        # Step 1: Load raw data
        train, items, stores, holidays = load_raw_data()
        
        # Step 2: Filter to selected stores and items
        filtered = filter_data(train)
        
        # Step 3: Add temporal features
        filtered = add_temporal_features(filtered)
        
        # Step 4: Add Sri Lankan holiday features  # ← CHANGED: Function name
        filtered = add_sri_lankan_holiday_features(filtered, holidays)
        
        # Step 5: Clean promotion data
        filtered = clean_promotions(filtered)
        
        # Step 6: Add product metadata
        filtered = add_product_metadata(filtered, items)
        
        # Step 7: Split into train/val/test
        train_df, val_df, test_df = split_data(filtered)
        
        # Step 8: Save processed data
        save_processed_data(train_df, val_df, test_df, filtered)
        
        # Calculate total time
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print("\n" + "="*60)
        print("DATA PROCESSING COMPLETE!")
        print("="*60)
        print(f"\n✓ Total time: {duration/60:.1f} minutes")
        print(f"✓ Processed data saved to: {PROCESSED_DATA_DIR}/")
        print(f"✓ Sri Lankan holidays integrated successfully!")  # ← CHANGED: Updated message
        print(f"✓ Ready for model training!")
        
        print("\n" )
        print("NEXT STEPS:")
        print("="*60)
        print("\n1. Re-run all 4 models with new holiday features")
        print("2. Compare performance (expect improvement in meat/poultry)")
        print("3. Proceed to dashboard development")
        print("="*60 )
        
    except Exception as e:
        print(f"\n ERROR: {str(e)}")
        print("Processing failed. Please check error above.")
        raise



# RUN THE PIPELINE


if __name__ == "__main__":
    main()