"""
CREATE SRI LANKAN HOLIDAYS CSV (2013-2017)
==========================================
Generates comprehensive holiday calendar with:
- All 12 monthly Poya days (60 total)
- Fixed holidays (Independence, New Year, Christmas, etc.)
- Variable holidays (Eid, Deepavali, Good Friday)
- Multi-day shopping patterns (pre-New Year, pre-Eid, etc.)

Total: ~180-200 holiday entries
"""

import pandas as pd
from datetime import datetime, timedelta

print("\n")
print("CREATING SRI LANKAN HOLIDAY CALENDAR (2013-2017)")
print("\n" )

holidays = []

# ============================================================================
# FIXED HOLIDAYS
# ============================================================================

print("\n Adding fixed holidays...")

# THAI PONGAL (January 14-15 every year)
for year in [2013, 2014, 2015, 2016, 2017]:
    holidays.extend([
        (f'{year}-01-10', 'Thai Pongal Pre-Shopping', 0, 0, 'Rice/flour shopping starts'),
        (f'{year}-01-11', 'Thai Pongal Pre-Shopping', 0, 0, 'Rice/flour continues'),
        (f'{year}-01-12', 'Thai Pongal Pre-Shopping', 0, 0, 'Rice/flour + milk starts'),
        (f'{year}-01-13', 'Pre Thai Pongal', +1, 0, 'PEAK: Rice/milk/veg/banana'),
        (f'{year}-01-14', 'Thai Pongal Day 1', 0, 0, 'Milk continues'),
        (f'{year}-01-15', 'Thai Pongal Day 2', 0, 0, 'Recovery'),
    ])

# INDEPENDENCE DAY (February 4)
for year in [2013, 2014, 2015, 2016, 2017]:
    holidays.append((f'{year}-02-04', 'Independence Day', 0, 1, 'Meat not sold'))

# SINHALA & TAMIL NEW YEAR (April 13-14 + shopping days)
for year in [2013, 2014, 2015, 2016, 2017]:
    holidays.extend([
        (f'{year}-04-06', 'Pre New Year Shopping', 0, 0, 'Shopping starts'),
        (f'{year}-04-07', 'Pre New Year Shopping', 0, 0, 'Shopping continues'),
        (f'{year}-04-08', 'Pre New Year Shopping', +1, 0, 'Shopping accelerates'),
        (f'{year}-04-09', 'Pre New Year Shopping', +1, 0, 'Shopping continues'),
        (f'{year}-04-10', 'Pre New Year Peak', +1, 0, 'PEAK: Rice/flour dominant'),
        (f'{year}-04-11', 'Pre New Year Peak', +1, 0, 'PEAK continues'),
        (f'{year}-04-12', 'Pre New Year Peak', +1, 0, 'Last shopping day'),
        (f'{year}-04-13', 'New Year Day 1', -1, 0, 'Shops CLOSED'),
        (f'{year}-04-14', 'New Year Day 2', -1, 0, 'Shops CLOSED'),
        (f'{year}-04-15', 'Post New Year', 0, 0, 'Recovery'),
    ])

# MAY DAY (May 1)
for year in [2013, 2014, 2015, 2016, 2017]:
    holidays.append((f'{year}-05-01', 'May Day', 0, 0, 'Government holiday'))

# CHRISTMAS (December 22-25)
for year in [2013, 2014, 2015, 2016, 2017]:
    holidays.extend([
        (f'{year}-12-22', 'Pre Christmas', +1, 0, 'Shopping spike'),
        (f'{year}-12-23', 'Pre Christmas', +1, 0, 'Shopping peak'),
        (f'{year}-12-24', 'Pre Christmas', +1, 0, 'Last shopping day'),
        (f'{year}-12-25', 'Christmas', -1, 0, 'Shops CLOSED'),
    ])

print(f"   Added {len([h for h in holidays if 'Pongal' in h[1] or 'Independence' in h[1] or 'New Year' in h[1] or 'May Day' in h[1] or 'Christmas' in h[1]])} fixed holiday entries")

# ============================================================================
# POYA DAYS (Full Moon Days - 12 per year)
# ============================================================================

print("\n Adding Poya days (full moon holidays)...")

# Exact full moon dates for Sri Lankan Poya days (researched)
poya_dates = {
    # 2013
    2013: [
        ('2013-01-27', 'Duruthu Poya'),
        ('2013-02-25', 'Navam Poya'),
        ('2013-03-27', 'Medin Poya'),
        ('2013-04-25', 'Bak Poya'),
        ('2013-05-24', 'Vesak Poya'),  # MAJOR
        ('2013-06-23', 'Poson Poya'),  # MAJOR
        ('2013-07-22', 'Esala Poya'),
        ('2013-08-20', 'Nikini Poya'),
        ('2013-09-19', 'Binara Poya'),
        ('2013-10-18', 'Vap Poya'),
        ('2013-11-16', 'Il Poya'),
        ('2013-12-16', 'Unduvap Poya'),
    ],
    # 2014
    2014: [
        ('2014-01-15', 'Duruthu Poya'),
        ('2014-02-14', 'Navam Poya'),
        ('2014-03-16', 'Medin Poya'),
        ('2014-04-15', 'Bak Poya'),
        ('2014-05-13', 'Vesak Poya'),  # MAJOR
        ('2014-06-11', 'Poson Poya'),  # MAJOR
        ('2014-07-11', 'Esala Poya'),
        ('2014-08-09', 'Nikini Poya'),
        ('2014-09-08', 'Binara Poya'),
        ('2014-10-08', 'Vap Poya'),
        ('2014-11-06', 'Il Poya'),
        ('2014-12-05', 'Unduvap Poya'),
    ],
    # 2015
    2015: [
        ('2015-01-04', 'Duruthu Poya'),
        ('2015-02-03', 'Navam Poya'),
        ('2015-03-05', 'Medin Poya'),
        ('2015-04-04', 'Bak Poya'),
        ('2015-05-04', 'Vesak Poya'),  # MAJOR
        ('2015-06-02', 'Poson Poya'),  # MAJOR
        ('2015-07-01', 'Esala Poya'),
        ('2015-07-30', 'Nikini Poya'),
        ('2015-08-29', 'Binara Poya'),
        ('2015-09-27', 'Vap Poya'),
        ('2015-10-27', 'Il Poya'),
        ('2015-11-25', 'Unduvap Poya'),
    ],
    # 2016
    2016: [
        ('2016-01-23', 'Duruthu Poya'),
        ('2016-02-22', 'Navam Poya'),
        ('2016-03-23', 'Medin Poya'),
        ('2016-04-21', 'Bak Poya'),
        ('2016-05-21', 'Vesak Poya'),  # MAJOR
        ('2016-06-20', 'Poson Poya'),  # MAJOR
        ('2016-07-19', 'Esala Poya'),
        ('2016-08-17', 'Nikini Poya'),
        ('2016-09-16', 'Binara Poya'),
        ('2016-10-15', 'Vap Poya'),
        ('2016-11-14', 'Il Poya'),
        ('2016-12-13', 'Unduvap Poya'),
    ],
    # 2017
    2017: [
        ('2017-01-12', 'Duruthu Poya'),
        ('2017-02-10', 'Navam Poya'),
        ('2017-03-12', 'Medin Poya'),
        ('2017-04-10', 'Bak Poya'),
        ('2017-05-10', 'Vesak Poya'),  # MAJOR
        ('2017-06-08', 'Poson Poya'),  # MAJOR
        ('2017-07-08', 'Esala Poya'),
        ('2017-08-07', 'Nikini Poya'),
        ('2017-09-05', 'Binara Poya'),
        ('2017-10-05', 'Vap Poya'),
        ('2017-11-03', 'Il Poya'),
        ('2017-12-03', 'Unduvap Poya'),
    ],
}

# Add regular Poya days (meat restricted, shops open)
for year, poyas in poya_dates.items():
    for date, name in poyas:
        if 'Vesak' in name:
            # Vesak is MAJOR - shops closed
            # Day before: shopping spike
            date_obj = datetime.strptime(date, '%Y-%m-%d')
            day_before = (date_obj - timedelta(days=1)).strftime('%Y-%m-%d')
            day_after = (date_obj + timedelta(days=1)).strftime('%Y-%m-%d')
            
            holidays.extend([
                (day_before, 'Pre Vesak', +1, 0, 'Shopping spike before Vesak'),
                (date, name, -1, 0, 'Shops CLOSED - Major Poya'),
                (day_after, 'Post Vesak', 0, 0, 'Recovery'),
            ])
        elif 'Poson' in name:
            # Poson is MAJOR - shops closed
            date_obj = datetime.strptime(date, '%Y-%m-%d')
            day_before = (date_obj - timedelta(days=1)).strftime('%Y-%m-%d')
            day_after = (date_obj + timedelta(days=1)).strftime('%Y-%m-%d')
            
            holidays.extend([
                (day_before, 'Pre Poson', +1, 0, 'Shopping spike before Poson'),
                (date, name, -1, 0, 'Shops CLOSED - Major Poya'),
                (day_after, 'Post Poson', 0, 0, 'Recovery'),
            ])
        else:
            # Regular Poya - meat restricted, shops open
            holidays.append((date, name, 0, 1, 'Meat restricted - Regular Poya'))

print(f"   Added {len([h for h in holidays if 'Poya' in h[1]])} Poya entries")

# ============================================================================
# GOOD FRIDAY (Variable - before Easter)
# ============================================================================

print("\n Adding Good Friday...")

good_friday_dates = [
    '2013-03-29',
    '2014-04-18',
    '2015-04-03',
    '2016-03-25',
    '2017-04-14',
]

for date in good_friday_dates:
    holidays.append((date, 'Good Friday', 0, 0, 'Minor reduction - small Christian population'))

print(f"   Added {len(good_friday_dates)} Good Friday entries")

# ============================================================================
# DEEPAVALI (Variable - October/November)
# ============================================================================

print("\n Adding Deepavali...")

deepavali_dates = {
    2013: '2013-11-03',
    2014: '2014-10-23',
    2015: '2015-11-11',
    2016: '2016-10-30',
    2017: '2017-10-19',
}

for year, date in deepavali_dates.items():
    date_obj = datetime.strptime(date, '%Y-%m-%d')
    day_before = (date_obj - timedelta(days=1)).strftime('%Y-%m-%d')
    
    holidays.extend([
        (day_before, 'Pre Deepavali', +1, 0, 'Milk/rice/flour/beverages spike'),
        (date, 'Deepavali', 0, 0, 'Tamil shops closed, others open'),
    ])

print(f"   Added {len([h for h in holidays if 'Deepavali' in h[1]])} Deepavali entries")

# ============================================================================
# EID AL-FITR (End of Ramadan - Variable)
# ============================================================================

print("\n Adding Eid al-Fitr...")

eid_fitr_dates = {
    2013: '2013-08-08',
    2014: '2014-07-28',
    2015: '2015-07-17',
    2016: '2016-07-06',
    2017: '2017-06-25',
}

for year, date in eid_fitr_dates.items():
    date_obj = datetime.strptime(date, '%Y-%m-%d')
    
    # Multi-day shopping pattern
    day_minus_4 = (date_obj - timedelta(days=4)).strftime('%Y-%m-%d')
    day_minus_3 = (date_obj - timedelta(days=3)).strftime('%Y-%m-%d')
    day_minus_2 = (date_obj - timedelta(days=2)).strftime('%Y-%m-%d')
    day_minus_1 = (date_obj - timedelta(days=1)).strftime('%Y-%m-%d')
    day_plus_1 = (date_obj + timedelta(days=1)).strftime('%Y-%m-%d')
    day_plus_2 = (date_obj + timedelta(days=2)).strftime('%Y-%m-%d')
    
    holidays.extend([
        (day_minus_4, 'Pre Eid Fitr Shopping', +1, 0, 'Meat/rice shopping starts'),
        (day_minus_3, 'Pre Eid Fitr Shopping', +1, 0, 'Shopping accelerates'),
        (day_minus_2, 'Pre Eid Fitr Peak', +1, 0, 'PEAK shopping'),
        (day_minus_1, 'Pre Eid Fitr Peak', +1, 0, 'Last minute rush'),
        (date, 'Eid al-Fitr Day 1', 0, 0, 'Muslim shops closed'),
        (day_plus_1, 'Eid al-Fitr Day 2', 0, 0, 'Celebrations continue'),
        (day_plus_2, 'Eid al-Fitr Day 3', 0, 0, 'Recovery'),
    ])

print(f"   Added {len([h for h in holidays if 'Eid Fitr' in h[1]])} Eid al-Fitr entries")

# ============================================================================
# EID AL-ADHA (Qurbani/Hajj - Variable)
# ============================================================================

print("\n Adding Eid al-Adha...")

eid_adha_dates = {
    2013: '2013-10-15',
    2014: '2014-10-04',
    2015: '2015-09-24',
    2016: '2016-09-12',
    2017: '2017-09-01',
}

for year, date in eid_adha_dates.items():
    date_obj = datetime.strptime(date, '%Y-%m-%d')
    
    # Multi-day pattern (Qurbani meat focus)
    day_minus_4 = (date_obj - timedelta(days=4)).strftime('%Y-%m-%d')
    day_minus_3 = (date_obj - timedelta(days=3)).strftime('%Y-%m-%d')
    day_minus_2 = (date_obj - timedelta(days=2)).strftime('%Y-%m-%d')
    day_minus_1 = (date_obj - timedelta(days=1)).strftime('%Y-%m-%d')
    day_plus_1 = (date_obj + timedelta(days=1)).strftime('%Y-%m-%d')
    day_plus_2 = (date_obj + timedelta(days=2)).strftime('%Y-%m-%d')
    
    holidays.extend([
        (day_minus_4, 'Pre Eid Adha Shopping', +1, 0, 'Shopping starts'),
        (day_minus_3, 'Pre Eid Adha Shopping', +1, 0, 'Meat shopping accelerates'),
        (day_minus_2, 'Pre Eid Adha Peak', +1, 0, 'Heavy meat shopping for Qurbani'),
        (day_minus_1, 'Pre Eid Adha Peak', +1, 0, 'PEAK: Animal purchases'),
        (date, 'Eid al-Adha Day 1', +1, 0, 'Qurbani meat distribution'),
        (day_plus_1, 'Eid al-Adha Day 2', 0, 0, 'Celebrations'),
        (day_plus_2, 'Eid al-Adha Day 3', 0, 0, 'Recovery'),
    ])

print(f"  Added {len([h for h in holidays if 'Eid Adha' in h[1]])} Eid al-Adha entries")

# ============================================================================
# CREATE DATAFRAME AND SAVE
# ============================================================================

print("\n" + "="*70)
print("FINALIZING HOLIDAY CSV")
print("="*70)

# Convert to DataFrame
df = pd.DataFrame(holidays, columns=['date', 'holiday_name', 'general_impact', 'meat_restriction', 'notes'])

# Convert date to datetime
df['date'] = pd.to_datetime(df['date'])

# Sort by date
df = df.sort_values('date').reset_index(drop=True)

# Save to CSV
output_file = '../data/holidays/sri_lanka_holidays.csv'
df.to_csv(output_file, index=False)

print(f"\n Created: {output_file}")
print(f"  Total entries: {len(df)}")
print(f"  Date range: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")

# Summary statistics
print(f"\n SUMMARY:")
print(f"  Spike days (general_impact=+1): {len(df[df['general_impact']==1])}")
print(f"  Closure days (general_impact=-1): {len(df[df['general_impact']==-1])}")
print(f"  Meat restriction days: {len(df[df['meat_restriction']==1])}")
print(f"  Neutral days: {len(df[df['general_impact']==0])}")

# Show sample
print(f"\n SAMPLE ENTRIES:")
print(df.head(20).to_string(index=False))

print("\n" + "="*70)
print("✓ SRI LANKAN HOLIDAY CSV COMPLETE!")
print("\n" )
print("\n✓ Next steps:")
print("  1. Update data_processing.py to use this CSV")
print("  2. Re-run data_processing.py")
print("  3. Re-run all 4 models")
print("  4. Compare results!")
print( "\n")