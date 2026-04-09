"""
CREATE SRI LANKAN HOLIDAYS CSV (2013-2017)
==========================================
Official holiday dates sourced programmatically via the Python 'holidays'
library (holidays.LK), which draws from government-designated public holiday
records for Sri Lanka.

Domain knowledge features (pre-shopping windows, shop closure impacts,
meat restrictions, impact scores) are encoded on top of the official dates
based on Sri Lankan retail consumer behaviour patterns.

Output columns:
    date             - holiday date
    holiday_name     - descriptive name
    general_impact   - shop impact: +1 spike, 0 neutral, -1 closure
    meat_restriction - 1 if meat sales restricted (Poya days, Independence Day)
    notes            - explanation of encoding decision

Total entries: ~270-290 (official dates + pre/post shopping windows)
"""

import pandas as pd
import holidays
from datetime import datetime, timedelta
import os

print("\n")
print("CREATING SRI LANKAN HOLIDAY CALENDAR (2013-2017)")
print("Date source: Python holidays library (holidays.LK)")
print("\n")


# CONFIGURATION

YEARS = [2013, 2014, 2015, 2016, 2017]
OUTPUT_DIR = '../data/holidays'
OUTPUT_FILE = f'{OUTPUT_DIR}/sri_lanka_holidays.csv'

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load official Sri Lankan holidays from library
lk_official = holidays.LK(years=YEARS)

print(f"Official holidays loaded from holidays.LK: {len(lk_official)} entries")
print("These cover: Poya days, Eid, Deepavali, New Year, Christmas,")
print("Good Friday, Independence Day, May Day, Thai Pongal, Maha Sivarathri,")
print("Prophet's Birthday, and more.\n")


# HELPER FUNCTION

def offset(date_str, days):
    """Return date string offset by N days from a given date string."""
    d = datetime.strptime(date_str, '%Y-%m-%d')
    return (d + timedelta(days=days)).strftime('%Y-%m-%d')


# BUILD HOLIDAY ENTRIES

entries = []  # Each entry: (date_str, name, general_impact, meat_restriction, notes)


# STEP 1: POYA DAYS
# Sourced from holidays.LK — all 12 monthly full moon Poya days per year.
# Domain knowledge applied:
#   - Regular Poya: shops open but meat restricted
#   - Vesak & Poson (major Poya): shops closed, pre-day shopping spike
#   - All Poya days: meat_restriction = 1

print("Processing Poya days...")

poya_dates = {}  # year -> list of (date_str, name)

for date, name in sorted(lk_official.items()):
    if 'Poya' in name:
        year = date.year
        date_str = date.strftime('%Y-%m-%d')
        if year not in poya_dates:
            poya_dates[year] = []
        poya_dates[year].append((date_str, name))

for year, poyas in poya_dates.items():
    for date_str, name in poyas:

        if 'Vesak' in name:
            # Major Poya — shops fully closed
            # Day before: shopping spike as people stock up
            entries.append((offset(date_str, -1), 'Pre Vesak Shopping', +1, 0,
                            'Shopping spike — people stock up before Vesak closure'))
            entries.append((date_str, name, -1, 1,
                            'Shops CLOSED — major Poya, meat fully restricted'))
            # Day Following Vesak is already in holidays.LK — handled below
            entries.append((offset(date_str, +1), 'Post Vesak Recovery', 0, 0,
                            'Partial recovery day after Vesak'))

        elif 'Poson' in name:
            # Major Poya — shops fully closed
            entries.append((offset(date_str, -1), 'Pre Poson Shopping', +1, 0,
                            'Shopping spike before Poson closure'))
            entries.append((date_str, name, -1, 1,
                            'Shops CLOSED — major Poya, meat fully restricted'))
            entries.append((offset(date_str, +1), 'Post Poson Recovery', 0, 0,
                            'Recovery day after Poson'))

        elif 'Day Following Vesak' in name:
            # Already added as Post Vesak Recovery above — skip to avoid duplicate
            pass

        elif 'Adhi Esala' in name or 'Esala' in name:
            # Regular Poya — shops open, meat restricted
            entries.append((date_str, name, 0, 1,
                            'Regular Poya — shops open, meat sales restricted'))

        else:
            # All other regular Poya days
            entries.append((date_str, name, 0, 1,
                            'Regular Poya — shops open, meat sales restricted'))

print(f"  Poya entries added: {len([e for e in entries if 'Poya' in e[1] or 'Vesak' in e[1] or 'Poson' in e[1]])}")


# STEP 2: SINHALA & TAMIL NEW YEAR (April 13-14)
# Sourced from holidays.LK for exact dates.
# Domain knowledge: major 7-day pre-shopping window for rice, flour, milk,
# vegetables, bananas. Shops closed on New Year days themselves.

print("Processing Sinhala & Tamil New Year...")

new_year_count = 0
for date, name in sorted(lk_official.items()):
    if 'Sinhala and Tamil New Year' in name and 'Day Before' not in name:
        date_str = date.strftime('%Y-%m-%d')

        # Pre-shopping window: 7 days before (gradual ramp up)
        entries.append((offset(date_str, -7), 'Pre New Year Shopping', 0, 0,
                        'Rice/flour shopping begins'))
        entries.append((offset(date_str, -6), 'Pre New Year Shopping', 0, 0,
                        'Shopping continues'))
        entries.append((offset(date_str, -5), 'Pre New Year Shopping', +1, 0,
                        'Shopping accelerates'))
        entries.append((offset(date_str, -4), 'Pre New Year Shopping', +1, 0,
                        'Shopping continues'))
        entries.append((offset(date_str, -3), 'Pre New Year Peak', +1, 0,
                        'PEAK: Rice/flour dominant'))
        entries.append((offset(date_str, -2), 'Pre New Year Peak', +1, 0,
                        'PEAK continues'))
        entries.append((offset(date_str, -1), 'Pre New Year Peak', +1, 0,
                        'Last shopping day before closure'))

        # New Year Day 1 (April 14) — shops closed
        entries.append((date_str, 'Sinhala & Tamil New Year Day 1', -1, 0,
                        'Shops CLOSED — New Year Day 1'))

        # New Year Day 2 (April 15) — shops closed
        entries.append((offset(date_str, +1), 'Sinhala & Tamil New Year Day 2', -1, 0,
                        'Shops CLOSED — New Year Day 2'))

        # Recovery
        entries.append((offset(date_str, +2), 'Post New Year Recovery', 0, 0,
                        'Gradual return to normal'))

        new_year_count += 1

print(f"  New Year entries added: {new_year_count * 10} (across {new_year_count} years)")


# STEP 3: THAI PONGAL (January 14-15)
# Sourced from holidays.LK.
# Domain knowledge: rice, milk, vegetables, banana shopping spike in days before.

print("Processing Thai Pongal...")

pongal_count = 0
for date, name in sorted(lk_official.items()):
    if 'Thai Pongal' in name:
        date_str = date.strftime('%Y-%m-%d')

        entries.append((offset(date_str, -4), 'Pre Thai Pongal Shopping', 0, 0,
                        'Rice/flour shopping starts'))
        entries.append((offset(date_str, -3), 'Pre Thai Pongal Shopping', 0, 0,
                        'Rice/flour continues'))
        entries.append((offset(date_str, -2), 'Pre Thai Pongal Shopping', 0, 0,
                        'Rice/flour + milk starts'))
        entries.append((offset(date_str, -1), 'Pre Thai Pongal Peak', +1, 0,
                        'PEAK: Rice/milk/veg/banana'))
        entries.append((date_str, 'Thai Pongal Day 1', 0, 0,
                        'Celebrations — moderate shopping'))
        entries.append((offset(date_str, +1), 'Thai Pongal Day 2', 0, 0,
                        'Recovery'))

        pongal_count += 1

print(f"  Thai Pongal entries added: {pongal_count * 6} (across {pongal_count} years)")


# STEP 4: INDEPENDENCE DAY (February 4)
# Sourced from holidays.LK.
# Domain knowledge: meat not sold — national observance day.

print("Processing Independence Day...")

ind_count = 0
for date, name in sorted(lk_official.items()):
    if 'Independence' in name:
        date_str = date.strftime('%Y-%m-%d')
        entries.append((date_str, 'Independence Day', 0, 1,
                        'National holiday — meat not sold by convention'))
        ind_count += 1

print(f"  Independence Day entries added: {ind_count}")


# STEP 5: EID AL-FITR
# Sourced from holidays.LK for exact variable dates each year.
# Domain knowledge: 4-day pre-shopping window (meat, rice focus),
# 3-day celebration/recovery period.

print("Processing Eid al-Fitr...")

eid_fitr_count = 0
for date, name in sorted(lk_official.items()):
    if 'Eid al-Fitr' in name:
        date_str = date.strftime('%Y-%m-%d')

        entries.append((offset(date_str, -4), 'Pre Eid Fitr Shopping', +1, 0,
                        'Meat/rice shopping starts'))
        entries.append((offset(date_str, -3), 'Pre Eid Fitr Shopping', +1, 0,
                        'Shopping accelerates'))
        entries.append((offset(date_str, -2), 'Pre Eid Fitr Peak', +1, 0,
                        'PEAK shopping'))
        entries.append((offset(date_str, -1), 'Pre Eid Fitr Peak', +1, 0,
                        'Last minute rush'))
        entries.append((date_str, 'Eid al-Fitr Day 1', 0, 0,
                        'Muslim shops closed, others open'))
        entries.append((offset(date_str, +1), 'Eid al-Fitr Day 2', 0, 0,
                        'Celebrations continue'))
        entries.append((offset(date_str, +2), 'Eid al-Fitr Day 3', 0, 0,
                        'Recovery'))

        eid_fitr_count += 1

print(f"  Eid al-Fitr entries added: {eid_fitr_count * 7} (across {eid_fitr_count} years)")


# STEP 6: EID AL-ADHA
# Sourced from holidays.LK for exact variable dates.
# Domain knowledge: Qurbani festival — heavy meat shopping in pre-days,
# meat distribution on day itself (positive impact for meat products).

print("Processing Eid al-Adha...")

eid_adha_count = 0
for date, name in sorted(lk_official.items()):
    if 'Eid al-Adha' in name:
        date_str = date.strftime('%Y-%m-%d')

        entries.append((offset(date_str, -4), 'Pre Eid Adha Shopping', +1, 0,
                        'Shopping starts'))
        entries.append((offset(date_str, -3), 'Pre Eid Adha Shopping', +1, 0,
                        'Meat shopping accelerates'))
        entries.append((offset(date_str, -2), 'Pre Eid Adha Peak', +1, 0,
                        'Heavy meat shopping for Qurbani'))
        entries.append((offset(date_str, -1), 'Pre Eid Adha Peak', +1, 0,
                        'PEAK: Last minute purchases'))
        entries.append((date_str, 'Eid al-Adha Day 1', +1, 0,
                        'Qurbani meat distribution — elevated sales'))
        entries.append((offset(date_str, +1), 'Eid al-Adha Day 2', 0, 0,
                        'Celebrations'))
        entries.append((offset(date_str, +2), 'Eid al-Adha Day 3', 0, 0,
                        'Recovery'))

        eid_adha_count += 1

print(f"  Eid al-Adha entries added: {eid_adha_count * 7} (across {eid_adha_count} years)")


# STEP 7: DEEPAVALI
# Sourced from holidays.LK for exact variable dates.
# Domain knowledge: milk, rice, flour, beverages spike day before.

print("Processing Deepavali...")

deepavali_count = 0
for date, name in sorted(lk_official.items()):
    if 'Deepavali' in name:
        date_str = date.strftime('%Y-%m-%d')

        entries.append((offset(date_str, -1), 'Pre Deepavali Shopping', +1, 0,
                        'Milk/rice/flour/beverages spike'))
        entries.append((date_str, 'Deepavali Festival Day', 0, 0,
                        'Tamil shops closed, others open'))

        deepavali_count += 1

print(f"  Deepavali entries added: {deepavali_count * 2} (across {deepavali_count} years)")


# STEP 8: CHRISTMAS
# Sourced from holidays.LK.
# Domain knowledge: 3-day pre-shopping spike, shops closed on Dec 25.

print("Processing Christmas...")

christmas_count = 0
for date, name in sorted(lk_official.items()):
    if 'Christmas' in name:
        date_str = date.strftime('%Y-%m-%d')

        entries.append((offset(date_str, -3), 'Pre Christmas Shopping', +1, 0,
                        'Shopping spike'))
        entries.append((offset(date_str, -2), 'Pre Christmas Shopping', +1, 0,
                        'Shopping peak'))
        entries.append((offset(date_str, -1), 'Pre Christmas Peak', +1, 0,
                        'Last shopping day'))
        entries.append((date_str, 'Christmas Day', -1, 0,
                        'Shops CLOSED'))

        christmas_count += 1

print(f"  Christmas entries added: {christmas_count * 4} (across {christmas_count} years)")


# STEP 9: GOOD FRIDAY
# Sourced from holidays.LK for exact variable dates.
# Domain knowledge: minor reduction — small Christian population in Sri Lanka.

print("Processing Good Friday...")

gf_count = 0
for date, name in sorted(lk_official.items()):
    if 'Good Friday' in name:
        date_str = date.strftime('%Y-%m-%d')
        entries.append((date_str, 'Good Friday', 0, 0,
                        'Minor reduction — small Christian population'))
        gf_count += 1

print(f"  Good Friday entries added: {gf_count}")


# STEP 10: MAY DAY
# Sourced from holidays.LK.
# Domain knowledge: government/bank holiday, retail largely open.

print("Processing May Day...")

may_count = 0
for date, name in sorted(lk_official.items()):
    if "Workers" in name or "May Day" in name:
        date_str = date.strftime('%Y-%m-%d')
        entries.append((date_str, "International Workers' Day", 0, 0,
                        'Government holiday — retail largely open'))
        may_count += 1

print(f"  May Day entries added: {may_count}")


# STEP 11: MAHA SIVARATHRI
# Sourced from holidays.LK — not in original manual script (bonus coverage).
# Domain knowledge: Hindu observance, minor impact on general retail.

print("Processing Maha Sivarathri...")

siva_count = 0
for date, name in sorted(lk_official.items()):
    if 'Sivarathri' in name or 'Shivaratri' in name:
        date_str = date.strftime('%Y-%m-%d')
        entries.append((date_str, 'Maha Sivarathri Day', 0, 0,
                        'Hindu observance — minor impact on general retail'))
        siva_count += 1

print(f"  Maha Sivarathri entries added: {siva_count}")


# STEP 12: PROPHET'S BIRTHDAY
# Sourced from holidays.LK — not in original manual script (bonus coverage).
# Domain knowledge: Muslim observance, minor impact on general retail.

print("Processing Prophet's Birthday...")

prophet_count = 0
for date, name in sorted(lk_official.items()):
    if "Prophet" in name:
        date_str = date.strftime('%Y-%m-%d')
        entries.append((date_str, "Prophet's Birthday", 0, 0,
                        "Muslim observance — minor impact on general retail"))
        prophet_count += 1

print(f"  Prophet's Birthday entries added: {prophet_count}")


# BUILD DATAFRAME

print("\nBuilding final DataFrame...")

df = pd.DataFrame(entries, columns=[
    'date', 'holiday_name', 'general_impact', 'meat_restriction', 'notes'
])

df['date'] = pd.to_datetime(df['date'])

# Remove any duplicates (e.g. if pre/post windows overlap with official dates)
df = df.drop_duplicates(subset=['date', 'holiday_name']).sort_values('date').reset_index(drop=True)

# Ensure correct data types
df['general_impact'] = df['general_impact'].astype('int8')
df['meat_restriction'] = df['meat_restriction'].astype('int8')


# SAVE

df.to_csv(OUTPUT_FILE, index=False)

print(f"\nSaved: {OUTPUT_FILE}")


# SUMMARY REPORT

print("\n" + "="*60)
print("HOLIDAY CALENDAR COMPLETE")
print("="*60)
print(f"Total entries:          {len(df)}")
print(f"Date range:             {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")
print(f"\nImpact breakdown:")
print(f"  Shopping spike  (+1): {len(df[df['general_impact']==1])}")
print(f"  Shop closure    (-1): {len(df[df['general_impact']==-1])}")
print(f"  Neutral          (0): {len(df[df['general_impact']==0])}")
print(f"  Meat restriction (1): {len(df[df['meat_restriction']==1])}")
print(f"\nHoliday types covered:")
print(f"  Poya days (all 12/year):    60 official dates")
print(f"  Sinhala & Tamil New Year:    {new_year_count * 10} entries (inc. pre/post windows)")
print(f"  Thai Pongal:                 {pongal_count * 6} entries")
print(f"  Eid al-Fitr:                 {eid_fitr_count * 7} entries")
print(f"  Eid al-Adha:                 {eid_adha_count * 7} entries")
print(f"  Deepavali:                   {deepavali_count * 2} entries")
print(f"  Christmas:                   {christmas_count * 4} entries")
print(f"  Independence Day:            {ind_count} entries")
print(f"  Good Friday:                 {gf_count} entries")
print(f"  May Day:                     {may_count} entries")
print(f"  Maha Sivarathri (NEW):       {siva_count} entries")
print(f"  Prophet's Birthday (NEW):    {prophet_count} entries")
print(f"\nDate source: Python holidays library (holidays.LK)")
print(f"Domain knowledge: Pre/post shopping windows, impact scores,")
print(f"  meat restrictions — encoded based on Sri Lankan retail behaviour")
print("="*60)
print("\nNext step: rerun data_processing.py")
print("="*60)