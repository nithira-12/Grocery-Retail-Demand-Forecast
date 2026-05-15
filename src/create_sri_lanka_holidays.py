import pandas as pd
import holidays
from datetime import datetime, timedelta
import os

YEARS = [2013, 2014, 2015, 2016, 2017]
OUTPUT_DIR = '../data/holidays'
OUTPUT_FILE = f'{OUTPUT_DIR}/sri_lanka_holidays.csv'

os.makedirs(OUTPUT_DIR, exist_ok=True)

lk_official = holidays.LK(years=YEARS)


def offset(date_str, days):
    d = datetime.strptime(date_str, '%Y-%m-%d')
    return (d + timedelta(days=days)).strftime('%Y-%m-%d')


entries = []

# POYA DAYS 
# Regular Poya: shops open, meat restricted.
# Vesak and Poson (major Poya): shops closed, pre-day shopping spike.

poya_dates = {}

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
            entries.append((offset(date_str, -1), 'Pre Vesak Shopping', +1, 0,
                            'Shopping spike — people stock up before Vesak closure'))
            entries.append((date_str, name, -1, 1,
                            'Shops CLOSED — major Poya, meat fully restricted'))
            entries.append((offset(date_str, +1), 'Post Vesak Recovery', 0, 0,
                            'Partial recovery day after Vesak'))

        elif 'Poson' in name:
            entries.append((offset(date_str, -1), 'Pre Poson Shopping', +1, 0,
                            'Shopping spike before Poson closure'))
            entries.append((date_str, name, -1, 1,
                            'Shops CLOSED — major Poya, meat fully restricted'))
            entries.append((offset(date_str, +1), 'Post Poson Recovery', 0, 0,
                            'Recovery day after Poson'))

        elif 'Day Following Vesak' in name:
           
            pass

        else:
            entries.append((date_str, name, 0, 1,
                            'Regular Poya — shops open, meat sales restricted'))


# SINHALA & TAMIL NEW YEAR (April 13-14) 
# 7-day pre-shopping window for rice, flour, milk, vegetables, bananas.
# Shops closed on New Year days themselves.

for date, name in sorted(lk_official.items()):
    if 'Sinhala and Tamil New Year' in name and 'Day Before' not in name:
        date_str = date.strftime('%Y-%m-%d')

        entries.append((offset(date_str, -7), 'Pre New Year Shopping', 0, 0, 'Rice/flour shopping begins'))
        entries.append((offset(date_str, -6), 'Pre New Year Shopping', 0, 0, 'Shopping continues'))
        entries.append((offset(date_str, -5), 'Pre New Year Shopping', +1, 0, 'Shopping accelerates'))
        entries.append((offset(date_str, -4), 'Pre New Year Shopping', +1, 0, 'Shopping continues'))
        entries.append((offset(date_str, -3), 'Pre New Year Peak', +1, 0, 'PEAK: Rice/flour dominant'))
        entries.append((offset(date_str, -2), 'Pre New Year Peak', +1, 0, 'PEAK continues'))
        entries.append((offset(date_str, -1), 'Pre New Year Peak', +1, 0, 'Last shopping day before closure'))
        entries.append((date_str, 'Sinhala & Tamil New Year Day 1', -1, 0, 'Shops CLOSED — New Year Day 1'))
        entries.append((offset(date_str, +1), 'Sinhala & Tamil New Year Day 2', -1, 0, 'Shops CLOSED — New Year Day 2'))
        entries.append((offset(date_str, +2), 'Post New Year Recovery', 0, 0, 'Gradual return to normal'))


# THAI PONGAL (January 14-15) 
# Rice, milk, vegetablesspike in days before.

for date, name in sorted(lk_official.items()):
    if 'Thai Pongal' in name:
        date_str = date.strftime('%Y-%m-%d')

        entries.append((offset(date_str, -4), 'Pre Thai Pongal Shopping', 0, 0, 'Rice/flour shopping starts'))
        entries.append((offset(date_str, -3), 'Pre Thai Pongal Shopping', 0, 0, 'Rice/flour continues'))
        entries.append((offset(date_str, -2), 'Pre Thai Pongal Shopping', 0, 0, 'Rice/flour + milk starts'))
        entries.append((offset(date_str, -1), 'Pre Thai Pongal Peak', +1, 0, 'PEAK: Rice/milk/veg/banana'))
        entries.append((date_str, 'Thai Pongal Day 1', 0, 0, 'Celebrations — moderate shopping'))
        entries.append((offset(date_str, +1), 'Thai Pongal Day 2', 0, 0, 'Recovery'))


# INDEPENDENCE DAY (February 4) 
# Meat not sold 

for date, name in sorted(lk_official.items()):
    if 'Independence' in name:
        date_str = date.strftime('%Y-%m-%d')
        entries.append((date_str, 'Independence Day', 0, 1,
                        'National holiday — meat not sold by convention'))


# EID AL-FITR 


for date, name in sorted(lk_official.items()):
    if 'Eid al-Fitr' in name:
        date_str = date.strftime('%Y-%m-%d')

        entries.append((offset(date_str, -4), 'Pre Eid Fitr Shopping', +1, 0, 'Meat/rice shopping starts'))
        entries.append((offset(date_str, -3), 'Pre Eid Fitr Shopping', +1, 0, 'Shopping accelerates'))
        entries.append((offset(date_str, -2), 'Pre Eid Fitr Peak', +1, 0, 'PEAK shopping'))
        entries.append((offset(date_str, -1), 'Pre Eid Fitr Peak', +1, 0, 'Last minute rush'))
        entries.append((date_str, 'Eid al-Fitr Day 1', 0, 0, 'Muslim shops closed, others open'))
        entries.append((offset(date_str, +1), 'Eid al-Fitr Day 2', 0, 0, 'Celebrations continue'))
        entries.append((offset(date_str, +2), 'Eid al-Fitr Day 3', 0, 0, 'Recovery'))


# EID AL-ADHA 
# Qurbani festival — heavy meat shopping pre-days.

for date, name in sorted(lk_official.items()):
    if 'Eid al-Adha' in name:
        date_str = date.strftime('%Y-%m-%d')

        entries.append((offset(date_str, -4), 'Pre Eid Adha Shopping', +1, 0, 'Shopping starts'))
        entries.append((offset(date_str, -3), 'Pre Eid Adha Shopping', +1, 0, 'Meat shopping accelerates'))
        entries.append((offset(date_str, -2), 'Pre Eid Adha Peak', +1, 0, 'Heavy meat shopping for Qurbani'))
        entries.append((offset(date_str, -1), 'Pre Eid Adha Peak', +1, 0, 'PEAK: Last minute purchases'))
        entries.append((date_str, 'Eid al-Adha Day 1', +1, 0, 'Qurbani meat distribution — elevated sales'))
        entries.append((offset(date_str, +1), 'Eid al-Adha Day 2', 0, 0, 'Celebrations'))
        entries.append((offset(date_str, +2), 'Eid al-Adha Day 3', 0, 0, 'Recovery'))


#  DEEPAVALI 
# Milk, rice, flour, beverages spike day before

for date, name in sorted(lk_official.items()):
    if 'Deepavali' in name:
        date_str = date.strftime('%Y-%m-%d')

        entries.append((offset(date_str, -1), 'Pre Deepavali Shopping', +1, 0,
                        'Milk/rice/flour/beverages spike'))
        entries.append((date_str, 'Deepavali Festival Day', 0, 0,
                        'Tamil shops closed, others open'))


#  CHRISTMAS 


for date, name in sorted(lk_official.items()):
    if 'Christmas' in name:
        date_str = date.strftime('%Y-%m-%d')

        entries.append((offset(date_str, -3), 'Pre Christmas Shopping', +1, 0, 'Shopping spike'))
        entries.append((offset(date_str, -2), 'Pre Christmas Shopping', +1, 0, 'Shopping peak'))
        entries.append((offset(date_str, -1), 'Pre Christmas Peak', +1, 0, 'Last shopping day'))
        entries.append((date_str, 'Christmas Day', -1, 0, 'Shops CLOSED'))


#  GOOD FRIDAY 


for date, name in sorted(lk_official.items()):
    if 'Good Friday' in name:
        date_str = date.strftime('%Y-%m-%d')
        entries.append((date_str, 'Good Friday', 0, 0,
                        'Minor reduction — small Christian population'))


# MAY DAY


for date, name in sorted(lk_official.items()):
    if "Workers" in name or "May Day" in name:
        date_str = date.strftime('%Y-%m-%d')
        entries.append((date_str, "International Workers' Day", 0, 0,
                        'Government holiday — retail largely open'))


# MAHA SIVARATHRI


for date, name in sorted(lk_official.items()):
    if 'Sivarathri' in name or 'Shivaratri' in name:
        date_str = date.strftime('%Y-%m-%d')
        entries.append((date_str, 'Maha Sivarathri Day', 0, 0,
                        'Hindu observance — minor impact on general retail'))


# PROPHET'S BIRTHDAY


for date, name in sorted(lk_official.items()):
    if "Prophet" in name:
        date_str = date.strftime('%Y-%m-%d')
        entries.append((date_str, "Prophet's Birthday", 0, 0,
                        "Muslim observance — minor impact on general retail"))


#  BUILD AND SAVE

df = pd.DataFrame(entries, columns=[
    'date', 'holiday_name', 'general_impact', 'meat_restriction', 'notes'
])

df['date'] = pd.to_datetime(df['date'])
df = df.drop_duplicates(subset=['date', 'holiday_name']).sort_values('date').reset_index(drop=True)
df['general_impact'] = df['general_impact'].astype('int8')
df['meat_restriction'] = df['meat_restriction'].astype('int8')

df.to_csv(OUTPUT_FILE, index=False)

print(f"Holiday calendar saved: {OUTPUT_FILE}")
print(f"Total entries: {len(df)}")
print(f"Spike days (+1): {len(df[df['general_impact']==1])}")
print(f"Closure days (-1): {len(df[df['general_impact']==-1])}")
print(f"Neutral days (0): {len(df[df['general_impact']==0])}")
print(f"Meat restriction days: {len(df[df['meat_restriction']==1])}")