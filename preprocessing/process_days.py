import pandas as pd
from pytimekr import pytimekr

df = pd.read_csv("../data/weather_data_with_feel.csv", encoding='utf-8-sig')
df['datetime'] = pd.to_datetime(df['datetime'])

df['is_cold'] = (df['month']<5) | (df['month']>9)
df['is_weekend'] = df['datetime'].dt.dayofweek >= 5
unique_years = df['datetime'].dt.year.unique()

kr_holidays = []
for year in unique_years:
    kr_holidays.extend(pytimekr.holidays(year))
temp_dates = df['datetime'].dt.date # 연,월,일만 추출
df['is_holiday'] = temp_dates.isin(kr_holidays)

print(df[['datetime', 'is_cold', 'is_weekend', 'is_holiday']].head(10))
df.to_csv("../data/weather_data_with_holiday.csv", index=False, encoding='utf-8-sig')