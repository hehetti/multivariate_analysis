import pandas as pd
from pytimekr import pytimekr

df = pd.read_csv("../data/청주_열공급량_기상변수_lag추가.csv", encoding='utf-8-sig')
df['datetime'] = pd.to_datetime(df['datetime'])

# 3-1. 동절기 여부
df['is_cold'] = (df['month']<5) | (df['month']>9)
df['is_cold'] = df['is_cold'].astype(int)

# 3-2. 주말 여부
df['is_weekend'] = (df['datetime'].dt.dayofweek >= 5).astype(int)

# 3-3. 공휴일 여부
unique_years = df['datetime'].dt.year.unique()
kr_holidays = []
for year in unique_years:
    kr_holidays.extend(pytimekr.holidays(year))
df['is_holiday'] = df['datetime'].dt.date.isin(kr_holidays).astype(int) # 연,월,일만 추출, 포함되는지 확인

# 3-4. 코로나 기간 여부
mask_corona = (df['datetime'] >= '2020-02-29') & (df['datetime'] <= '2021-10-31')
df['is_corona'] = mask_corona.astype(int)

# 3-5. 일일 누적 강수량
df['rn_day'] = df.groupby(df['datetime'].dt.date)['강수량(mm)'].cumsum().round(2)