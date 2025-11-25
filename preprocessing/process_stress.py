import pandas as pd
import numpy as np

def get_wet_bulb_temp(ta, hm):
    """
    Stull의 습구온도 추정식
    """
    term1 = ta * np.arctan(0.151977 * np.sqrt(hm + 8.313659))
    term2 = np.arctan(ta + hm)
    term3 = np.arctan(hm - 1.67633)
    term4 = 0.00391838 * np.power(hm, 1.5) * np.arctan(0.023101 * hm)

    Tw = term1 + term2 - term3 + term4 - 4.686035
    return Tw

def calc_summer_feel(ta, hm):
    """
    여름철 체감온도 (기상청 공식)
    """
    tw = get_wet_bulb_temp(ta, hm) # 습구온도 먼저 계산

    # 공식 적용
    feel = -0.2442 + (0.55399 * tw) + (0.45535 * ta) \
           - (0.0022 * tw**2) + (0.00278 * tw * ta) + 3.0
    return feel

def calc_winter_wind_chill(ta, ws):
    """
    겨울철 풍냉 체감온도 (북미/기상청 통합 공식)
    """
    # 풍속이 1.3m/s 미만일 때의 예외처리는 일단 제외하고 공식 그대로 적용
    # 필요하다면 np.where로 풍속 조건을 추가할 수 있음

    # 지수 연산(**0.16)
    v_pow = np.power(ws, 0.16)

    wc = 13.12 + (0.6215 * ta) - (11.37 * v_pow) + (0.3965 * ta * v_pow)


    return wc

df = pd.read_csv("data/청주_열공급량_기상변수_corona_추가.csv")
df['datetime'] = pd.to_datetime(df['datetime'])
df['month'] = df['datetime'].dt.month

is_summer = (df['month'] >= 5) & (df['month'] <= 9)
is_winter = ~is_summer
df['heat_stress'] = np.nan
df['cold_stress'] = np.nan

if is_summer.any():
    df.loc[is_summer, 'heat_stress'] = calc_summer_feel(
        df.loc[is_summer, '기온(°C)'],
        df.loc[is_summer, '풍속(m/s)']
    )
if is_winter.any():
    df.loc[is_winter, 'cold_stress'] = calc_winter_wind_chill(
        df.loc[is_winter, '기온(°C)'],
        df.loc[is_winter, '풍속(m/s)']
    )
print(df.head())
df.to_csv('weather_data_with_feel.csv', index=False, encoding='utf-8-sig')