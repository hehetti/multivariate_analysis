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
    return term1 + term2 - term3 + term4 - 4.686035

def calc_heat_stress(ta, hm):
    """
    여름철 체감온도 (기상청 공식)
    """
    tw = get_wet_bulb_temp(ta, hm) # 습구온도 먼저 계산

    # 공식 적용
    feel = -0.2442 + (0.55399 * tw) + (0.45535 * ta) \
           - (0.0022 * tw**2) + (0.00278 * tw * ta) + 3.0
    stress = feel-ta

    return stress.clip(lower=0)

def calc_cold_stress(ta, ws):
    """
    겨울철 풍냉 체감온도 (북미/기상청 통합 공식)
    """
    v_kmh = ws * 3.6

    # 지수 연산(**0.16)
    v_pow = np.power(v_kmh, 0.16)

    feel = 13.12 + (0.6215 * ta) - (11.37 * v_pow) + (0.3965 * ta * v_pow)
    stress = ta-feel

    return stress.clip(lower=0)

df = pd.read_csv("../data/청주_열공급량_기상변수_lag추가.csv", encoding='utf-8-sig')
df['datetime'] = pd.to_datetime(df['datetime'])
df['month'] = df['datetime'].dt.month

is_summer = (df['month'] >= 5) & (df['month'] <= 9)
is_winter = ~is_summer
df['heat_stress'] = 0.0
df['cold_stress'] = 0.0

if is_summer.any():
    df.loc[is_summer, 'heat_stress'] = calc_heat_stress(
        df.loc[is_summer, '기온(°C)'],
        df.loc[is_summer, '상대습도(%)']
    )
if is_winter.any():
    winter_stress = calc_cold_stress(
        df.loc[is_winter, '기온(°C)'],
        df.loc[is_winter, '풍속(m/s)']
    )
    df.loc[is_winter, 'cold_stress'] = winter_stress
    invalid_condition = (df['기온(°C)'] >= 10) | (df['풍속(m/s)'] < 1.3)
    df.loc[is_winter & invalid_condition, 'cold_stress'] = 0

print(df[['datetime', '기온(°C)', '상대습도(%)', '풍속(m/s)', 'heat_stress', 'cold_stress']].head())
df.to_csv('../data/weather_data_with_feel.csv', index=False, encoding='utf-8-sig')