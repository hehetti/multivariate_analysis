import pandas as pd
import numpy as np
from pytimekr import pytimekr

# 1. 기상 파생변수 추가
def get_wet_bulb_temp(ta, hm):
    """Stull의 습구온도 추정식"""
    term1 = ta * np.arctan(0.151977 * np.sqrt(hm + 8.313659))
    term2 = np.arctan(ta + hm)
    term3 = np.arctan(hm - 1.67633)
    term4 = 0.00391838 * np.power(hm, 1.5) * np.arctan(0.023101 * hm)
    return term1 + term2 - term3 + term4 - 4.686035

def add_weather_features(df):
    print("Step 1: 기상 파생변수 생성 중...")

    # 컬럼명 매핑 (본인 csv에 맞게 수정 필요)
    col_temp = '기온(°C)'
    col_humid = '상대습도(%)'
    col_ws = '풍속(m/s)'
    col_wd = '풍향(16방위)'

    # 1-1. 습구온도 계산
    tw = get_wet_bulb_temp(df[col_temp], df[col_humid])

    # 1-2. Heat Stress (여름)
    # 공식: 체감 - 실제 (음수는 0처리)
    feel_summer = -0.2442 + (0.55399 * tw) + (0.45535 * df[col_temp]) \
                  - (0.0022 * tw**2) + (0.00278 * tw * df[col_temp]) + 3.0
    stress_summer = feel_summer - df[col_temp]
    df['heat_stress'] = stress_summer.clip(lower=0)

    # 1-3. Cold Stress (겨울)
    # 공식: 실제 - 체감 (양수화, 조건 미달 0처리)
    v_kmh = df[col_ws] * 3.6
    v_pow = np.power(v_kmh, 0.16)
    feel_winter = 13.12 + (0.6215 * df[col_temp]) - (11.37 * v_pow) + (0.3965 * df[col_temp] * v_pow)

    stress_winter = df[col_temp] - feel_winter

    # 조건: 기온 < 10, 풍속 >= 1.3m/s (조건 안맞으면 0)
    cold_mask = (df[col_temp] < 10) & (df[col_ws] >= 1.3)
    df['cold_stress'] = np.where(cold_mask, stress_winter, 0)
    df['cold_stress'] = df['cold_stress'].clip(lower=0) # 혹시 모를 음수 처리

    # 1-4. 풍속 벡터 변환 (Wind Vector)
    theta = np.deg2rad(df[col_wd])
    df['wind_x'] = df[col_ws] * np.sin(theta)
    df['wind_y'] = df[col_ws] * np.cos(theta)
    return df

# 2. 연중일월 시간 주기 계산
def add_cyclic_time_features(df, datetime_col='datetime'):
    """
    df: pandas DataFrame
    datetime_col: 날짜시간 컬럼 이름 (예: 'datetime', '일시' 등)
    """
    print("Step 2: 시간 주기성 인코딩 중...")
    # 1) datetime 형식으로 변환
    df[datetime_col] = pd.to_datetime(df[datetime_col])
    dt = df[datetime_col]

    # 2) 월/일/시간, 연 기준 일자(day of year)
    df['month']     = dt.dt.month        # 1 ~ 12
    df['day']       = dt.dt.day          # 1 ~ 28~31
    df['hour']      = dt.dt.hour         # 0 ~ 23
    df['dayofyear'] = dt.dt.dayofyear    # 1 ~ 365 또는 366
    df['weekday'] = dt.dt.weekday
    wd_col='풍향(16방위)'
    ws_col='풍속(m/s)'

    # 해당 연도가 윤년인지 여부 (True/False)
    is_leap = dt.dt.is_leap_year
    # 연도의 총 일수: 윤년이면 366, 아니면 365
    days_in_year = np.where(is_leap, 366, 365)
    # 해당 월의 총 일수 (28, 29, 30, 31 중 하나)
    days_in_month = dt.dt.days_in_month

    # 3) 월(month) 주기 인코딩 (12개월 주기)
    theta_month = 2 * np.pi * (df['month'] - 1) / 12.0
    df['month_sin'] = np.sin(theta_month)
    df['month_cos'] = np.cos(theta_month)

    # 4) 시간(hour) 주기 인코딩 (24시간 주기)
    theta_hour = 2 * np.pi * df['hour'] / 24.0
    df['hour_sin'] = np.sin(theta_hour)
    df['hour_cos'] = np.cos(theta_hour)

    # 5) 월 안의 날짜(day in month) 주기 인코딩
    theta_day = 2 * np.pi * (df['day'] - 1) / days_in_month
    df['day_sin'] = np.sin(theta_day)
    df['day_cos'] = np.cos(theta_day)

    # 6) 연 기준 일자(day of year) 주기 인코딩
    #    평년이면 365일, 윤년이면 366일을 주기로 사용
    theta_doy = 2 * np.pi * (df['dayofyear'] - 1) / days_in_year
    df['doy_sin'] = np.sin(theta_doy)
    df['doy_cos'] = np.cos(theta_doy)

    #    월=0, ..., 일=6 이라서 그대로 0~6 사용
    theta_wd = 2 * np.pi * df['weekday'] / 7.0
    df['weekday_sin'] = np.sin(theta_wd)
    df['weekday_cos'] = np.cos(theta_wd)

    # 불필요한 임시 컬럼 삭제
    df.drop(columns=['month', 'day', 'hour', 'dayofyear', 'weekday'], inplace=True)

    return df

# 3. 지연변수 및 차분변수 추가
def add_lag_and_diff_features(df, cols):
    """
    df   : 시간 순서대로 정렬된 DataFrame
    cols : lag / 차이 / 이동평균을 만들 열 이름 리스트

    생성되는 것:
      - {col}_lag{lag}  : lag 값
      - {col}_diff{lag} : 현재 - lag 값
      - {col}_ma{w}     : w개 window 이동평균
    """
    print("Step 4: Lag/Diff/MA 피처 생성 중 (시간이 좀 걸릴 수 있음)...")
    # 1) 사용할 lag, window 정의
    lags = list(range(1, 25)) + [48, 72, 168]               # 1~24, 48, 72, 168
    windows = list(range(3, 25, 3)) + list(range(48, 169, 24))  # 3,6,...,24, 48,72,...,168

    # 2) 새로 만들 모든 컬럼을 여기다가 모은 뒤, 마지막에 한 번에 concat
    new_cols = {}

    # 2-1) lag & diff 컬럼
    for col in cols:
        for lag in lags:
            lag_series = df[col].shift(lag)                      # 이전 값
            new_cols[f'{col}_lag{lag}'] = lag_series             # lag 컬럼
            new_cols[f'{col}_diff{lag}'] = df[col] - lag_series  # 현재 - 이전
            print(f'{lag}')

        # 2-2) 이동평균 컬럼; ma = moving average
        for w in windows:
            new_cols[f'{col}_ma{w}' ] = df[col].rolling(window=w, min_periods=w).mean()
            print(w)

    # 3) 새 컬럼들만 모은 DataFrame 만들고, 원본 df와 한 번에 결합
    new_df = pd.DataFrame(new_cols, index=df.index)
    df = pd.concat([df, new_df], axis=1)

    # 4) fragment 해소용 copy (선택이지만 경고 줄이기에 좋음)
    df = df.copy()

    return df

# 4. 달력 및 이벤트 정보
def add_calendar_features(df):
    print("Step 3: 달력/이벤트 정보 추가 중...")
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['month'] = df['datetime'].dt.month

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
    df.drop(columns='month', inplace=True)
    return df

def main():
    # 1. 원본 데이터 로드
    input_path = "C:\\GitHub\\multivariate_analysis\\data\\final_dataset_complete\\final_dataset_complete.csv"
    output_path = "C:\\GitHub\\multivariate_analysis\\data\\final_heatmap_lag.csv" # 최종 저장 파일명

    print(f"파일 로드 중: {input_path}")
    df = pd.read_csv(input_path, encoding='utf-8-sig')
    df['datetime'] = pd.to_datetime(df['datetime'])
    print("정상작동")
    # 3. Lag 생성 (대상 컬럼 지정)
    target_cols_for_lag = [
        '청주 지역공급량(Gcal)'
    ]
    df = add_lag_and_diff_features(df, cols=target_cols_for_lag)

    # 4. 최종 저장
    print(f"최종 파일 저장 중: {output_path}")
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print("저장 완료!")

if __name__ == "__main__":
    main()