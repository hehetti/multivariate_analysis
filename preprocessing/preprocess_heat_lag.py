import pandas as pd
import numpy as np
from pytimekr import pytimekr

# 지연변수 및 차분변수 추가
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
        lag1_series = df[col].shift(1)
        new_cols[f'{col}_lag1'] = lag1_series

        for lag in lags:
            if lag==1: continue
            lag_series = df[col].shift(lag)                      # 이전 값
            new_cols[f'{col}_lag{lag}'] = lag_series             # lag 컬럼
            new_cols[f'{col}_diff{lag}'] = lag1_series - lag_series # 한시간 전 - lag시간 전의 차분값; Y(t-1)-Y(t-lag)

            print(f'{col}의 {lag}번째 처리 중')

        # 2-2) 이동평균 컬럼; ma = moving average
        for w in windows:
            new_cols[f'{col}_ma{w}' ] = lag1_series.rolling(window=w, min_periods=w).mean()
            print(f'{col}의 {w} 처리 중')

    # 3) 새 컬럼들만 모은 DataFrame 만들고, 원본 df와 한 번에 결합
    new_df = pd.DataFrame(new_cols, index=df.index)
    df = pd.concat([df, new_df], axis=1)

    # 4) fragment 해소용 copy (선택이지만 경고 줄이기에 좋음)
    df = df.copy()

    return df

def main():
    # 1. 원본 데이터 로드
    input_path = "../data/final_dataset_complete.csv"
    output_path = "../data/final_heatmap_lag_without_leakage.csv" # 최종 저장 파일명

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