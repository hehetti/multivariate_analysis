import numpy as np
import pandas as pd
from scipy import stats

# ===== 설정 =====
input_path = "/home/jwlee/eunhwa/catboost/final_heatmap_lag.csv"
output_path = "/home/jwlee/eunhwa/catboost/granger_lags.csv"
max_lag = 168
alpha = 0.05

# ---------- OLS + F-test로 Granger p-value 계산 ----------
def granger_pvalue_single_lag(y, x, lag):
    """
    y, x: 1D numpy array
    lag: int (1 이상)
    return: p-value (float)
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    T = len(y)
    if len(x) != T:
        raise ValueError("y and x must have same length")

    # 가장 큰 lag를 기준으로 윈도우 맞추기
    maxlag = lag
    T_eff = T - maxlag
    if T_eff <= 2 * lag + 1:  # 자유도 부족
        return 1.0

    # 종속변수 (maxlag 시점 이후)
    Y = y[maxlag:]

    # y의 lag들 (열: y_{t-1},...,y_{t-lag})
    y_lags = np.column_stack([
        y[maxlag - k : T - k] for k in range(1, lag + 1)
    ])

    # x의 lag들
    x_lags = np.column_stack([
        x[maxlag - k : T - k] for k in range(1, lag + 1)
    ])

    # 제한모형: Y ~ const + y_lags
    X_r = np.column_stack([np.ones(T_eff), y_lags])
    # 전체모형: Y ~ const + y_lags + x_lags
    X_f = np.column_stack([np.ones(T_eff), y_lags, x_lags])

    # OLS (정규방정식 사용: (X'X)^(-1) X'Y)
    # 제한모형
    XtX_r = X_r.T @ X_r
    XtY_r = X_r.T @ Y
    beta_r = np.linalg.solve(XtX_r, XtY_r)
    resid_r = Y - X_r @ beta_r
    SSR_r = float(resid_r.T @ resid_r)

    # 전체모형
    XtX_f = X_f.T @ X_f
    XtY_f = X_f.T @ Y
    beta_f = np.linalg.solve(XtX_f, XtY_f)
    resid_f = Y - X_f @ beta_f
    SSR_f = float(resid_f.T @ resid_f)

    # F-test
    df1 = lag                         # 추가된 x lag 개수
    df2 = T_eff - (2 * lag + 1)       # 전체모형 자유도: T_eff - (p_y + p_x + const)
    if df2 <= 0:
        return 1.0

    num = (SSR_r - SSR_f) / df1
    den = SSR_f / df2
    if den <= 0:
        return 1.0

    F = num / den
    if F < 0:
        return 1.0

    p_value = 1.0 - stats.f.cdf(F, df1, df2)
    return float(p_value)


def find_first_sig_lag(y, x, max_lag=168, alpha=0.05):
    """
    lag=1부터 max_lag까지 순서대로 검사하면서
    처음으로 Granger 유의한 lag를 반환.
    없으면 0 반환.
    """
    for lag in range(1, max_lag + 1):
        p = granger_pvalue_single_lag(y, x, lag)
        if p < alpha:
            return lag
    return 0        # 유의수준 (Granger 검정 기준)

# ===== 데이터 불러오기 =====
df = pd.read_csv(input_path)

y = df.iloc[:, 0].values
n_cols = df.shape[1]

lags_result = np.zeros(n_cols, dtype=int)

# (예시: 0번이 y, 1번은 id/날짜라서 2번부터 Granger 검사)
for j in range(2, n_cols):
    x = df.iloc[:, j].values

    try:
        lag = find_first_sig_lag(y, x, max_lag=max_lag, alpha=alpha)
        lags_result[j] = lag   # 없으면 0, 있으면 최초 유의 lag
    except Exception as e:
        print(f"컬럼 {j} ({df.columns[j]})에서 오류 발생: {e}")
        lags_result[j] = 0
        continue

# 결과 저장
result_df = pd.DataFrame([lags_result], columns=df.columns)
result_df.to_csv(output_path, index=False)

print("완료! 결과가 저장된 파일:", output_path)
