import pandas as pd
import numpy as np
import lightgbm as lgb
from lightgbm import early_stopping, log_evaluation
from sklearn.preprocessing import StandardScaler
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# -------------------------------------------------
# 1. CSV 읽기 + 최근 30000개만 사용
# -------------------------------------------------
file_path = "../data/final_heatmap_lag_without_leakage.csv"
df_raw = pd.read_csv(file_path)

# 실험 데이터셋(2021)과 분리
df_raw['datetime'] = pd.to_datetime(df_raw['datetime'])
print(f"전체 로드된 데이터셋: {len(df_raw)}")
print(f"날짜 범위: {df_raw['datetime'].min()} ~ {df_raw['datetime'].max()}")

df_test = df_raw[df_raw['datetime'] >= '2021-01-01'].copy().reset_index(drop=True)
df = df_raw[df_raw['datetime']<'2021-01-01'].copy().reset_index(drop=True)

# 첫 168행 제거
df = df.iloc[168:].reset_index(drop=True)
print(f"전체 train&eval 데이터 개수: {len(df)}")
print(f"날짜 범위: {df['datetime'].min()} ~ {df['datetime'].max()}")
print(f"test 데이터셋 개수: {len(df_test)}")
print(f"날짜 범위: {df_test['datetime'].min()} ~ {df_test['datetime'].max()}")

# -------------------------------------------------
# 2. 첫 번째 열 = y, 나머지 = X
# -------------------------------------------------
y = df.iloc[:, 0].values
X = df.iloc[:, 2:].values

y_2021 = df_test.iloc[:, 0].values
X_2021 = df_test.iloc[:, 2:].values

# -------------------------------------------------
# 3. Train/Test Split (앞쪽 1/4 test)
# -------------------------------------------------
test_size = len(df) // 4
X_train, X_test = X[test_size:], X[:test_size]
y_train, y_test = y[test_size:], y[:test_size]

scaler2 = StandardScaler()
scaler2.fit(X_train)

# train과 test 모두 transform
X_train = scaler2.transform(X_train)
X_test = scaler2.transform(X_test)

X_2021_scaled = scaler2.transform(X_2021)

# -------------------------------------------------
# 4. Dataset 생성
# -------------------------------------------------
train_data = lgb.Dataset(X_train, label=y_train)
valid_data = lgb.Dataset(X_test, label=y_test)

# -------------------------------------------------
# 5. 파라미터
# -------------------------------------------------
params = {
    "objective": "regression_l2",   # ← MSE 기반 학습
    "metric": "l1",
    "learning_rate": 0.01,
    "num_leaves": 100,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 0.5,
    "min_child_samples": 10,
    "random_state": 42
}

params_tweedie = {
    # 1. 핵심 목표 (Objective)
    "objective": "tweedie",
    "tweedie_variance_power": 1.525,  # 표의 값: 1.525 (1~2 사이: Poisson과 Gamma의 복합)
    "metric": "rmse",                 # 평가는 RMSE로

    # 2. 부스팅 및 트리 구조 (Boosting & Tree)
    "boosting_type": "gbdt",
    "extra_trees": True,              # 표의 값: True (일반 GBDT보다 과적합 방지에 강함)
    "num_leaves": 81,                 # 표의 값: 81
    "max_depth": -1,                  # 표의 값: -1 (깊이 제한 없음)
    "max_bin": 255,                   # 표의 값: 255
    "min_data_in_leaf": 1155,         # [주의] 표의 값: 1155 (데이터가 적으면 이 값을 줄여야 함!)
    "min_sum_hessian_in_leaf": 0.1908, # 표의 값: 1.908 * 10^-1

    # 3. 학습률 및 반복 (Learning)
    "learning_rate": 0.01171,         # 표의 값: 1.171 * 10^-2
    "n_estimators": 6144,             # 표의 값: 6144 (매우 많음, early_stopping 필수)

    # 4. 샘플링 (Sampling)
    "feature_fraction": 0.9543,       # 표의 값: 9.543 * 10^-1
    "bagging_fraction": 0.8671,       # 표의 값: 8.671 * 10^-1
    "bagging_freq": 1,

    # 5. 규제 (Regularization)
    "lambda_l1": 0.6595,              # 표의 값: 6.595 * 10^-1
    "lambda_l2": 1.410,               # 표의 값: 1.410

    # 6. 시스템 설정
    "random_state": 42,
    "n_jobs": -1,
    # "device_type": "gpu"            # GPU 사용 환경이라면 주석 해제
}

# -------------------------------------------------
# 6. Train (callback 기반 early stopping)
# -------------------------------------------------
model = lgb.train(
    params_tweedie,
    train_data,
    num_boost_round=5000,
    valid_sets=[valid_data],

    # ← early stopping 처리
    callbacks=[
        early_stopping(stopping_rounds=200),  # 200회 개선 없으면 stop
        log_evaluation(200)                   # 200 iteration마다 로그 출력
    ]
)

# -------------------------------------------------
# 7. Predict
# -------------------------------------------------
y_pred = model.predict(X_2021_scaled)
print("Sample predictions:", y_pred[:5])
mae = mean_absolute_error(y_2021, y_pred)
print("MAE:", mae)

mse = mean_squared_error(y_2021, y_pred)
rmse = np.sqrt(mse)
print(f"RMSE: {rmse}")

print("실 데이터 통계치 확인 === ")
y_mean = np.mean(y_2021)
# print(f"실제 값 평균(Mean): {y_mean:.4f}")
# print(f"실제 값 최소~최대: {np.min(y_2021):.4f} ~ {np.max(y_2021):.4f}")

# 퍼센트 오차(MAPE) 대략 계산 (평균 대비 오차율)
mape_approx = (mae / y_mean) * 100
print(f"평균 대비 오차율(Approx MAPE): {mape_approx:.2f}%")

# R2 Score (설명력)
r2 = r2_score(y_2021, y_pred)
print(f"R2 Score: {r2:.4f}")

# -------------------------------------------------
# 시각화 - 앞부분 100개만
# -------------------------------------------------
plt.figure(figsize=(15, 6))
plt.plot(y_2021[:150], label='Actual', color='blue', alpha=0.7)
plt.plot(y_pred[:150], label='Prediction', color='red', linestyle='--', alpha=0.7)
plt.title(f'Actual vs Prediction (MAE: {mae:.2f}, RMSE: {rmse:.2f})')
plt.legend()
plt.show()

# 7. 저장
base_path = "../model/"
model_path = base_path + "lgb_model_tweedie.pkl"
scaler_path = base_path + "lgb_scaler_tweedie.pkl"
joblib.dump(model, model_path)
joblib.dump(scaler2, scaler_path)
print("모델과 스케일러가 저장됐습니다.")
