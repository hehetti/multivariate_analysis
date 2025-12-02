import pandas as pd
import numpy as np
import lightgbm as lgb
from lightgbm import early_stopping, log_evaluation
from sklearn.preprocessing import StandardScaler
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# -------------------------------------------------
# 1. 데이터 구성
# -------------------------------------------------
file_path = "../../data/final_heatmap_lag_without_leakage.csv"
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

y = df.iloc[:, 0].values
X = df.iloc[:, 2:].values

y_2021 = df_test.iloc[:, 0].values
X_2021 = df_test.iloc[:, 2:].values

test_size = len(df) // 4
X_train, X_test = X[test_size:], X[:test_size]
y_train, y_test = y[test_size:], y[:test_size]

scaler2 = StandardScaler()
scaler2.fit(X_train)

X_train = scaler2.transform(X_train)
X_test = scaler2.transform(X_test)

X_2021_scaled = scaler2.transform(X_2021)

train_data = lgb.Dataset(X_train, label=y_train)
valid_data = lgb.Dataset(X_test, label=y_test)

# -------------------------------------------------
# 2. 파라미터 -> optuna에서 추출한 최적 파라미터로
# -------------------------------------------------
params_l2 = {'learning_rate': 0.04209891275728966, 'num_leaves': 59, 'feature_fraction': 0.7300948366339031, 'bagging_fraction': 0.9811027608098394, 'bagging_freq': 5, 'lambda_l1': 1.0833737287677067e-07, 'lambda_l2': 2.658097575079241e-07, 'min_child_samples': 17}

params_tweedie = {'tweedie_variance_power': 1.216613798570502, 'learning_rate': 0.02166082006547862, 'num_leaves': 96, 'max_depth': 14, 'min_data_in_leaf': 36, 'feature_fraction': 0.6635243528116432, 'bagging_fraction': 0.9821246548236764, 'bagging_freq': 1, 'lambda_l1': 1.9052833646205864e-08, 'lambda_l2': 1.7159611853713213e-07}

# -------------------------------------------------
# 3. 학습 (callback 기반 early stopping)
# -------------------------------------------------
model = lgb.train(
    params_l2,
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
# 4. 2021년 데이터 예측
# -------------------------------------------------
y_pred = model.predict(X_2021_scaled)
print("Sample predictions:", y_pred[:5])

mae = mean_absolute_error(y_2021, y_pred)
mse = mean_squared_error(y_2021, y_pred); rmse = np.sqrt(mse)
y_mean = np.mean(y_2021); mape_approx = (mae / y_mean) * 100 # 퍼센트 오차(MAPE) 근사치 계산 (평균 대비 오차율)
r2 = r2_score(y_2021, y_pred) # R2 Score (설명력)

print(f"MAE: {mae}")
print(f"RMSE: {rmse}")
print(f"MAPE(평균 대비 오차율): {mape_approx:.2f}%")
print(f"R2 Score: {r2:.4f}")

# -------------------------------------------------
# 5. 시각화 - 앞부분 100개만
# -------------------------------------------------
plt.figure(figsize=(15, 6))
plt.plot(y_2021[:150], label='Actual', color='blue', alpha=0.7)
plt.plot(y_pred[:150], label='Prediction', color='red', linestyle='--', alpha=0.7)
plt.title(f'Actual vs Prediction (MAE: {mae:.2f}, RMSE: {rmse:.2f})')
plt.legend()
plt.show()

# -------------------------------------------------
# 6. 모델을 로컬에 저장
# -------------------------------------------------
base_path = "../../model/"
model_path = base_path + "lgb_model_l2_optuna.pkl"
scaler_path = base_path + "lgb_scaler_l2_optuna.pkl"
joblib.dump(model, model_path)
joblib.dump(scaler2, scaler_path)
print("모델과 스케일러가 저장됐습니다.")