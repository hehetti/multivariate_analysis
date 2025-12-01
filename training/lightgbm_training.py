import pandas as pd
import numpy as np
import lightgbm as lgb
from lightgbm import early_stopping, log_evaluation
from sklearn.preprocessing import StandardScaler
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error

# -------------------------------------------------
# 1. CSV 읽기 + 최근 30000개만 사용
# -------------------------------------------------
file_path = "../data/final_heatmap_lag/final_heatmap_lag.csv"
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
    "learning_rate": 0.03,
    "num_leaves": 64,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 2.0,
    "random_state": 42
}

# -------------------------------------------------
# 6. Train (callback 기반 early stopping)
# -------------------------------------------------
model = lgb.train(
    params,
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

# 7. 저장
base_path = "../model/"
model_path = base_path + "lgb_model.pkl"
scaler_path = base_path + "lgb_scaler.pkl"
joblib.dump(model, model_path)
joblib.dump(scaler2, scaler_path)
print("모델과 스케일러가 저장됐습니다.")
