import pandas as pd
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import numpy as np

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


final_params = {'tweedie_variance_power': 1.216613798570502, 'learning_rate': 0.02166082006547862, 'num_leaves': 96, 'max_depth': 14, 'min_data_in_leaf': 36, 'feature_fraction': 0.6635243528116432, 'bagging_fraction': 0.9821246548236764, 'bagging_freq': 1, 'lambda_l1': 1.9052833646205864e-08, 'lambda_l2': 1.7159611853713213e-07}

final_params.update({
    "objective": "tweedie",
    "metric": "l1",
    # "boosting_type": "gdbt",
    "n_jobs": -1,
    "verbosity": -1
})

seeds = [42, 2023, 0, 777, 999]
models = []

for seed in seeds:
    final_params["random_state"] = seed  # 시드만 변경

    print(f"Training with seed {seed}...")
    model = lgb.train(
        final_params,
        train_data,
        num_boost_round=10000,
        valid_sets=[valid_data],
        callbacks=[lgb.early_stopping(stopping_rounds=100)]
    )
    models.append(model)

# 최종 예측: 5개 모델의 예측값 평균
preds_list = [model.predict(X_test) for model in models]
final_pred = np.mean(preds_list, axis=0) # 평균값 사용

print(f"최종 예측값: {final_pred}")