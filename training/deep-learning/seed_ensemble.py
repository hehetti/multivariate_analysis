# import joblib
# import pandas as pd
# from sklearn.preprocessing import StandardScaler
# import lightgbm as lgb
# import numpy as np
#
# file_path = "../../data/final_heatmap_lag_without_leakage.csv"
# df_raw = pd.read_csv(file_path)
#
# # 실험 데이터셋(2021)과 분리
# df_raw['datetime'] = pd.to_datetime(df_raw['datetime'])
# print(f"전체 로드된 데이터셋: {len(df_raw)}")
# print(f"날짜 범위: {df_raw['datetime'].min()} ~ {df_raw['datetime'].max()}")
#
# df_test = df_raw[df_raw['datetime'] >= '2021-01-01'].copy().reset_index(drop=True)
# df = df_raw[df_raw['datetime']<'2021-01-01'].copy().reset_index(drop=True)
#
# # 첫 168행 제거
# df = df.iloc[168:].reset_index(drop=True)
# print(f"전체 train&eval 데이터 개수: {len(df)}")
# print(f"날짜 범위: {df['datetime'].min()} ~ {df['datetime'].max()}")
# print(f"test 데이터셋 개수: {len(df_test)}")
# print(f"날짜 범위: {df_test['datetime'].min()} ~ {df_test['datetime'].max()}")
#
# # -------------------------------------------------
# # 2. 첫 번째 열 = y, 나머지 = X
# # -------------------------------------------------
# y = df.iloc[:, 0].values
# X = df.iloc[:, 2:].values
#
# y_2021 = df_test.iloc[:, 0].values
# X_2021 = df_test.iloc[:, 2:].values
#
# # -------------------------------------------------
# # 3. Train/Test Split (앞쪽 1/4 test)
# # -------------------------------------------------
# test_size = len(df) // 4
# X_train, X_test = X[test_size:], X[:test_size]
# y_train, y_test = y[test_size:], y[:test_size]
#
# scaler = StandardScaler()
# scaler.fit(X_train)
#
# # train과 test 모두 transform
# X_train = scaler.transform(X_train)
# X_test = scaler.transform(X_test)
#
# X_2021_scaled = scaler.transform(X_2021)
#
# # -------------------------------------------------
# # 4. Dataset 생성
# # -------------------------------------------------
# train_data = lgb.Dataset(X_train, label=y_train)
# valid_data = lgb.Dataset(X_test, label=y_test)
#
#
# final_params = {'tweedie_variance_power': 1.216613798570502, 'learning_rate': 0.02166082006547862, 'num_leaves': 96, 'max_depth': 14, 'min_data_in_leaf': 36, 'feature_fraction': 0.6635243528116432, 'bagging_fraction': 0.9821246548236764, 'bagging_freq': 1, 'lambda_l1': 1.9052833646205864e-08, 'lambda_l2': 1.7159611853713213e-07}
#
# final_params.update({
#     "objective": "tweedie",
#     "metric": "l1",
#     # "boosting_type": "gdbt",
#     "n_jobs": -1,
#     "verbosity": -1
# })
#
# seeds = [42, 2023, 0, 777, 999]
# models = []
#
# for seed in seeds:
#     final_params["random_state"] = seed  # 시드만 변경
#
#     print(f"Training with seed {seed}...")
#     model = lgb.train(
#         final_params,
#         train_data,
#         num_boost_round=10000,
#         valid_sets=[valid_data],
#         callbacks=[lgb.early_stopping(stopping_rounds=100)]
#     )
#     base_path = "../../model/"
#     model_path = base_path + f"ensemble_{seed}.pkl"
#     models.append(model)
#     joblib.dump(model, model_path)
#
# # 최종 예측: 5개 모델의 예측값 평균
# preds_list = [model.predict(X_test) for model in models]

import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -------------------------------------------------
# 1. 2021년 테스트 데이터 로드 (위와 동일한 전처리)
# -------------------------------------------------
file_path = "../../data/final_heatmap_lag_without_leakage.csv"
df_raw = pd.read_csv(file_path)
df_raw['datetime'] = pd.to_datetime(df_raw['datetime'])

# 2021년 데이터만 분리
df_test = df_raw[df_raw['datetime'] >= '2021-01-01'].copy().reset_index(drop=True)

y_2021 = df_test.iloc[:, 0].values
X_2021 = df_test.iloc[:, 2:].values

base_path = "../../model/"
scaler_path = base_path+"lgb_scaler.pkl"
print(f"Loading scaler from {scaler_path}...")
scaler = joblib.load(scaler_path)
X_2021_scaled = scaler.transform(X_2021)

seeds = [42, 2023, 0, 777, 999]

preds_2021_list = [] # 각 모델의 예측값을 저장할 리스트

print(f"앙상블 모델 {len(seeds)}개 로드 및 예측 시작...")

for seed in seeds:
    # 파일 경로 생성
    model_path = base_path + f"ensemble_{seed}.pkl"

    # 모델 로드
    loaded_model = joblib.load(model_path)
    print(f"Loaded: {model_path}")

    # 2021년 데이터(Scaled)로 예측 수행
    # 주의: LightGBM 모델 객체는 predict() 메서드 사용
    pred = loaded_model.predict(X_2021_scaled)
    preds_2021_list.append(pred)

# -------------------------------------------------
# 2. Soft Voting (평균내기)
# -------------------------------------------------
# (5, 데이터개수) 형태의 배열을 (데이터개수,) 형태로 평균
final_pred_2021 = np.mean(preds_2021_list, axis=0)

# -------------------------------------------------
# 3. 성능 평가 (MAE, RMSE, R2)
# -------------------------------------------------
mae = mean_absolute_error(y_2021, final_pred_2021)
rmse = mean_squared_error(y_2021, final_pred_2021, squared=False) # RMSE
r2 = r2_score(y_2021, final_pred_2021)

print("="*50)
print(f"Ensemble Result (Seeds: {seeds})")
print(f"Final MAE : {mae:.4f}")
print(f"Final RMSE: {rmse:.4f}")
print(f"Final R2  : {r2:.4f}")
print("="*50)

# -------------------------------------------------
# 4. 시각화 (실제값 vs 예측값)
# -------------------------------------------------
plt.figure(figsize=(15, 6))

# 전체를 다 그리면 너무 빽빽하므로 앞부분 300개만 시각화 (원하시면 슬라이싱 제거하세요)
subset_n = 100
plt.plot(y_2021[:subset_n], label='Actual (2021)', color='blue', alpha=0.6)
plt.plot(final_pred_2021[:subset_n], label='Ensemble Pred', color='red', alpha=0.7, linestyle='--')

plt.title(f"Ensemble Prediction vs Actual (MAE: {mae:.2f})")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()