import pandas as pd
import numpy as np
import joblib
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# -------------------------------------------------
# 1. 2021년 테스트 데이터 로드 (위와 동일한 전처리)
# -------------------------------------------------
file_path = "../data/final_heatmap_lag_without_leakage.csv"
df_raw = pd.read_csv(file_path)
df_raw['datetime'] = pd.to_datetime(df_raw['datetime'])

# 2021년 데이터만 분리
df_test = df_raw[df_raw['datetime'] >= '2021-01-01'].copy().reset_index(drop=True)

y_2021 = df_test.iloc[:, 0].values
X_2021 = df_test.iloc[:, 2:].values

# -------------------------------------------------
# 2. 저장된 스케일러와 모델 불러오기
# -------------------------------------------------
# 저장했던 파일명에 맞춰주세요
scaler_path = "scaler.pkl"
model_path = "best_model.pkl" # 학습 후 저장한 모델 파일

print(f"Loading scaler from {scaler_path}...")
scaler = joblib.load(scaler_path)

print(f"Loading model from {model_path}...")
model = joblib.load(model_path)

# 2021 데이터 스케일링
X_2021_scaled = scaler.transform(X_2021)

# -------------------------------------------------
# 3. 모델 타입에 따른 예측 수행
# -------------------------------------------------
# 불러온 모델의 타입 확인
print(f"Model Type: {type(model)}")

preds_2021 = None

# Case 1: LightGBM
if isinstance(model, lgb.Booster) or isinstance(model, lgb.LGBMRegressor):
    preds_2021 = model.predict(X_2021_scaled)

# Case 2: XGBoost
elif isinstance(model, xgb.Booster) or isinstance(model, xgb.XGBRegressor):
    # XGBoost는 DMatrix 필요 (네이티브 부스터인 경우)
    if isinstance(model, xgb.Booster):
        dtest_2021 = xgb.DMatrix(X_2021_scaled)
        # best_ntree_limit 등을 사용할 수도 있음
        preds_2021 = model.predict(dtest_2021)
    else:
        preds_2021 = model.predict(X_2021_scaled)

# Case 3: CatBoost
elif isinstance(model, CatBoostRegressor):
    preds_2021 = model.predict(X_2021_scaled)

# -------------------------------------------------
# 4. 결과 평가 및 시각화
# -------------------------------------------------
if preds_2021 is not None:
    mae = mean_absolute_error(y_2021, preds_2021)
    r2 = r2_score(y_2021, preds_2021)

    print("="*50)
    print(f"2021 Test MAE: {mae:.4f}")
    print(f"2021 Test R2 : {r2:.4f}")
    print("="*50)

    # 그래프 그리기
    plt.figure(figsize=(15, 6))
    plt.plot(y_2021, label='Actual', alpha=0.7)
    plt.plot(preds_2021, label='Prediction', alpha=0.7, linestyle='--')
    plt.title(f"2021 Prediction Result (MAE: {mae:.2f})")
    plt.legend()
    plt.show()
else:
    print("모델 타입을 인식하지 못해 예측에 실패했습니다.")