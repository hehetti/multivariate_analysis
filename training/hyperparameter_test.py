import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import optuna
from catboost import CatBoostRegressor, Pool

# -------------------------------------------------
# 1. 데이터셋 준비
# -------------------------------------------------
file_path = "../data/final_heatmap_lag_without_leakage.csv"
df_raw = pd.read_csv(file_path)

# 실험 데이터셋(2021)과 분리
df_raw['datetime'] = pd.to_datetime(df_raw['datetime'])
print(f"전체 로드된 데이터셋: {len(df_raw)}")
print(f"날짜 범위: {df_raw['datetime'].min()} ~ {df_raw['datetime'].max()}")

df_test = df_raw[df_raw['datetime'] >= '2021-01-01'].copy().reset_index(drop=True)
df = df_raw[df_raw['datetime']<'2021-01-01'].copy().reset_index(drop=True)

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
# 2. 하이퍼파라미터 시도
# -------------------------------------------------
def lightgbm_objective(trial):
    tweedie_param = {
        "objective": "tweedie",
        "metric": "l1",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "random_state": 42,
        "n_jobs": -1,
        "feature_pre_filter": False,

        # 1. Tweedie Power: 1.0(Poisson) ~ 2.0(Gamma) 사이 실수
        "tweedie_variance_power": trial.suggest_float("tweedie_variance_power", 1.0, 1.9),

        # 2. 학습률: 0.001 ~ 0.1 (로그 스케일로 탐색)
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.1, log=True),

        # 3. 트리 구조
        "num_leaves": trial.suggest_int("num_leaves", 16, 128),
        "max_depth": trial.suggest_int("max_depth", 3, 15),

        # 4. 데이터 샘플링 (과적합 방지)
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 100), # 데이터 적으므로 작게 설정
        "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),

        # 5. 규제 (Regularization)
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
    }
    l2_param = {
        "objective": "regression_l2",
        "metric": "l1",
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.1, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 16, 128),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 30),
        "feature_pre_filter": False,
        "random_state": 42,
        "verbosity": 0,         # 로그 출력 끔
    }

    pruning_callback = optuna.integration.LightGBMPruningCallback(trial, "l1")
    model = lgb.train(
        l2_param,
        train_data,
        num_boost_round=5000,           # 충분히 큰 값
        valid_sets=[valid_data],
        callbacks=[
            lgb.early_stopping(stopping_rounds=100),
            pruning_callback
        ]
    )

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)

    return mae

def xgb_objective(trial):
    xgboost_param = {
        "objective": "reg:squarederror",
        "eval_metric": "l1",
        "tree_method": "hist",  # GPU 사용 시 "gpu_hist"
        "random_state": 32,
        "verbosity": 0,         # 로그 출력 끔

        # 1. 학습률 (Learning Rate)
        "eta": trial.suggest_float("eta", 0.001, 0.3, log=True),

        # 2. 트리 구조 (Tree Structure)
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),

        # 3. 샘플링 (Sampling)
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),

        # 4. 규제 (Regularization)
        "lambda": trial.suggest_float("lambda", 1e-8, 10.0, log=True),
        "alpha": trial.suggest_float("alpha", 1e-8, 10.0, log=True),
    }

    pruning_callback = optuna.integration.XGBoostPruningCallback(trial, "valid-mae")
    model = xgb.train(
        params=xgboost_param,
        dtrain=train_data,
        num_boost_round=5000,
        evals=[(valid_data, "valid")],
        early_stopping_rounds=100,
        callbacks=[pruning_callback],
        verbose_eval=False
    )

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)

    return mae

def cat_objective(trial):
    param = {
        "iterations": 5000,
        "loss_function": "RMSE",
        "eval_metric": "MAE",
        "random_seed": 42,
        "bootstrap_type": "Bernoulli", # subsample을 쓰기 위해 설정
        "verbose": 0,                  # 로그 출력 끔
        "allow_writing_files": False,  # 불필요한 파일 생성 방지

        # 1. 학습률
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.3, log=True),

        # 2. 트리 구조
        "depth": trial.suggest_int("depth", 4, 10),

        # 3. 규제
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-8, 10.0, log=True),

        # 4. 샘플링
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),

        # 5. 추가 파라미터 (선택 사항)
        "random_strength": trial.suggest_float("random_strength", 1e-8, 10.0, log=True),
    }

    # CatBoostRegressor 객체 생성
    model = CatBoostRegressor(**param)

    # Pruning Callback 설정
    pruning_callback = optuna.integration.CatBoostPruningCallback(trial, "MAE")

    # 학습 (Pool 객체 사용 권장)
    train_pool = Pool(X_train, y_train)
    test_pool = Pool(X_test, y_test)

    model.fit(
        train_pool,
        eval_set=test_pool,
        early_stopping_rounds=100,
        callbacks=[pruning_callback],
        verbose=False
    )

    # 검증 데이터 예측 및 MAE 계산
    preds = model.predict(test_pool)
    mae = mean_absolute_error(y_test, preds)

    return mae

# -------------------------------------------------
# 6. 스터디(Study) 생성 및 최적화 실행
# -------------------------------------------------
study = optuna.create_study(direction="minimize") # RMSE는 작을수록 좋으므로 minimize
study.optimize(objective, n_trials=50) # 50번 시도 -> 시간 여유있으면 100번 해볼 것

# -------------------------------------------------
# 7. 결과 확인
# -------------------------------------------------
print("="*50)
print("Best RMSE:", study.best_value)
print("Best Params:", study.best_params)
print("="*50)

# -------------------------------------------------
# 8. 최적의 파라미터 저장
# -------------------------------------------------
best_params = study.best_params
