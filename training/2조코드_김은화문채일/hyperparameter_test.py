import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor, Pool
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import optuna
import joblib # 모델 저장을 위해 필요

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

df = df.iloc[169:].reset_index(drop=True)

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

# -------------------------------------------------
# 2. 통합 Objective 함수 (모델 & Loss 선택)
# -------------------------------------------------
def objective(trial):
    # 1. 모델 종류 선택
    model_type = trial.suggest_categorical("model_type", ["lightgbm", "xgboost", "catboost"])

    # 2. 목적 함수(Loss) 종류 선택 (L2=MSE, L1=MAE, Tweedie)
    obj_type = trial.suggest_categorical("obj_type", ["l2", "tweedie", "l1"])

    # === [Case A] LightGBM ===
    if model_type == "lightgbm":
        # 데이터셋 준비
        train_ds = lgb.Dataset(X_train, label=y_train)
        valid_ds = lgb.Dataset(X_test, label=y_test)

        # 공통 파라미터
        param = {
            "metric": "l1", # 평가는 MAE로 통일
            "verbosity": -1,
            "boosting_type": "gbdt",
            "random_state": 42,
            "n_jobs": -1,
            "feature_pre_filter": False,
            "learning_rate": trial.suggest_float("lgb_lr", 0.001, 0.1, log=True),
            "num_leaves": trial.suggest_int("lgb_leaves", 16, 128),
            "max_depth": trial.suggest_int("lgb_depth", 3, 15),
            "min_child_samples": trial.suggest_int("lgb_min_child", 5, 50),
            "feature_fraction": trial.suggest_float("lgb_feat_frac", 0.4, 1.0),
            "bagging_fraction": trial.suggest_float("lgb_bag_frac", 0.4, 1.0),
            "bagging_freq": trial.suggest_int("lgb_bag_freq", 1, 7),
            "lambda_l1": trial.suggest_float("lgb_l1", 1e-8, 10.0, log=True),
            "lambda_l2": trial.suggest_float("lgb_l2", 1e-8, 10.0, log=True),
        }

        # Loss별 파라미터 설정
        if obj_type == "l2":
            param["objective"] = "regression" # L2 (MSE)
        elif obj_type == "l1":
            param["objective"] = "regression_l1" # L1 (MAE)
        elif obj_type == "tweedie":
            param["objective"] = "tweedie"
            param["tweedie_variance_power"] = trial.suggest_float("lgb_tweedie_power", 1.0, 1.9)

        pruning_callback = optuna.integration.LightGBMPruningCallback(trial, "l1")

        model = lgb.train(
            param,
            train_ds,
            num_boost_round=5000,
            valid_sets=[valid_ds],
            callbacks=[lgb.early_stopping(stopping_rounds=100), pruning_callback],
            verbose_eval=False
        )
        preds = model.predict(X_test)

    # === [Case B] XGBoost ===
    elif model_type == "xgboost":
        # 데이터셋 준비
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dvalid = xgb.DMatrix(X_test, label=y_test)

        param = {
            "eval_metric": "mae",
            "verbosity": 0,
            "tree_method": "hist",
            "random_state": 32,
            "eta": trial.suggest_float("xgb_eta", 0.001, 0.3, log=True),
            "max_depth": trial.suggest_int("xgb_depth", 3, 12),
            "min_child_weight": trial.suggest_int("xgb_min_child", 1, 10),
            "subsample": trial.suggest_float("xgb_sub", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("xgb_col", 0.5, 1.0),
            "lambda": trial.suggest_float("xgb_lambda", 1e-8, 10.0, log=True),
            "alpha": trial.suggest_float("xgb_alpha", 1e-8, 10.0, log=True),
        }

        # Loss별 파라미터
        if obj_type == "l2":
            param["objective"] = "reg:squarederror"
        elif obj_type == "l1":
            param["objective"] = "reg:absoluteerror"
        elif obj_type == "tweedie":
            param["objective"] = "reg:tweedie"
            param["tweedie_variance_power"] = trial.suggest_float("xgb_tweedie_power", 1.0, 1.9)

        pruning_callback = optuna.integration.XGBoostPruningCallback(trial, "valid-mae")

        model = xgb.train(
            params=param,
            dtrain=dtrain,
            num_boost_round=5000,
            evals=[(dvalid, "valid")],
            early_stopping_rounds=100,
            callbacks=[pruning_callback],
            verbose_eval=False
        )
        preds = model.predict(dvalid, iteration_range=(0, model.best_iteration + 1))

    # === [Case C] CatBoost ===
    elif model_type == "catboost":
        param = {
            "iterations": 5000,
            "eval_metric": "MAE",
            "random_seed": 42,
            "verbose": 0,
            "allow_writing_files": False,
            "learning_rate": trial.suggest_float("cat_lr", 0.001, 0.3, log=True),
            "depth": trial.suggest_int("cat_depth", 4, 10),
            "l2_leaf_reg": trial.suggest_float("cat_l2", 1e-8, 10.0, log=True),
            "subsample": trial.suggest_float("cat_sub", 0.5, 1.0),
            "bootstrap_type": "Bernoulli",
        }

        # Loss별 파라미터
        if obj_type == "l2":
            param["loss_function"] = "RMSE"
        elif obj_type == "l1":
            param["loss_function"] = "MAE"
        elif obj_type == "tweedie":
            # CatBoost Tweedie 포맷: 'Tweedie:variance_power=1.5'
            power = trial.suggest_float("cat_tweedie_power", 1.0, 1.9)
            param["loss_function"] = f"Tweedie:variance_power={power}"

        pruning_callback = optuna.integration.CatBoostPruningCallback(trial, "MAE")

        # CatBoost는 객체 생성 방식
        model_cat = CatBoostRegressor(**param)
        train_pool = Pool(X_train, y_train)
        test_pool = Pool(X_test, y_test)

        model_cat.fit(
            train_pool,
            eval_set=test_pool,
            early_stopping_rounds=100,
            callbacks=[pruning_callback],
            verbose=False
        )
        preds = model_cat.predict(test_pool)

    # 공통 평가 (MAE)
    mae = mean_absolute_error(y_test, preds)
    return mae

# -------------------------------------------------
# 3. 스터디 실행
# -------------------------------------------------
# 'minimize' -> MAE가 작을수록 좋음
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50)

# -------------------------------------------------
# 4. 결과 확인 및 저장
# -------------------------------------------------
print("="*50)
print("Best MAE:", study.best_value)
print("Best Params:", study.best_params)
print("="*50)

# 베스트 파라미터 및 스터디 저장
joblib.dump(study, "optuna_model_.pkl")
# 스케일러도 나중을 위해 저장
joblib.dump(scaler2, "optuna_scaler.pkl")