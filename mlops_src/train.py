"""
mlops_src/train.py
"""
# Tuning and training script 
import os
import sys
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import joblib

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


# LOGGING
from mlops_src.utils.logger import get_logger

LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
logger = get_logger("train", os.path.join(LOG_DIR, "train.log"))
logger.info("===== TRAINING STARTED =====")


# PATHS

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
BACKEND_ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "backend_artifacts")

TRAIN_PATH = os.path.join(DATA_DIR,"processed", "train_data.csv")
VAL_PATH = os.path.join(DATA_DIR, "processed", "val_data.csv")

TRANSFORMER_PATH = os.path.join(BACKEND_ARTIFACTS_DIR, "latest_column_transformer.joblib")
MODEL_PATH = os.path.join(BACKEND_ARTIFACTS_DIR, "xgb_flight_price_model.joblib")

logger.info(f"Artifacts directory: {BACKEND_ARTIFACTS_DIR}")

os.makedirs(BACKEND_ARTIFACTS_DIR, exist_ok=True)



def prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Preparing dataframe for transformation...")

    date = pd.to_datetime(df.rename(columns={
        "dtoj_year": "year",
        "dtoj_month": "month",
        "dtoj_day": "day",
    })[["year", "month", "day"]])

    df = df.assign(is_weekend=(date.dt.weekday >= 5).astype(int))

    df = df.drop(columns=["dep_time_min", "dtoj_year"], errors="ignore")

    return df



# LOADING DATA

logger.info("Loading train & val datasets...")

train_df = pd.read_csv(TRAIN_PATH)
val_df = pd.read_csv(VAL_PATH)

logger.info(f"Train shape: {train_df.shape}")
logger.info(f"Val shape:   {val_df.shape}")

X_train = train_df.drop(columns=["price"])
y_train = train_df["price"]

X_val = val_df.drop(columns=["price"])
y_val = val_df["price"]

X_train = prepare_df(X_train.copy())
X_val = prepare_df(X_val.copy())



# LOADING TRANSFORMER

if not os.path.exists(TRANSFORMER_PATH):
    logger.error("Transformer NOT FOUND. Run feature_pipeline.py first!")
    raise FileNotFoundError(f"Missing transformer → {TRANSFORMER_PATH}")

logger.info(f"Loading transformer: {TRANSFORMER_PATH}")
column_transformer = joblib.load(TRANSFORMER_PATH)

logger.info("Transforming train & val data...")
X_train_tf = column_transformer.transform(X_train)
X_val_tf = column_transformer.transform(X_val)

 

# MLflow AUTO TUNING

logger.info("Starting MLflow experiment: flight-price-training")
mlflow.set_experiment("flight-price-training")

search_space = [
    {"n_estimators": 150, "learning_rate": 0.1, "max_depth": 6},
    {"n_estimators": 200, "learning_rate": 0.08, "max_depth": 7},
    {"n_estimators": 250, "learning_rate": 0.07, "max_depth": 8}
]

best_rmse = float("inf")
best_model = None
best_params = None

logger.info("Beginning hyperparameter tuning...")

for params in search_space:
    with mlflow.start_run():
        mlflow.log_params(params)
        logger.info(f"Trying params: {params}")

        model = XGBRegressor(
            **params,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42
        )

        model.fit(X_train_tf, y_train)
        preds = model.predict(X_val_tf)
        rmse = np.sqrt(mean_squared_error(y_val, preds))

        mlflow.log_metric("rmse", rmse)
        logger.info(f"RMSE: {rmse}")

        if rmse < best_rmse:
            best_rmse = rmse
            best_model = model
            best_params = params


logger.info(f"BEST PARAMS: {best_params}")
logger.info(f"BEST RMSE:   {best_rmse}")

with mlflow.start_run():
    mlflow.log_params(best_params)
    mlflow.log_metric("best_rmse", best_rmse)
    mlflow.sklearn.log_model(best_model, "best-xgb-model")

joblib.dump(best_model, MODEL_PATH)
logger.info(f"Saved best model → {MODEL_PATH}")

logger.info("===== TRAINING COMPLETED SUCCESSFULLY =====")

