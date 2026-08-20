# inference/model_loader.py

import os
import joblib

# Required for unpickling
from utils.feature_utils import (
    is_same_region,
    part_of_month,
    part_of_day,
    make_month_object,
    direct_flight,
    duration_category
)

import utils.rbf



ARTIFACTS_DIR = "/app/artifacts"

TRANSFORMER_PATH = os.path.join(
    ARTIFACTS_DIR,
    "latest_column_transformer.joblib"
)

MODEL_PATH = os.path.join(
    ARTIFACTS_DIR,
    "xgb_flight_price_model.joblib"
)

COLUMN_TRANSFORMER = joblib.load(TRANSFORMER_PATH)
MODEL = joblib.load(MODEL_PATH)


