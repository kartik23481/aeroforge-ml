"""
update_artifacts.py

Ensuring backend_artifacts → docker_backend/artifacts sync
and verifies copied files exist.
"""

import os
import shutil
import logging

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

SRC_DIR = os.path.join(PROJECT_ROOT, "backend_artifacts")
DEST_DIR = os.path.join(PROJECT_ROOT, "docker_backend", "artifacts")

EXPECTED = [
    "latest_column_transformer.joblib",
    "xgb_flight_price_model.joblib"
]
 
# Logging
from mlops_src.utils.logger import get_logger

LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
logger = get_logger("update_artifacts", os.path.join(LOG_DIR, "update_artifacts.log"))



def ensure_destination_exists():
    if not os.path.exists(DEST_DIR):
        logger.info(f"Creating destination folder: {DEST_DIR}")
        os.makedirs(DEST_DIR, exist_ok=True)


def verify_sources_exist():
    missing = [f for f in EXPECTED if not os.path.exists(os.path.join(SRC_DIR, f))]
    if missing:
        logger.error(f"Missing BEFORE copy: {missing}")
        raise FileNotFoundError(f"backend_artifacts missing: {missing}")


def clean_destination():
    for item in os.listdir(DEST_DIR):
        path = os.path.join(DEST_DIR, item)
        if os.path.isfile(path):
            logger.info(f"Removing old artifact: {path}")
            os.remove(path)


def copy_artifacts():
    for fname in EXPECTED:
        src = os.path.join(SRC_DIR, fname)
        dest = os.path.join(DEST_DIR, fname)
        shutil.copy2(src, dest)
        logger.info(f"Copied {src} → {dest}")


def validate_destination():
    """
    Verify copied files exist at DEST_DIR.
    Raise error early if missing -> stops CI before Docker build.
    """
    missing_after = [f for f in EXPECTED if not os.path.exists(os.path.join(DEST_DIR, f))]
    
    if missing_after:
        logger.error(f"Missing AFTER copy: {missing_after}")
        raise FileNotFoundError(
            f"ERROR: Files missing in docker_backend/artifacts after copy → {missing_after}"
        )
    
    logger.info("Destination contains all expected artifacts.")


def main():
    logger.info("===== UPDATE ARTIFACTS STARTED =====")

    if not os.path.exists(SRC_DIR):
        raise FileNotFoundError(f"backend_artifacts missing: {SRC_DIR}")

    ensure_destination_exists()
    verify_sources_exist()
    clean_destination()
    copy_artifacts()
    validate_destination()   

    logger.info("===== UPDATE ARTIFACTS COMPLETED =====")


if __name__ == "__main__":
    main()
