# AeroForge ML: Route-Aware Predictive Pricing System

AeroForge ML is a complete end-to-end machine learning system for flight ticket price prediction, operationalized through a fully reproducible MLOps pipeline. It transforms notebook experimentation into a structured, deployable, CI/CD-driven ML architecture — solving real production challenges: spatial data leakage from route-dependent splits, training-serving skew from divergent feature logic, environment reproducibility across retraining cycles, and intelligent pipeline execution through DVC-managed stage-level dependency tracking.

<p align="left">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/pandas-150458?style=for-the-badge&logo=pandas&logoColor=white"/>
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white"/>
  <img src="https://img.shields.io/badge/XGBoost-FF6600?style=for-the-badge&logo=xgboost&logoColor=white"/>
  <img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white"/>
  <img src="https://img.shields.io/badge/Pydantic-E92063?style=for-the-badge&logo=pydantic&logoColor=white"/>
  <img src="https://img.shields.io/badge/Uvicorn-499848?style=for-the-badge&logo=gunicorn&logoColor=white"/>
  <img src="https://img.shields.io/badge/Git-F05032?style=for-the-badge&logo=git&logoColor=white"/>
  <img src="https://img.shields.io/badge/MLflow-0194E2?style=for-the-badge&logo=mlflow&logoColor=white"/>
  <img src="https://img.shields.io/badge/DVC-945DD6?style=for-the-badge&logo=dvc&logoColor=white"/>
  <img src="https://img.shields.io/badge/DagsHub-FF6B35?style=for-the-badge&logo=dagshub&logoColor=white"/>
  <img src="https://img.shields.io/badge/Docker_Hub-2496ED?style=for-the-badge&logo=docker&logoColor=white"/>
  <img src="https://img.shields.io/badge/GitHub_Actions-2088FF?style=for-the-badge&logo=github-actions&logoColor=white"/>
  <img src="https://img.shields.io/badge/Render-46E3B7?style=for-the-badge&logo=render&logoColor=white"/>
</p>

**Live System Dashboard:** [aeroforge-ml.streamlit.app](https://aeroforge-ml.streamlit.app)

---

## System Objectives

- Learn route-specific pricing structure without overfitting to memorized historical values.
- Preserve global fare ranking across diverse routing configurations — validated by Spearman rank correlation of 0.95.
- Eliminate training-serving skew by using the same fitted `ColumnTransformer` object for both offline training and live FastAPI inference.
- Enforce full reproducibility: DVC for dataset and artifact versioning with DagsHub as remote storage, Docker for environment isolation, GitHub Actions for automated retraining.
- Execute ML pipeline stages intelligently: `dvc.yaml` defines stage-level dependency graphs so only stages affected by a code or data change are re-executed — not the entire pipeline.

---

## Repository Structure

```
.
├── .dvc/
│   ├── .gitignore
│   └── config                        # DagsHub remote URL (committed); credentials via CI secrets
│
├── .github/
│   └── workflows/
│       └── ci.yml                    # CI/CD: dvc pull → dvc repro → dvc push → Docker build → deploy
│
├── mlops_src/                        # ML pipeline used in CI pipeline
│   ├── data_preprocess.py            # Raw data cleaning, feature extraction, stratified splitting
│   ├── feature_pipeline.py           # ColumnTransformer construction, fit, and serialization
│   ├── train.py                      # XGBoost training with MLflow hyperparameter tracking
│   ├── update_artifacts.py           # Validates and syncs artifacts → docker_backend/artifacts/
│   └── utils/
│       ├── logger.py                 # Centralized logger factory
│       ├── feature_utils.py          # Custom sklearn-compatible transformers and feature functions
│       └── rbf.py                    # RouteCreator: (source, destination) → route ID
│
├── docker_backend/                   # Self-contained production inference environment
│   ├── api/
│   │   ├── main.py                   # FastAPI app: GET /healthz, POST /predict
│   │   └── schemas.py                # Pydantic input/output models
│   ├── inference/
│   │   ├── model_loader.py           # Loads transformer + model at container startup
│   │   ├── predict.py                # preprocess → transform → predict orchestration
│   │   └── preprocess.py             # API payload → model-ready DataFrame
│   ├── utils/
│   │   ├── feature_utils.py          # Exact mirror of mlops_src/utils/feature_utils.py
│   │   └── rbf.py                    # Exact mirror of mlops_src/utils/rbf.py
│   ├── artifacts/
│   │   └── .gitkeep                  # Populated by update_artifacts.py before Docker build
│   ├── Dockerfile
│   └── docker_requirements.txt
│
├── data/
│   └── raw/
│       ├── .gitignore                # Ignores flight_price.csv — managed entirely by DVC
│       └── flight_price.csv.dvc      # DVC pointer file (MD5 hash + size) — committed to Git
│
├── notebooks/                        # Offline experimentation (not part of the pipeline)
│   ├── exploratory_data_analysis.ipynb
│   ├── latest_datapreprocessing.ipynb
│   ├── latest_feature_engineering.ipynb
│   └── latest_model_training_tuning.ipynb
|
├── logs/                        # Logs for each pipeline stage
│   ├── data_preprocess.log      # Log file for data_preprocess.py
│   ├── feature_pipeline.log     # Log file for feature_pipeline.py
│   ├── train.log                # Log file for train.py
│   └── update_artifacts.log     # Log file for update_artifacts.py
|
├── .dvcignore
├── .gitignore
├── dvc.lock                          # DVC pipeline state — records MD5 of every input and output
├── dvc.yaml                          # DVC pipeline definition: 3 stages with deps and outs
├── latest_app.py                     # Streamlit live dashboard
└── requirements.txt
```

---

## Core Architecture

### 1. Route-Aware Stratified Data Splitting

Flight prices are structurally route-dependent. A naive random split causes low-frequency routes to disappear entirely from one or more splits, producing an evaluation that does not reflect real deployment conditions — a form of spatial data leakage.

**Implementation in `data_preprocess.py`:**

After cleaning, a composite `route_key = zip(source, destination)` is created. Routes with fewer than 2 samples are filtered out entirely. Two nested stratified splits are then applied:

- First split: 80% remainder / 20% test, stratified by `route_key`
- Second split: 80% train / 20% val on the remainder, stratified by `route_key`
- `route_key` is dropped from all three splits before saving

**Resulting proportions of route-filtered data:**

| Split | Proportion |
|---|---|
| Train | 64% |
| Validation | 16% |
| Test | 20% |

Every route appears proportionally across all three splits, ensuring that validation and test metrics reflect real deployment behavior.

---

### 2. Raw Data Cleaning

All cleaning is performed in `data_preprocess.py` before any feature engineering.

**Column normalization:** All column names are lowercased, stripped, and spaces replaced with underscores. All string-valued columns are lowercased and stripped.

**Airline normalization:** The substrings ` premium economy` and ` business` are stripped from airline names, then the result is title-cased. This ensures that `"IndiGo Business"` and `"IndiGo"` map to the same category before encoding.

**`total_stops` → integer mapping:**

| Raw value | Mapped integer |
|---|---|
| non-stop / non stop / 0 | 0 |
| 1 stop | 1 |
| 2 stops | 2 |
| 3 stops | 3 |
| anything else | 4 |

**`duration` → total minutes:** Parses strings like `"2h 30m"` by splitting on whitespace, extracting hours and minutes separately, and computing `hours × 60 + minutes`. Single-component strings like `"3h"` or `"45m"` are handled by checking for the `h` or `m` suffix.

**Temporal extraction from `dep_time`:** Parsed with `format="%H:%M"`. Extracts `dep_time_hour` and `dep_time_min`. The `dep_time_min` column is carried through to the processed CSVs but is dropped before the transformer is fit (not used as a feature in the current model).

**Temporal extraction from journey date:** Tries column `date_of_journey` first, then `date_of_journey_(dd/mm/yyyy)`. Parses with `dayfirst=True, format="%d/%m/%Y"`. Extracts `dtoj_day`, `dtoj_month`, `dtoj_year`.

**Columns dropped after extraction:** `date_of_journey`, `dep_time`, `route`, `arrival_time`.

**Final cleanup:** All remaining object columns are cast to `str`. Duplicate rows are dropped.

---

### 3. Unified Feature Engineering Pipeline

All feature engineering is centralized in a single Scikit-learn `ColumnTransformer` built and fitted in `mlops_src/feature_pipeline.py`. The serialized fitted object is loaded verbatim in the production inference container. There is no notebook-only logic and no divergence between training and serving transformations.

**Before fitting**, `fit_and_save` performs two preparatory steps on the training DataFrame:

1. Derives `is_weekend` by reconstructing a date from `dtoj_year`, `dtoj_month`, `dtoj_day` and checking `weekday() >= 5`.
2. Drops `dep_time_min` and `dtoj_year` (both served their purpose in derivation; neither is a model feature).

**ColumnTransformer — per-column transformation detail:**

**`airline` → `tf1`**

Pipeline: `RareLabelEncoder(tol=0.1, n_categories=2, replace_with="Other")` → `OneHotEncoder(sparse_output=False, handle_unknown="ignore")`

Airlines appearing in fewer than 10% of training rows, or outside the top 2 most frequent, are grouped as `"Other"` before one-hot encoding.

**`source`, `destination` → `tf2`**

`FeatureUnion` of two parallel branches:

- **Branch 1:** `RouteCreator(route_map)` → `OneHotEncoder(sparse_output=False, handle_unknown="ignore")`

  `RouteCreator` maps `(source, destination)` tuples to a categorical route ID using a hardcoded lookup of 6 known routes. Any route pair outside this mapping receives `"Other"`, which the downstream OHE handles via `handle_unknown="ignore"`.

  The 6 hardcoded routes:

  | Source | Destination | Route ID |
  |---|---|---|
  | delhi | cochin | 1 |
  | kolkata | banglore | 2 |
  | mumbai | hyderabad | 3 |
  | banglore | new delhi | 4 |
  | banglore | delhi | 5 |
  | chennai | kolkata | 6 |

- **Branch 2:** `FunctionTransformer(is_same_region)` — returns 1 binary column (`same_region`). Currently flags routes where both source and destination are in `['delhi', 'new delhi']`.

**`dep_time_hour` → `tf3`**

Pipeline: `FunctionTransformer(part_of_day)` → `OneHotEncoder(handle_unknown="ignore", sparse_output=False)`

`part_of_day` maps departure hour to one of four categories:

| Hour range | Category |
|---|---|
| 4 ≤ hour < 12 | morning |
| 12 ≤ hour < 16 | afternoon |
| 16 ≤ hour < 20 | evening |
| all other hours | night |

**`dtoj_day` → `tf4`**

Pipeline: `FunctionTransformer(part_of_month)` → `OneHotEncoder(handle_unknown="ignore", sparse_output=False)`

| Day range | Category |
|---|---|
| 1 – 10 | early |
| 11 – 20 | mid |
| 21 – 31 | late |

**`dtoj_month` → `tf5`**

Pipeline: `FunctionTransformer(make_month_object)` → `OneHotEncoder(sparse_output=False, handle_unknown="ignore")`

`make_month_object` casts the integer month column to `object` dtype so OHE treats it as a nominal category rather than an ordinal number.

**`is_weekend` → `tf6`**

`SimpleImputer(strategy="most_frequent")` only. Passes the binary 0/1 value through after imputation.

**`duration` → `tf7`**

`FeatureUnion` of two parallel branches:

- **Numeric branch:** `SimpleImputer(strategy="median")` — passes raw duration in minutes after median imputation.
- **Categorical branch:** `FunctionTransformer(duration_category)` → `OneHotEncoder(handle_unknown="ignore", sparse_output=False)`

  `duration_category` thresholds: `< 180 min` → short, `180 – 400 min` → medium, `>= 400 min` → long.

The two branches are concatenated by `FeatureUnion`, producing one numeric column and three OHE columns per row.

**`total_stops` → `tf8`**

Pipeline: `SimpleImputer(strategy="most_frequent")` → `FunctionTransformer(direct_flight)`

`direct_flight` adds an `is_direct_flight` column (`total_stops == 0` → 1, else 0) while **retaining the original `total_stops` column**. The output is therefore 2 columns: the imputed stop count and the binary direct-flight flag.

**`additional_info` → `tf9`**

Outer pipeline: `SimpleImputer(strategy="constant", fill_value="unknown")` → inner pipeline

Inner pipeline: `ToDataFrame(["additional_info"])` → `RareLabelEncoder(tol=0.2, n_categories=2, replace_with="Other")` → `OneHotEncoder(handle_unknown="ignore", sparse_output=False)`

`ToDataFrame` is required here because `RareLabelEncoder` from `feature_engine` expects a named DataFrame, not a numpy array. The `SimpleImputer` upstream outputs a numpy array, so `ToDataFrame` restores the column name before `feature_engine` sees it.

**`remainder="drop"`** — all columns not explicitly listed in the transformer are silently discarded. This includes `dtoj_year`, which is present in the input DataFrame passed to `transform` at inference time but is intentionally excluded from the feature set.

---

### 4. DVC Pipeline Definition (`dvc.yaml`)

The ML pipeline is defined as a stage graph in `dvc.yaml`. This is the architectural separation between a DevOps CI orchestrator (GitHub Actions) and an ML pipeline tool (DVC): GitHub Actions triggers `dvc repro`; DVC decides which stages to execute based on dependency state.

```yaml
stages:
  preprocess:
    cmd: python mlops_src/data_preprocess.py --input data/raw/flight_price.csv --out_dir data/processed
    deps:
      - data/raw/flight_price.csv
      - mlops_src/data_preprocess.py
    outs:
      - data/processed/cleaned_flight_data.csv
      - data/processed/train_data.csv
      - data/processed/val_data.csv
      - data/processed/test_data.csv

  feature_pipeline:
    cmd: python mlops_src/feature_pipeline.py --data data/processed/train_data.csv --artifacts backend_artifacts
    deps:
      - data/processed/train_data.csv
      - mlops_src/feature_pipeline.py
      - mlops_src/utils/feature_utils.py
      - mlops_src/utils/rbf.py
    outs:
      - backend_artifacts/latest_column_transformer.joblib

  train:
    cmd: python mlops_src/train.py
    deps:
      - data/processed/train_data.csv
      - data/processed/val_data.csv
      - backend_artifacts/latest_column_transformer.joblib
      - mlops_src/train.py
    outs:
      - backend_artifacts/xgb_flight_price_model.joblib
```

**Stage skipping logic:** After every successful `dvc repro`, DVC writes the MD5 hash of every declared `dep` and `out` into `dvc.lock`, which is committed to Git. On the next CI run, `dvc repro` recomputes the MD5 of each `dep`, compares against `dvc.lock`, and skips any stage whose inputs are unchanged — restoring its outputs from the DagsHub remote cache instead of re-executing it.

**Practical impact by change type:**

| Change Made | Stages Re-executed |
|---|---|
| `flight_price.csv` data | preprocess → feature_pipeline → train |
| `data_preprocess.py` | preprocess → feature_pipeline → train |
| `feature_pipeline.py` or `feature_utils.py` or `rbf.py` | feature_pipeline → train |
| `train.py` only | train only |
| `docker_backend/` only | none (all stages skipped) |

---

### 5. Data and Artifact Versioning with DVC + DagsHub

DVC tracks data and pipeline artifacts independently from Git.

**Raw dataset:** `data/raw/flight_price.csv` is not committed to Git. It is managed exclusively by DVC:

- `data/raw/flight_price.csv.dvc` — a 5-line pointer file containing the MD5 hash and file size — is committed to Git.
- The actual CSV resides in DagsHub remote storage (`https://dagshub.com/kartiksri2005/aeroforge-ml.dvc`).
- Updating the dataset requires: `dvc add data/raw/flight_price.csv` → `dvc push` → `git add data/raw/flight_price.csv.dvc` → `git push`.

**Pipeline outputs:** All `outs` declared in `dvc.yaml` (processed CSVs, `.joblib` artifacts) are DVC-cached and pushed to DagsHub after every successful CI run. This enables stage-level output restoration: if inputs to a stage haven't changed, DVC fetches its outputs from the DagsHub cache rather than re-running the stage.

**Remote configuration (`.dvc/config`):**

```ini
[core]
    remote = dagshub
['remote "dagshub"']
    url = https://dagshub.com/kartiksri2005/aeroforge-ml.dvc
```

Credentials (`user`, `password`) are injected at CI runtime via `DAGSHUB_TOKEN` GitHub Secret using `dvc remote modify --local`. They are never stored in the repository.

---

## MLOps Pipeline

All pipeline stages are defined in `dvc.yaml` and orchestrated by a single `dvc repro` command. CI does not call individual scripts — it delegates orchestration entirely to DVC.

**Execution order (when all stages run):**

**Stage 1 — Preprocess**

```bash
python mlops_src/data_preprocess.py \
    --input data/raw/flight_price.csv \
    --out_dir data/processed/
```

Runs the full cleaning sequence. Outputs `cleaned_flight_data.csv`, `train_data.csv`, `val_data.csv`, `test_data.csv` to `data/processed/`.

**Stage 2 — Feature Pipeline**

```bash
python mlops_src/feature_pipeline.py \
    --data data/processed/train_data.csv \
    --artifacts backend_artifacts/
```

Derives `is_weekend`, drops `dep_time_min` and `dtoj_year`, builds and fits the `ColumnTransformer` on training data, saves `backend_artifacts/latest_column_transformer.joblib`.

**Stage 3 — Train**

```bash
python mlops_src/train.py
```

Paths hardcoded inside `train.py`. Applies `prepare_df` preprocessing, loads the fitted transformer, transforms both train and val splits, and runs a manual grid search over three XGBoost configurations under MLflow experiment `"flight-price-training"`:

| n_estimators | learning_rate | max_depth |
|---|---|---|
| 150 | 0.10 | 6 |
| 200 | 0.08 | 7 |
| 250 | 0.07 | 8 |

Fixed across all runs: `subsample=0.9`, `colsample_bytree=0.9`, `random_state=42`.

Each run logs its parameters and validation RMSE to MLflow. The configuration with the lowest RMSE is selected. The best model is saved to disk via `joblib`.

MLflow run storage: `./mlruns` (local, controlled by `MLFLOW_TRACKING_URI`).

**Stage 4 — Artifact Sync (outside DVC, in CI)**

```bash
python mlops_src/update_artifacts.py
```

Source: `backend_artifacts/`. Destination: `docker_backend/artifacts/`.

Execution order inside the script:
1. Creates destination directory if it does not exist.
2. Verifies both expected files exist in source — raises `FileNotFoundError` if either is missing.
3. Removes all existing files from the destination directory.
4. Copies both `.joblib` files to destination.
5. Validates both files exist in destination — raises `FileNotFoundError` if either is missing after copy.

This script acts as a hard gate in CI: a missing artifact will fail the job before any Docker build is attempted.

---

## Model Performance

The final model is an **XGBoost Regressor** selected by lowest validation RMSE.

**Global metrics:**

| Metric | Random Forest | XGBoost |
|---|---|---|
| R² | 0.85 | **0.86** |
| RMSE (₹) | 2,524 | **2,380** |
| MAE (₹) | 1,750 | **1,620** |
| Spearman Rank Correlation | — | **0.95** |

A Spearman rank correlation of 0.95 indicates that the model correctly orders fares relative to each other with high consistency — it learned structural pricing patterns rather than memorizing point values.

**Error distribution by price segment:**

| Segment | Avg Error (₹) | % Error | Sample Count |
|---|---|---|---|
| Budget (< 5k) | 445 | ~11% | 1,533 |
| Mid (5k – 10k) | 892 | ~12% | 2,470 |
| Upper-mid (10k – 20k) | 963 | ~7–8% | 2,592 |
| Premium (> 20k) | 6,437 | ~25% | 94 |

The premium segment error is a direct consequence of 94 training samples against 6,500+ in lower bands. Standard regression mean-bias compresses high-value predictions toward the training mean. No oversampling or segment-specific loss weighting was applied — the decision was to maintain stable majority-market performance rather than inflate metrics on statistically insignificant tail samples.

---

## Inference Architecture

`docker_backend/` is a fully self-contained inference environment with zero runtime dependency on `mlops_src/`.

**Container startup (`model_loader.py`):**

`latest_column_transformer.joblib` and `xgb_flight_price_model.joblib` are loaded from `/app/artifacts/` into module-level globals `COLUMN_TRANSFORMER` and `MODEL` at import time. All subsequent requests reuse these in-memory objects with no per-request I/O.

`model_loader.py` explicitly imports every custom class (`is_same_region`, `part_of_month`, `part_of_day`, `make_month_object`, `direct_flight`, `duration_category`, `ToDataFrame`, `RouteCreator`) before calling `joblib.load()`. This is required because `joblib` uses `pickle` for deserialization — the class definitions must be importable at the exact same module path they were serialized from. The `docker_backend/utils/` directory is a verbatim mirror of `mlops_src/utils/` for this reason.

**Request lifecycle:**

1. `POST /predict` receives JSON, validated by Pydantic `FlightInput`.
2. `predict_price(payload.dict())` is called.
3. `preprocess_input()` converts the dict to a DataFrame, normalizes `airline`, `source`, `destination`, `additional_info` to lowercase + stripped, derives `dtoj_day`, `dtoj_month`, `dtoj_year`, `is_weekend` from the `date` field, then drops `date` and `dep_time_min`.
4. `COLUMN_TRANSFORMER.transform()` applies the identical fitted transformations from training. `dtoj_year` is present in the DataFrame at this point but is silently discarded by `remainder="drop"`.
5. `MODEL.predict()` returns a scalar, cast to `float` and returned as `PredictionResponse`.

---

## API Reference

**Health check**

```
GET /healthz
```

Response:
```json
{"status": "ok"}
```

**Price prediction**

```
POST /predict
Content-Type: application/json
```

Request schema (`FlightInput`):

| Field | Type | Notes |
|---|---|---|
| `airline` | string | Case-insensitive; normalized to lowercase internally |
| `source` | string | Case-insensitive; departure city |
| `destination` | string | Case-insensitive; arrival city |
| `duration` | integer | Total flight duration in minutes |
| `total_stops` | integer | Number of stops (0 = non-stop) |
| `additional_info` | string | Case-insensitive; e.g. "No info", "In-flight meal" |
| `dep_time_hour` | integer | Departure hour in 24-hour format (0–23) |
| `date` | date | Journey date as `YYYY-MM-DD`; used to derive day, month, year, weekend flag |

Example request:

```json
{
  "airline": "IndiGo",
  "source": "Delhi",
  "destination": "Cochin",
  "duration": 180,
  "total_stops": 0,
  "additional_info": "No info",
  "dep_time_hour": 14,
  "date": "2026-06-15"
}
```

Response:

```json
{
  "predicted_price": 6450.50
}
```

---

## Docker

**Dockerfile breakdown:**

- Base image: `python:3.10-slim`
- System packages: `cmake`, `build-essential` — required for XGBoost source compilation
- Dependencies: `docker_requirements.txt` installed first, then `xgboost==3.0.2` installed separately with `--no-binary=xgboost` to compile from source and avoid binary ABI issues on the slim image
- Copied into `/app/`: `api/`, `inference/`, `utils/`, `artifacts/`
- Exposed port: `8000`
- Entrypoint: `uvicorn api.main:app --host 0.0.0.0 --port 8000`

`artifacts/` is populated by `update_artifacts.py` before building the image. The `.gitkeep` file holds the directory in version control while keeping it empty until the sync step runs.

---

## CI/CD Pipeline

Defined in `.github/workflows/ci.yml`. Triggered on every push to `main`. Single job `mlops` on `ubuntu-latest`.

Environment: `MLFLOW_TRACKING_URI: file:./mlruns`, `PYTHONPATH: .`

**Step-by-step execution:**

1. **Checkout** — `actions/checkout@v3` with `persist-credentials: true` (required for the dvc.lock commit-back step)
2. **Python setup** — `actions/setup-python@v4` with Python 3.10
3. **Install dependencies** — `pip install -r requirements.txt` + `pip install "dvc[http]"`
4. **Configure DVC remote** — injects DagsHub credentials from `DAGSHUB_TOKEN` secret using `dvc remote modify --local`; credentials written to `.dvc/config.local` (ephemeral, never committed)
5. **DVC pull** — `dvc pull` downloads `flight_price.csv` from DagsHub (via `.dvc` pointer) and restores any previously cached pipeline outputs (processed CSVs, `.joblib` files) from DagsHub cache
6. **DVC repro** — `dvc repro` compares MD5 of all declared `deps` against `dvc.lock`; re-executes only stages with changed inputs; skips all others by restoring their outputs from DagsHub cache
7. **DVC push** — `dvc push` uploads all new or updated pipeline outputs to DagsHub remote cache
8. **Commit dvc.lock** — `git commit dvc.lock [skip ci]` pushes the updated pipeline state back to GitHub; `[skip ci]` tag prevents infinite CI loop
9. **Artifact sync** — `update_artifacts.py` validates and copies both `.joblib` files to `docker_backend/artifacts/`; CI fails here if either is missing before Docker build is attempted
10. **Docker build + push** — logs into Docker Hub with `DOCKERHUB_USERNAME` and `DOCKERHUB_TOKEN` secrets, builds from `./docker_backend`, tags as `latest`, pushes
11. **Deploy to Render** — `curl POST` to Render deploy API with `RENDER_TOKEN` and `RENDER_SERVICE_ID` secrets, triggering a redeploy of the running service with the new image

**Required GitHub secrets:**

| Secret | Purpose |
|---|---|
| `DAGSHUB_TOKEN` | DagsHub access token for DVC remote authentication |
| `DOCKERHUB_USERNAME` | Docker Hub account username |
| `DOCKERHUB_TOKEN` | Docker Hub access token |
| `RENDER_TOKEN` | Render API bearer token |
| `RENDER_SERVICE_ID` | Render service identifier |

---

## Local Setup

**Prerequisites:** Python 3.10, Git, Docker, DVC

```bash
git clone https://github.com/kartik23481/aeroforge-ml.git
cd aeroforge-ml
pip install -r requirements.txt
pip install "dvc[http]"
```

**Pull the dataset from DagsHub:**

```bash
dvc remote modify dagshub --local auth basic
dvc remote modify dagshub --local user <your_dagshub_username>
dvc remote modify dagshub --local password <your_dagshub_token>
dvc pull
```

**Run the full pipeline:**

```bash
dvc repro
```

DVC executes only stages whose inputs have changed. On a clean clone, all three stages run in sequence.

**Sync artifacts and run inference container:**

```bash
python mlops_src/update_artifacts.py

cd docker_backend
docker build -t aeroprice-backend .
docker run -p 8000:8000 aeroprice-backend
```

**Test the API:**

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "airline": "IndiGo",
    "source": "Delhi",
    "destination": "Cochin",
    "duration": 180,
    "total_stops": 0,
    "additional_info": "No info",
    "dep_time_hour": 14,
    "date": "2026-06-15"
  }'
```

**View MLflow experiment runs:**

```bash
mlflow ui --backend-store-uri ./mlruns
# Opens at http://localhost:5000
```

**Update the dataset:**

```bash
# Replace data/raw/flight_price.csv with new data, then:
dvc add data/raw/flight_price.csv
dvc push
git add data/raw/flight_price.csv.dvc
git commit -m "data: update flight price dataset"
git pull
git push
# GitHub Actions triggers: all 3 pipeline stages re-execute, new model deployed automatically
```

---

## Why This Is Production-Grade

**No training-serving skew.** The identical fitted `ColumnTransformer` is used in both offline training and live inference. The custom transformer classes are importable at the same module paths in both environments via the `docker_backend/utils/` mirror. Feature logic cannot diverge silently.

**Route-aware evaluation.** Stratified splitting by `route_key` guarantees that every route appears proportionally across train, val, and test. Metrics are not inflated by routes that only appear in training data.

**Intelligent pipeline execution.** `dvc.yaml` defines an explicit dependency graph. DVC tracks MD5 hashes of every input and output in `dvc.lock`. Only stages whose inputs have changed are re-executed — the rest are restored from DagsHub remote cache. A change to `train.py` does not trigger data preprocessing or feature pipeline re-execution.

**Data versioned outside Git.** `flight_price.csv` is stored in DagsHub and tracked by DVC — not committed to Git. Git holds only the 5-line `.dvc` pointer file. Updating the dataset requires `dvc add` + `dvc push` + committing the pointer, triggering a full pipeline rerun automatically.

**Artifact validation as a CI gate.** `update_artifacts.py` checks for both `.joblib` files before and after the copy step. A missing artifact raises `FileNotFoundError` and fails the CI job before any Docker build runs, preventing deployment of a broken image.

**Decoupled inference environment.** `docker_backend/` has no import dependency on `mlops_src/` at runtime. It has its own requirements file, its own utility copies, and loads exclusively from `/app/artifacts/`. Retraining does not touch the running container until an explicit redeploy.

**Full experiment traceability.** Every training run logs hyperparameters and validation RMSE to MLflow. Every dataset version is tracked by DVC with DagsHub as remote. The complete lineage of any deployed model — data version, hyperparameters, validation metric — is recoverable from the repository state alone.

---

## Author

**Kartik Srivastava**  
AI & Machine Learning Enthusiast  
🔗 LinkedIn: https://www.linkedin.com/in/kartik-srivastava-462609285/
