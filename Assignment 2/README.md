# End-to-end machine learning for loan default prediction

Predicts, at loan start, the probability that a new loan will be **30+ days past due at month-on-book 6**. The project
builds a PySpark medallion data pipeline (Bronze → Silver → Gold), develops and selects a model (XGBoost champion,
logistic-regression baseline), registers it in MLflow, scores new loans every month, monitors them and applies rules
for retraining. Everything runs in Docker, either as **three Airflow DAGs** or as **two Jupyter notebooks**.

## Requirements

- Docker Desktop with at least **8 GB of memory** for Docker (Settings → Resources).
- The four raw files in `data/`: `lms_loan_daily.csv`, `features_attributes.csv`, `features_financials.csv`,
  `feature_clickstream.csv`.
- Ports 8080 (Airflow) and 8888 (JupyterLab) free.

## Option 1: Airflow (three DAGs)

```bash
docker compose up -d --build airflow-init airflow-webserver airflow-scheduler
```

1. Open **http://localhost:8080** and sign in with **admin / admin**.
2. Unpause the three DAGs: `data_pipeline`, `model_training` and `inference`. They are created paused.
3. They back-fill on their own, in the right order:

| DAG | Schedule | What it does |
|---|---|---|
| `data_pipeline` | monthly, back-fills 2023-01 → 2025-11 | Bronze → Silver → Gold feature store + Gold label store for that month |
| `model_training` | 1 January / 1 July (first run 2025-07), or triggered | waits for the month's data → full-history data validation → split → Gold EDA → feature engineering → model development → registers the models in MLflow (`@champion`, `@challenger`, `@baseline`) → monitoring reference |
| `inference` | monthly, back-fills 2024-07 → 2025-07 | waits for the month's data and for `@champion` → scores the month's new loans → monitoring → governance check (triggers `model_training` if the retraining rules fire) |

On a fresh setup, the inference runs wait until the 2025-07 training run has registered the first model. That's the
first month in which every label the training needs has matured.

**Run time (measured, clean Docker run):** about **6 hours** with Airflow's one-task-at-a-time executor. Almost all of it is the monthly data back-fill (35 months × 4 tasks, each starting its own Spark session); the training run takes about 13 minutes and the 13 inference runs about 10 minutes. No task failed or needed a retry.

Each task is a command such as `python util/pipeline_monthly.py silver --snapshotdate 2024-03-01`, run with the
pipeline's own Python environment inside the Airflow image.

## Option 2: Jupyter notebooks

```bash
docker compose up -d --build jupyter
```

Open **http://localhost:8888**, choose the kernel **CS611 A2 (Py3.11 Spark)**, then run, in order:

1. `script/datapipeline.ipynb`: data contract tests, Bronze, Silver, Gold (full history).
2. `script/mlpipeline.ipynb`: split, Gold EDA, feature engineering, model development and final evaluation, registry,
   batch inference, monitoring, governance.

**Run time:** about 15–20 minutes on its own (measured 42 minutes while an Airflow test shared the machine).

The notebooks in `notebooks/` are the development versions with explanations and every chart. Each one is converted
to the module of the same name in `util/`, and the outputs are verified to be identical.

## Project layout

| Folder / file | Content |
|---|---|
| `config/` | data contract (sources, keys, label definition, allowed predictors) and Silver schema |
| `util/` | pipeline modules: `dataprocessing_bronze/silver/gold.py`, `ml_split.py`, `ml_feature_engineering.py`, `ml_model_development.py`, `ml_deployment.py`, `pipeline_monthly.py` (Airflow tasks) |
| `notebooks/` | development notebooks, one per module |
| `script/` | the two orchestrator notebooks (Option 2) |
| `dags/` | the three Airflow DAGs (Option 1) |
| `Dockerfile`, `airflow/Dockerfile`, `docker-compose.yaml` | Jupyter image, Airflow image (the pipeline runs in its own environment there), services |
| `output_data/` | presentations and the model governance / refresh SOP |
| `datamart/` | everything the pipeline writes (created on the first run, not in git) |

## Main outputs (`datamart/`)

| Path | Content |
|---|---|
| `bronze/`, `silver/` | raw-as-is CSV per month; validated, cleaned Parquet, with reports |
| `gold/feature_store/v1/`, `gold/label_store/30dpd_6mob/` | one row per loan, features as of the loan start date; labels |
| `gold/split/v1/` | train / validation / test / out-of-time assignment |
| `ml/model_development/v1/` | leaderboard, SHAP, final models, model cards, final evaluation, robustness analysis, charts |
| `ml/mlflow/mlflow.db` | MLflow registry (browse with `mlflow ui --backend-store-uri sqlite:///datamart/ml/mlflow/mlflow.db`) |
| `gold/model_predictions/loan_default_30dpd_6mob/` | monthly predictions: probability, high-risk flag, 3 reason codes, model version |
| `ml/deployment/v1/` | inference log, monitoring (drift, performance, alerts), governance decisions, charts |

## Results (out-of-time loans, 2024-07 → 2025-01)

| Model | PR-AUC | ROC-AUC |
|---|---|---|
| Logistic regression baseline | 0.542 | 0.762 |
| **XGBoost champion** (30 features, no clickstream) | **0.619** | **0.807** |

At the chosen threshold, the champion catches 70% of defaulters while flagging about 35% of loans.

## Notes

- **Reproducibility:** seeds are fixed. On Linux (Docker), XGBoost scores differ from the Windows development run by at
  most about 0.005 PR-AUC, because of floating-point differences between the builds; all decisions are the same. Data,
  splits, features, the baseline and random forest are identical.
- **Airflow and notebooks give identical results:** the clean Docker runs of both options produced exactly the same predictions for all 3,526 out-of-time loans.
- **Spark resources:** in Docker, Spark uses 4 cores and a 4 GB driver (`SPARK_MASTER`, `SPARK_DRIVER_MEMORY`).
- **Start again from scratch:** delete `datamart/`, and for Airflow also run `docker compose down -v` (removes the
  Airflow metadata volume).
