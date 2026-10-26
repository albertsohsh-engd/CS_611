"""DAG 2 — model training: every 6 months (January / July) or when triggered (by hand or by the inference DAG).

Scheduled runs wait for the data pipeline's run of the same month. Triggered runs (no matching data run) start
directly. Steps: full-history data validation -> split (14) -> Gold EDA (15-17) -> feature engineering (18-23) ->
model development (24-32) -> register in MLflow and set the aliases (33) -> monitoring reference (35).

The first scheduled run is 2025-07, the first month in which every label the pipeline needs has matured
(development loans up to 2024-06 -> labels by 2024-12; out-of-time loans up to 2025-01 -> labels by 2025-07).
"""
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import BranchPythonOperator
from airflow.sensors.external_task import ExternalTaskSensor

PY = "/opt/pipeline-venv/bin/python"


def run(cmd):
    return f"cd /app && {cmd}"


def scheduled_or_triggered(**context):
    # scheduled runs wait for the month's data; manual / triggered runs use the data that exists
    return "wait_for_data" if context["dag_run"].run_type == "scheduled" else "use_existing_data"


default_args = {"owner": "albertsoh", "retries": 0}

with DAG(
    dag_id="model_training",
    description="Six-monthly (or on-demand) retraining: validation, split, features, models, registry",
    schedule="0 0 1 1,7 *",                     # 1 January and 1 July
    start_date=datetime(2025, 7, 1),
    end_date=datetime(2025, 12, 31),            # project data ends 2025-11: one scheduled run (2025-07)
    catchup=True,
    max_active_runs=1,
    default_args=default_args,
    tags=["cs611", "training"],
    doc_md=__doc__,
) as dag:
    route = BranchPythonOperator(task_id="scheduled_or_triggered", python_callable=scheduled_or_triggered)
    wait_for_data = ExternalTaskSensor(
        task_id="wait_for_data", external_dag_id="data_pipeline", external_task_id="data_ready",
        allowed_states=["success"], mode="reschedule", poke_interval=120, timeout=24 * 3600)
    use_existing_data = EmptyOperator(task_id="use_existing_data")
    full_data_validation = BashOperator(
        task_id="full_data_validation", trigger_rule="none_failed_min_one_success",
        bash_command=run(f"{PY} util/pipeline_monthly.py full_data_validation"),
        execution_timeout=timedelta(hours=1))
    split = BashOperator(task_id="split", bash_command=run(f"{PY} util/ml_split.py"))
    gold_eda = BashOperator(task_id="gold_eda", bash_command=run(f"{PY} util/dataprocessing_gold.py --eda"))
    feature_engineering = BashOperator(task_id="feature_engineering",
                                       bash_command=run(f"{PY} util/ml_feature_engineering.py"),
                                       execution_timeout=timedelta(hours=1))
    model_development = BashOperator(task_id="model_development",
                                     bash_command=run(f"{PY} util/ml_model_development.py"),
                                     execution_timeout=timedelta(hours=2))
    register = BashOperator(task_id="register_models",
                            bash_command=run(f"{PY} util/pipeline_monthly.py register"))
    monitoring_reference = BashOperator(task_id="monitoring_reference",
                                        bash_command=run(f"{PY} util/pipeline_monthly.py monitoring_reference"))

    route >> [wait_for_data, use_existing_data] >> full_data_validation
    full_data_validation >> split >> gold_eda >> feature_engineering >> model_development >> register >> monitoring_reference
