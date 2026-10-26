"""DAG 1 — data pipeline (monthly): Bronze -> Silver -> Gold feature store + label store for the run month.

Back-fills every month from 2023-01 to 2025-11 (catchup). Each task runs one month only
(util/pipeline_monthly.py, developed and verified in notebooks/pipeline_monthly.ipynb): month by month it rebuilds
exactly the same data as the full build in script/datapipeline.ipynb.
"""
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator

PY = "/opt/pipeline-venv/bin/python"          # the pipeline's own Python environment inside the Airflow image


def task(name):
    return f"cd /app && {PY} util/pipeline_monthly.py {name} --snapshotdate {{{{ ds }}}}"


default_args = {
    "owner": "albertsoh",
    "depends_on_past": True,                    # months are processed in order (Gold uses all earlier snapshots)
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}

with DAG(
    dag_id="data_pipeline",
    description="Monthly Bronze -> Silver -> Gold (feature store, label store) for one snapshot month",
    schedule="0 0 1 * *",                       # 00:00 on the 1st of every month
    start_date=datetime(2023, 1, 1),
    end_date=datetime(2025, 11, 1),
    catchup=True,                               # back-fill 2023-01 .. 2025-11
    max_active_runs=1,
    default_args=default_args,
    tags=["cs611", "data"],
    doc_md=__doc__,
) as dag:
    start = EmptyOperator(task_id="start")
    bronze = BashOperator(task_id="bronze", bash_command=task("bronze"))
    silver = BashOperator(task_id="silver", bash_command=task("silver"))
    gold_feature_store = BashOperator(task_id="gold_feature_store", bash_command=task("gold_features"))
    gold_label_store = BashOperator(task_id="gold_label_store", bash_command=task("gold_labels"))
    data_ready = EmptyOperator(task_id="data_ready")   # the training and inference DAGs wait for this task

    start >> bronze >> silver >> [gold_feature_store, gold_label_store] >> data_ready
