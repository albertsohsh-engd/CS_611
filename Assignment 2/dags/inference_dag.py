"""DAG 3 — inference (monthly): score the month's new loans with @champion, monitor, check the refresh rules.

Back-fills 2024-07 .. 2025-07: loans start up to 2025-01; the later runs complete the performance monitoring
(labels arrive 6 months after loan start). Each run waits for the month's data and for a model behind @champion
(on a fresh setup, until the training DAG has registered one). If the step-36 refresh rules call for a refit
(a champion trigger persisting 2 consecutive months), the run triggers the model_training DAG.
"""
import json
import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import ShortCircuitOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.sensors.bash import BashSensor
from airflow.sensors.external_task import ExternalTaskSensor

PY = "/opt/pipeline-venv/bin/python"


def task(name):
    return f"cd /app && {PY} util/pipeline_monthly.py {name} --snapshotdate {{{{ ds }}}}"


def retrain_needed(ds, **_):
    # reads the decision written by the governance task (plain JSON: no pipeline libraries needed here)
    path = f"/app/datamart/ml/deployment/v1/governance/decision_{ds}.json"
    with open(path, encoding="utf-8") as f:
        decision = json.load(f)
    print(json.dumps(decision["triggers"], indent=1))
    return bool(decision["retrain_champion"])


default_args = {
    "owner": "albertsoh",
    "depends_on_past": True,                    # monitoring history is built month after month
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}

with DAG(
    dag_id="inference",
    description="Monthly scoring with @champion, monitoring and refresh check",
    schedule="0 0 1 * *",
    start_date=datetime(2024, 7, 1),            # first month after the training period
    end_date=datetime(2025, 7, 1),
    catchup=True,
    max_active_runs=1,
    default_args=default_args,
    tags=["cs611", "inference"],
    doc_md=__doc__,
) as dag:
    wait_for_data = ExternalTaskSensor(
        task_id="wait_for_data", external_dag_id="data_pipeline", external_task_id="data_ready",
        allowed_states=["success"], mode="reschedule", poke_interval=120, timeout=24 * 3600)
    wait_for_champion = BashSensor(
        task_id="wait_for_champion", bash_command=f"cd /app && {PY} util/pipeline_monthly.py champion_ready",
        mode="reschedule", poke_interval=300, timeout=24 * 3600)
    score = BashOperator(task_id="score_new_loans", bash_command=task("score"))
    monitor = BashOperator(task_id="monitor", bash_command=task("monitor"))
    governance = BashOperator(task_id="governance_check", bash_command=task("governance"))
    check = ShortCircuitOperator(task_id="retrain_needed", python_callable=retrain_needed,
                                 ignore_downstream_trigger_rules=True)
    trigger_training = TriggerDagRunOperator(task_id="trigger_training", trigger_dag_id="model_training",
                                             conf={"reason": "monitoring trigger (step 36 rules)"})

    [wait_for_data, wait_for_champion] >> score >> monitor >> governance >> check >> trigger_training
