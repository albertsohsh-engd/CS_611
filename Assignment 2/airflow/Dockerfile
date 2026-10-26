# Airflow image for the three DAGs (dags/). Airflow keeps its own Python packages; the pipeline runs in a separate
# virtual environment (/opt/pipeline-venv) with the project's pinned requirements, so the two never conflict
# (e.g. Airflow 2.x needs SQLAlchemy < 2, MLflow / Optuna use newer libraries).
FROM apache/airflow:2.10.5-python3.11

USER root
# Java 17 for Spark, procps (Spark uses 'ps'), libgomp1 (OpenMP for LightGBM / XGBoost)
RUN apt-get update && \
    apt-get install -y --no-install-recommends openjdk-17-jre-headless procps libgomp1 && \
    rm -rf /var/lib/apt/lists/*
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64 \
    PROJECT_DIR=/app \
    SPARK_MASTER=local[4] \
    SPARK_DRIVER_MEMORY=4g \
    MPLBACKEND=Agg \
    MLFLOW_DISABLE_AGENT_HINT=1

# Pipeline environment (same pinned versions as the Jupyter image and the development venv)
COPY requirements.txt /tmp/requirements.txt
RUN python -m venv /opt/pipeline-venv && \
    /opt/pipeline-venv/bin/pip install --no-cache-dir -r /tmp/requirements.txt && \
    chown -R airflow: /opt/pipeline-venv

USER airflow
