# Python 3.11 on Debian 12 (bookworm), plus Java 17 for PySpark
FROM python:3.11-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Java 17 (required by Spark), procps (Spark uses 'ps'), bash, libgomp1 (OpenMP runtime for LightGBM / XGBoost)
RUN apt-get update && \
    apt-get install -y --no-install-recommends openjdk-17-jre-headless procps bash libgomp1 && \
    rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH=$PATH:$JAVA_HOME/bin

# Spark resources inside the container (read by util/dataprocessing_bronze.get_spark): 4 cores and a 4 GB driver.
# More parallel tasks writing to the mounted project folder ran out of memory under Docker Desktop.
ENV SPARK_MASTER=local[4] \
    SPARK_DRIVER_MEMORY=4g

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# The notebooks ask for the kernel "cs611-a2" (the name used on the development PC): register it here too
RUN python -m ipykernel install --name cs611-a2 --display-name "CS611 A2 (Py3.11 Spark)"

EXPOSE 8888

# JupyterLab without a token (local use only)
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", \
     "--ServerApp.token=", "--ServerApp.password=", "--notebook-dir=/app"]
