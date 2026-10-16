# Use the official Apache Airflow image (adjust the version as needed)
FROM apache/airflow:2.6.1

# Switch to root to install additional packages
USER root

# Set non-interactive mode for apt-get
ENV DEBIAN_FRONTEND=noninteractive

# # Install Java (OpenJDK 17 headless), procps (for 'ps') and bash
# RUN apt-get update && \
#     apt-get install -y --no-install-recommends openjdk-17-jdk-headless procps bash && \
#     rm -rf /var/lib/apt/lists/* && \
#     # Ensure Spark’s scripts run with bash instead of dash
#     ln -sf /bin/bash /bin/sh && \
#     # Create expected JAVA_HOME directory and symlink the java binary there
#     mkdir -p /usr/lib/jvm/java-17-openjdk-amd64/bin && \
#     ln -s "$(which java)" /usr/lib/jvm/java-17-openjdk-amd64/bin/java

# Install Java (OpenJDK 17 headless), procps (for 'ps') and bash
# The base image is Debian 11 (bullseye), which is end-of-life: its packages now live on archive.debian.org,
# and its PostgreSQL apt repo was retired (not needed here), so repoint apt before installing
RUN rm -f /etc/apt/sources.list.d/pgdg.list && \
    printf 'deb http://archive.debian.org/debian bullseye main\ndeb http://archive.debian.org/debian bullseye-updates main\ndeb http://archive.debian.org/debian-security bullseye-security main\n' > /etc/apt/sources.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends openjdk-17-jdk-headless procps bash && \
    rm -rf /var/lib/apt/lists/* && \
    # Ensure Spark’s scripts run with bash instead of dash
    ln -sf /bin/bash /bin/sh

# Set JAVA_HOME to the directory expected by Spark
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH=$PATH:$JAVA_HOME/bin

# Set the working directory
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt ./

# Switch to the airflow user before installing Python dependencies
USER airflow

# Install Python dependencies using requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Create a volume mount point for notebooks
VOLUME /app
