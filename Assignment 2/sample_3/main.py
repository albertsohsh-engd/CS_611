# main.py

import os
from pyspark.sql import SparkSession
from utils.bronze_processing import ingest_bronze_tables
from utils.silver_processing import (
    clean_financials_table,
    clean_attributes_table,
    clean_clickstream_table,
    clean_loans_table
)
from utils.gold_processing import (
    build_label_store,
    build_feature_store
)

def create_spark_session():
    spark = SparkSession.builder\
        .appName("LoanFeaturePipeline")\
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    return spark

def init_datamart():
    for layer in ["datamart/bronze", "datamart/silver", "datamart/gold"]:
        os.makedirs(layer, exist_ok=True)

if __name__ == "__main__":
    spark = create_spark_session()
    init_datamart()

    # Bronze
    ingest_bronze_tables(spark)
    print("🥉 Bronze complete!")

    # Silver
    clean_financials_table(spark)
    clean_attributes_table(spark)
    clean_clickstream_table(spark)
    clean_loans_table(spark)
    print("🥈 Silver complete!")

    # Gold
    build_label_store(spark, dpd_cutoff=30, mob_cutoff=6)
    build_feature_store(spark, dpd_cutoff=30, mob_cutoff=6)
    print("🥇 Gold complete!")
