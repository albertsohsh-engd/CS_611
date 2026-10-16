from pyspark.sql.functions import col 

def ingest_bronze_tables(spark):
    sources = {
        "clickstream": "data/feature_clickstream.csv",
        "attributes": "data/features_attributes.csv",
        "financials": "data/features_financials.csv",
        "loans": "data/lms_loan_daily.csv"
    }

    for name, path in sources.items():
        print(f"Ingesting {name} from {path}")
        df = spark.read.csv(path, header = True, inferSchema = True)

    #partition time series sources
        if "snapshot_date" in df.columns and name in ["clickstream", "financials"]:
            df.write.partitionBy("snapshot_date").mode("overwrite").parquet(f"datamart/bronze/{name}")
        else: 
            df.write.mode("overwrite").parquet(f"datamart/bronze/{name}")

        print(f"✅Saved to bronze/{name}") 


