import argparse
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pprint
import pyspark
import pyspark.sql.functions as F

from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType

import utils.data_processing_bronze_table
import utils.data_processing_silver_table
import utils.data_processing_gold_table

# to call this script: python gold_label_store.py --snapshotdate "2023-01-01"

def main(snapshotdate, dpd, mob):
    print('\n\n---starting job---\n\n')

    # Initialize SparkSession
    spark = pyspark.sql.SparkSession.builder \
        .appName("dev") \
        .master("local[*]") \
        .getOrCreate()

    # Set log level to ERROR to hide warnings
    spark.sparkContext.setLogLevel("ERROR")

    # load arguments
    date_str = snapshotdate

    # create silver datalake
    silver_loan_daily_directory = "datamart/silver/loan_daily/"

    if not os.path.exists(silver_loan_daily_directory):
        os.makedirs(silver_loan_daily_directory)

    # create gold datalake
    gold_label_store_directory = "datamart/gold/label_store/"

    if not os.path.exists(gold_label_store_directory):
        os.makedirs(gold_label_store_directory)

    # run data processing
    utils.data_processing_gold_table.process_labels_gold_table(date_str, silver_loan_daily_directory, gold_label_store_directory, spark, dpd=dpd, mob=mob)

    # end spark session
    spark.stop()

    print('\n\n---completed job---\n\n')

if __name__ == "__main__":
    # Setup argparse to parse command-line arguments
    parser = argparse.ArgumentParser(description="run job")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")
    parser.add_argument("--dpd", type=int, default=30, help="days past due threshold for label (default: 30)")
    parser.add_argument("--mob", type=int, default=6, help="month on book at which label is observed (default: 6)")

    args = parser.parse_args()

    # Call main with arguments explicitly passed
    main(args.snapshotdate, args.dpd, args.mob)
