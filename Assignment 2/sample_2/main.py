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
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType, BooleanType

import utils.data_processing_bronze_table
import utils.data_processing_silver_table
import utils.data_processing_gold_table


# Initialize SparkSession
spark = pyspark.sql.SparkSession.builder \
    .appName("dev") \
    .master("local[*]") \
    .getOrCreate()

# Set log level to ERROR to hide warnings
spark.sparkContext.setLogLevel("ERROR")

# set up config
snapshot_date_str = "2023-01-01"

start_date_str = "2023-01-01"
end_date_str = "2024-12-01"

# generate list of dates to process
def generate_first_of_month_dates(start_date_str, end_date_str):
    # Convert the date strings to datetime objects
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
    
    # List to store the first of month dates
    first_of_month_dates = []

    # Start from the first of the month of the start_date
    current_date = datetime(start_date.year, start_date.month, 1)

    while current_date <= end_date:
        # Append the date in yyyy-mm-dd format
        first_of_month_dates.append(current_date.strftime("%Y-%m-%d"))
        
        # Move to the first of the next month
        if current_date.month == 12:
            current_date = datetime(current_date.year + 1, 1, 1)
        else:
            current_date = datetime(current_date.year, current_date.month + 1, 1)

    return first_of_month_dates

dates_str_lst = generate_first_of_month_dates(start_date_str, end_date_str)
print(dates_str_lst)


# ---------------------- 
# Build Bronze Tables
# ---------------------- 

# Loan Management System Data
bronze_lms_directory = "datamart/bronze/lms/"

if not os.path.exists(bronze_lms_directory):
    os.makedirs(bronze_lms_directory)

for date_str in dates_str_lst:
    utils.data_processing_bronze_table.process_bronze_loan_table(date_str, bronze_lms_directory, spark)

# Clickstream Data
bronze_clks_directory = "datamart/bronze/clks/"

if not os.path.exists(bronze_clks_directory):
    os.makedirs(bronze_clks_directory)

for date_str in dates_str_lst:
    utils.data_processing_bronze_table.process_bronze_clickstream_table(date_str, bronze_clks_directory, spark)

# Attributes Data
bronze_attr_directory = "datamart/bronze/attr/"

if not os.path.exists(bronze_attr_directory):
    os.makedirs(bronze_attr_directory)

for date_str in dates_str_lst:
    utils.data_processing_bronze_table.process_bronze_attributes_table(date_str, bronze_attr_directory, spark)

# Financials Data
bronze_fin_directory = "datamart/bronze/fin/"

if not os.path.exists(bronze_fin_directory):
    os.makedirs(bronze_fin_directory)

for date_str in dates_str_lst:
    utils.data_processing_bronze_table.process_bronze_financials_table(date_str, bronze_fin_directory, spark)


# ---------------------- 
# Build Silver Tables
# ---------------------- 

# Loan Management System Data
silver_lms_directory = "datamart/silver/lms/"

if not os.path.exists(silver_lms_directory):
    os.makedirs(silver_lms_directory)

for date_str in dates_str_lst:
    utils.data_processing_silver_table.process_silver_loan_table(date_str, bronze_lms_directory, silver_lms_directory, spark)

# Clickstream Data
silver_clks_directory = "datamart/silver/clks/"

if not os.path.exists(silver_clks_directory):
    os.makedirs(silver_clks_directory)

for date_str in dates_str_lst:
    utils.data_processing_silver_table.process_silver_clickstream_table(date_str, bronze_clks_directory, silver_clks_directory, spark)

# Attributes Data
silver_attr_directory = "datamart/silver/attr/"

if not os.path.exists(silver_attr_directory):
    os.makedirs(silver_attr_directory)

for date_str in dates_str_lst:
    utils.data_processing_silver_table.process_silver_attributes_table(date_str, bronze_attr_directory, silver_attr_directory, spark)

# Financials Data
silver_fin_directory = "datamart/silver/fin/"

if not os.path.exists(silver_fin_directory):
    os.makedirs(silver_fin_directory)

for date_str in dates_str_lst:
    utils.data_processing_silver_table.process_silver_financials_table(date_str, bronze_fin_directory, silver_fin_directory, spark)


# ---------------------- 
# Build Gold Tables
# ---------------------- 

# Build Feature Store
# engagement_tab
gold_clks_directory = "datamart/gold/feature_store/eng/"

if not os.path.exists(gold_clks_directory):
    os.makedirs(gold_clks_directory)

for date_str in dates_str_lst:
    utils.data_processing_gold_table.process_fts_gold_engag_table(date_str, silver_clks_directory, gold_clks_directory, spark)

# cust_fin_risk_tab
gold_fin_directory = "datamart/gold/feature_store/cust_fin_risk/"

if not os.path.exists(gold_fin_directory):
    os.makedirs(gold_fin_directory)

for date_str in dates_str_lst:
    utils.data_processing_gold_table.process_fts_gold_cust_risk_table(date_str, silver_fin_directory, gold_fin_directory, spark)

# Inspect Feature Store Tables
## engagement_tab
folder_path = gold_clks_directory
files_list = [folder_path+os.path.basename(f) for f in glob.glob(os.path.join(folder_path, '*'))]
df = spark.read.parquet(*files_list)
print("row_count:",df.count())
df.show()

## cust_fin_risk_tab
folder_path = gold_fin_directory
files_list = [folder_path+os.path.basename(f) for f in glob.glob(os.path.join(folder_path, '*'))]
df = spark.read.parquet(*files_list)
print("row_count:",df.count())
df.show()



# Build Label Store (based on Loan Mgmt System Data)
gold_label_store_directory = "datamart/gold/label_store/"

if not os.path.exists(gold_label_store_directory):
    os.makedirs(gold_label_store_directory)

# run gold backfill
for date_str in dates_str_lst:
    utils.data_processing_gold_table.process_labels_gold_table(date_str, silver_lms_directory, gold_label_store_directory, spark, dpd = 30, mob = 6)

# Inspect Label Store 
folder_path = gold_label_store_directory
files_list = [folder_path+os.path.basename(f) for f in glob.glob(os.path.join(folder_path, '*'))]
df = spark.read.option("header", "true").parquet(*files_list)
print("row_count:",df.count())
df.show()

# ---------------------- 
# Stop Spark Session
# ---------------------- 

spark.stop()
print("Script finished execution. Stop Spark session.")

    