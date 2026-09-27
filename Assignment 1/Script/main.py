# %% [markdown]
# This the full code for data pipeline. 
# Section 1 - Create bronze layer 
# Section 2 - EDA 
# Section 3 - Create silver layer - feature engineering 
# Section 4 - Create gold layer - label  store 
# Section 5 - Model dataset - split into train / valid / test / OOT by loan start date


# %%
# Section 1 Create bronze layer 
# Copy all files from data 

import os
import sys
import glob

# Ensure PySpark can find a JVM and uses this notebook's Python interpreter
# for its worker processes, regardless of the shell environment it's launched from.
os.environ.setdefault("JAVA_HOME", r"C:\Program Files\Eclipse Adoptium\jdk-17.0.20.101-hotspot")
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

import pyspark
import pyspark.sql.functions as F

# Initialize SparkSession
spark = pyspark.sql.SparkSession.builder \
    .appName("dev") \
    .master("local[*]") \
    .getOrCreate()

# Set log level to ERROR to hide warnings
spark.sparkContext.setLogLevel("ERROR")

# create datamart folder
datamart_directory = "../datamart/"

if not os.path.exists(datamart_directory):
    os.makedirs(datamart_directory)

# create bronze folder
bronze_directory = os.path.join(datamart_directory, "bronze/")

if not os.path.exists(bronze_directory):
    os.makedirs(bronze_directory)

# source data directory
data_directory = "../Data/"

# copy every file in data/ into datamart/bronze/, renamed with a "bronze_" prefix
source_files = glob.glob(os.path.join(data_directory, "*.csv"))

for file_path in source_files:
    filename = os.path.basename(file_path)

    # load data
    df = spark.read.csv(file_path, header=True, inferSchema=True)
    print(filename, 'row count:', df.count())

    # save to bronze layer in datamart
    bronze_filename = "bronze_" + filename
    bronze_filepath = os.path.join(bronze_directory, bronze_filename)
    df.toPandas().to_csv(bronze_filepath, index=False)
    print('saved to:', bronze_filepath)


# %%
# Section 2 - EDA 

bronze_directory = "../datamart/bronze/"

df_bronze_clickstream = spark.read.csv(bronze_directory + "bronze_feature_clickstream.csv", header=True, inferSchema=True)
df_bronze_attributes = spark.read.csv(bronze_directory + "bronze_features_attributes.csv", header=True, inferSchema=True)
df_bronze_financials = spark.read.csv(bronze_directory + "bronze_features_financials.csv", header=True, inferSchema=True)
df_bronze_lms = spark.read.csv(bronze_directory + "bronze_lms_loan_daily.csv", header=True, inferSchema=True)


# %%
dataframes_bronze = {
    "df_bronze_clickstream": df_bronze_clickstream,
    "df_bronze_attributes": df_bronze_attributes,
    "df_bronze_financials": df_bronze_financials,
    "df_bronze_lms": df_bronze_lms,
}

for name, df in dataframes_bronze.items():
    print(f"===== {name} =====")

    print(f"--- {name}.show(5) ---")
    df.show(5)

    row_count = df.count()
    col_count = len(df.columns)
    print(f"--- {name}.shape --- ({row_count}, {col_count})")

    print(f"--- {name}.columns ---")
    print(df.columns)

    print(f"--- {name}.printSchema() ---")
    df.printSchema()

    print(f"--- {name}.describe() ---")
    df.describe().show()

    print()


# %%
# Section 3 - Create silver 
# Clean data 

bronze_directory = "../datamart/bronze/"

df_silver_clickstream = spark.read.csv(bronze_directory + "bronze_feature_clickstream.csv", header=True, inferSchema=True)
df_silver_attributes = spark.read.csv(bronze_directory + "bronze_features_attributes.csv", header=True, inferSchema=True)
df_silver_financials = spark.read.csv(bronze_directory + "bronze_features_financials.csv", header=True, inferSchema=True)
df_silver_lms = spark.read.csv(bronze_directory + "bronze_lms_loan_daily.csv", header=True, inferSchema=True)

dataframes_silver = {
    "df_silver_clickstream": df_silver_clickstream,
    "df_silver_attributes": df_silver_attributes,
    "df_silver_financials": df_silver_financials,
    "df_silver_lms": df_silver_lms,
}


# %%
# check isnull / missing values and duplicates before cleaning

for name, df in dataframes_silver.items():
    print(f"===== {name} =====")

    row_count = df.count()

    print(f"--- {name} isnull count per column ---")
    null_counts = df.select([
        F.count(F.when(F.col(c).isNull(), c)).alias(c) for c in df.columns
    ])
    null_counts.show()

    print(f"--- {name} missing % per column ---")
    null_pct = df.select([
        (F.count(F.when(F.col(c).isNull(), c)) / row_count * 100).alias(c) for c in df.columns
    ])
    null_pct.show()

    print(f"--- {name} duplicate rows ---")
    duplicate_count = row_count - df.dropDuplicates().count()
    print(duplicate_count)

    print()


# %%
# clean data: fix junk characters, placeholders, impossible values and data types
# bad values are set to null / "Unknown" instead of dropping the whole row

from pyspark.sql.types import IntegerType, FloatType, DoubleType, StringType, DateType


def clean_numeric(col_name, dtype):
    # strip "_" from values like "32_" or "52312.68_"
    # values starting with "_" are placeholders ("_", "__10000__", "__-333...__") -> null
    c = F.trim(F.col(col_name).cast(StringType()))
    return F.when(c.startswith("_"), None) \
            .otherwise(F.regexp_replace(c, "_", "")) \
            .cast(DoubleType()).cast(dtype)


def keep_in_range(df, col_name, low, high):
    # impossible values / extreme outliers -> null
    return df.withColumn(col_name, F.when(F.col(col_name).between(low, high), F.col(col_name)))


def replace_placeholder(df, col_name, placeholder, new_value="Unknown"):
    return df.withColumn(col_name, F.when(F.col(col_name) == placeholder, new_value).otherwise(F.col(col_name)))


# ---------- attributes ----------
df = dataframes_silver["df_silver_attributes"]

df = df.withColumn("Age", clean_numeric("Age", IntegerType()))
df = keep_in_range(df, "Age", 0, 100)  # real ages are 14-56; -500 and 100+ are errors

df = df.withColumn("Occupation", F.when(F.col("Occupation").rlike(r"^_+$"), "Unknown").otherwise(F.col("Occupation")))
df = df.withColumn("SSN", F.when(F.col("SSN").rlike(r"^\d{3}-\d{2}-\d{4}$"), F.col("SSN")))  # "#F%$D@*&8" -> null
df = df.withColumn("snapshot_date", F.col("snapshot_date").cast(DateType()))

dataframes_silver["df_silver_attributes"] = df

# ---------- financials ----------
df = dataframes_silver["df_silver_financials"]

numeric_cols = {
    "Annual_Income": FloatType(),
    "Monthly_Inhand_Salary": FloatType(),
    "Num_Bank_Accounts": IntegerType(),
    "Num_Credit_Card": IntegerType(),
    "Interest_Rate": IntegerType(),
    "Num_of_Loan": IntegerType(),
    "Delay_from_due_date": IntegerType(),
    "Num_of_Delayed_Payment": IntegerType(),
    "Changed_Credit_Limit": FloatType(),
    "Num_Credit_Inquiries": IntegerType(),
    "Outstanding_Debt": FloatType(),
    "Credit_Utilization_Ratio": FloatType(),
    "Total_EMI_per_month": FloatType(),
    "Amount_invested_monthly": FloatType(),
    "Monthly_Balance": FloatType(),
}
for col_name, dtype in numeric_cols.items():
    df = df.withColumn(col_name, clean_numeric(col_name, dtype))

# valid ranges, chosen from the gap between real values and junk values
valid_ranges = {
    "Num_Bank_Accounts": (0, 20),       # real values 0-11, junk up to 1756
    "Num_Credit_Card": (0, 20),         # real values 0-11, junk up to 1499
    "Interest_Rate": (0, 50),           # real values 1-34, junk up to 5789
    "Num_of_Loan": (0, 20),             # real values 0-9, -100 and up to 1495 are junk
    "Num_of_Delayed_Payment": (0, 30),  # real values 0-28, negatives and up to 4293 are junk
    "Num_Credit_Inquiries": (0, 20),    # real values 0-17, junk up to 2554
    "Annual_Income": (0, 200000),       # nothing between 180k and 300k, values above are junk
}
for col_name, (low, high) in valid_ranges.items():
    df = keep_in_range(df, col_name, low, high)

df = replace_placeholder(df, "Credit_Mix", "_")
df = replace_placeholder(df, "Payment_of_Min_Amount", "NM")
df = replace_placeholder(df, "Payment_Behaviour", "!@9#%8")
df = df.fillna({"Type_of_Loan": "Not Specified"})

# "16 Years and 2 Months" -> 194 months
df = df.withColumn(
    "Credit_History_Age_months",
    F.regexp_extract("Credit_History_Age", r"(\d+) Years", 1).cast(IntegerType()) * 12
    + F.regexp_extract("Credit_History_Age", r"(\d+) Months", 1).cast(IntegerType())
).drop("Credit_History_Age")

df = df.withColumn("snapshot_date", F.col("snapshot_date").cast(DateType()))

dataframes_silver["df_silver_financials"] = df

# ---------- clickstream ----------
df = dataframes_silver["df_silver_clickstream"]
for col_name in [f"fe_{i}" for i in range(1, 21)]:
    df = df.withColumn(col_name, F.col(col_name).cast(IntegerType()))
df = df.withColumn("snapshot_date", F.col("snapshot_date").cast(DateType()))
dataframes_silver["df_silver_clickstream"] = df

# ---------- lms ----------
df = dataframes_silver["df_silver_lms"]
for col_name in ["tenure", "installment_num"]:
    df = df.withColumn(col_name, F.col(col_name).cast(IntegerType()))
for col_name in ["loan_amt", "due_amt", "paid_amt", "overdue_amt", "balance"]:
    df = df.withColumn(col_name, F.col(col_name).cast(FloatType()))
for col_name in ["loan_start_date", "snapshot_date"]:
    df = df.withColumn(col_name, F.col(col_name).cast(DateType()))
dataframes_silver["df_silver_lms"] = df

# ---------- drop duplicates (no blanket dropna - nulls are kept and handled per column) ----------
for name, df in dataframes_silver.items():
    rows_before = df.count()
    df = df.dropDuplicates()
    dataframes_silver[name] = df
    print(f"{name}: {rows_before} -> {df.count()} rows")


# %%
# validate again after cleaning: schema, isnull, describe

for name, df in dataframes_silver.items():
    print(f"===== {name} =====")

    row_count = df.count()

    print(f"--- {name}.printSchema() ---")
    df.printSchema()

    print(f"--- {name} isnull count per column ---")
    null_counts = df.select([
        F.count(F.when(F.col(c).isNull(), c)).alias(c) for c in df.columns
    ])
    null_counts.show()

    print(f"--- {name} duplicate rows ---")
    duplicate_count = row_count - df.dropDuplicates().count()
    print(duplicate_count)

    print(f"--- {name}.describe() ---")
    df.describe().show()

    print()


# %%
# save silver tables to datamart/silver/

silver_directory = "../datamart/silver/"

if not os.path.exists(silver_directory):
    os.makedirs(silver_directory)

silver_filenames = {
    "df_silver_clickstream": "silver_feature_clickstream.parquet",
    "df_silver_attributes": "silver_features_attributes.parquet",
    "df_silver_financials": "silver_features_financials.parquet",
    "df_silver_lms": "silver_lms_loan_daily.parquet",
}

for name, df in dataframes_silver.items():
    silver_filepath = os.path.join(silver_directory, silver_filenames[name])
    df.toPandas().to_parquet(silver_filepath, index=False)
    print("saved to:", silver_filepath)


# %%
# Section 4 - Create gold 
# Create features and labels

from pyspark.sql.types import IntegerType, StringType, DateType

silver_directory = "../datamart/silver/"

df_gold_clickstream = spark.read.parquet(silver_directory + "silver_feature_clickstream.parquet")
df_gold_attributes = spark.read.parquet(silver_directory + "silver_features_attributes.parquet")
df_gold_financials = spark.read.parquet(silver_directory + "silver_features_financials.parquet")
df_gold_lms = spark.read.parquet(silver_directory + "silver_lms_loan_daily.parquet")


# %%
# Feature engineering for gold_lms

dpd_threshold = 30
mob_checkpoint = 6

# add mob (month on book)
df_gold_lms = df_gold_lms.withColumn("mob", F.col("installment_num").cast(IntegerType()))

# add dpd (days past due)
df_gold_lms = df_gold_lms.withColumn(
    "installments_missed",
    F.ceil(F.col("overdue_amt") / F.col("due_amt")).cast(IntegerType())
).fillna(0)

df_gold_lms = df_gold_lms.withColumn(
    "first_missed_date",
    F.when(F.col("installments_missed") > 0,
           F.add_months(F.col("snapshot_date"), -1 * F.col("installments_missed"))
    ).cast(DateType())
)

df_gold_lms = df_gold_lms.withColumn(
    "dpd",
    F.when(F.col("overdue_amt") > 0.0, F.datediff(F.col("snapshot_date"), F.col("first_missed_date")))
     .otherwise(0)
     .cast(IntegerType())
)

# keep only the mob checkpoint snapshot per loan
df_gold_lms = df_gold_lms.filter(F.col("mob") == mob_checkpoint)

# add label
df_gold_lms = df_gold_lms.withColumn(
    "label", F.when(F.col("dpd") >= dpd_threshold, 1).otherwise(0).cast(IntegerType())
)
df_gold_lms = df_gold_lms.withColumn(
    "label_def", F.lit(f"{dpd_threshold}dpd_{mob_checkpoint}mob").cast(StringType())
)

dataframes_gold = {
    "df_gold_clickstream": df_gold_clickstream,
    "df_gold_attributes": df_gold_attributes,
    "df_gold_financials": df_gold_financials,
    "df_gold_lms": df_gold_lms,
}


# %%



# %%
# Feature engineering for gold_financials
# New features
# Debt-to-income ratio = Outstanding_Debt / Annual_Income
# Monthly saving ratio = Monthly_Balance / Monthly_Inhand_Salary
# Total credit facilities = Number of Bank accounts + cards + loans


# %%
# save gold tables to datamart/gold/

gold_directory = "../datamart/gold/"

if not os.path.exists(gold_directory):
    os.makedirs(gold_directory)

gold_filenames = {
    "df_gold_clickstream": "gold_feature_clickstream.parquet",
    "df_gold_attributes": "gold_features_attributes.parquet",
    "df_gold_financials": "gold_features_financials.parquet",
    "df_gold_lms": "gold_lms_loan_daily.parquet",
}

for name, df in dataframes_gold.items():
    gold_filepath = os.path.join(gold_directory, gold_filenames[name])
    df.toPandas().to_parquet(gold_filepath, index=False)
    print("saved to:", gold_filepath)


# %%
# Section 5 - Model dataset and data split
# one row per loan: label (at mob 6) + features known at the loan start date (no future data)

gold_directory = "../datamart/gold/"

df_labels = spark.read.parquet(gold_directory + "gold_lms_loan_daily.parquet") \
    .select("loan_id", "Customer_ID", "loan_start_date", "label", "label_def")

# feature tables are joined where snapshot_date == loan_start_date
# Name and SSN are personal identifiers, not features
df_attributes = spark.read.parquet(gold_directory + "gold_features_attributes.parquet") \
    .drop("Name", "SSN") \
    .withColumnRenamed("snapshot_date", "loan_start_date")
df_financials = spark.read.parquet(gold_directory + "gold_features_financials.parquet") \
    .withColumnRenamed("snapshot_date", "loan_start_date")
df_clickstream = spark.read.parquet(gold_directory + "gold_feature_clickstream.parquet") \
    .withColumnRenamed("snapshot_date", "loan_start_date")

df_model = df_labels \
    .join(df_attributes, on=["Customer_ID", "loan_start_date"], how="left") \
    .join(df_financials, on=["Customer_ID", "loan_start_date"], how="left") \
    .join(df_clickstream, on=["Customer_ID", "loan_start_date"], how="left")

print("model dataset:", df_model.count(), "rows,", len(df_model.columns), "columns")


# %%
# time-based split by loan start date
# each set comes after the previous one, no overlap, OOT is the final holdout

split_periods = {
    "train": ("2023-01-01", "2024-04-01"),  # 16 months
    "valid": ("2024-05-01", "2024-07-01"),  # 3 months
    "test":  ("2024-08-01", "2024-10-01"),  # 3 months
    "oot":   ("2024-11-01", "2025-01-01"),  # 3 months
}

split_col = None
for split_name, (start_date, end_date) in split_periods.items():
    in_period = F.col("loan_start_date").between(F.to_date(F.lit(start_date)), F.to_date(F.lit(end_date)))
    split_col = F.when(in_period, split_name) if split_col is None else split_col.when(in_period, split_name)

df_model = df_model.withColumn("split", split_col)

# every loan must fall into exactly one split
assert df_model.filter(F.col("split").isNull()).count() == 0

# summary: period, duration, size, default rate and clickstream coverage per split
df_model.groupBy("split").agg(
    F.min("loan_start_date").alias("start"),
    F.max("loan_start_date").alias("end"),
    F.countDistinct("loan_start_date").alias("months"),
    F.count("*").alias("loans"),
    F.round(F.count("*") / df_model.count() * 100, 1).alias("share_%"),
    F.round(F.mean("label") * 100, 1).alias("default_rate_%"),
    F.round(F.mean(F.col("fe_1").isNotNull().cast(IntegerType())) * 100, 1).alias("clickstream_%"),
).orderBy("start").show()


# %%
# save each split to datamart/gold/model_data/

model_data_directory = os.path.join(gold_directory, "model_data/")

if not os.path.exists(model_data_directory):
    os.makedirs(model_data_directory)

for split_name in split_periods:
    df_split = df_model.filter(F.col("split") == split_name)
    split_filepath = os.path.join(model_data_directory, f"{split_name}.parquet")
    df_split.toPandas().to_parquet(split_filepath, index=False)
    print("saved to:", split_filepath, "-", df_split.count(), "rows")
