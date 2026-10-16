import os
from datetime import datetime
from dateutil.relativedelta import relativedelta
from pyspark.sql import DataFrame
from pyspark.sql.functions import (
    col, when, lit, coalesce, log1p, regexp_extract, size
)
from pyspark.sql.types import IntegerType, StringType
from pyspark.ml.feature import StringIndexer, OneHotEncoder
from pyspark.ml import Pipeline

def build_label_store(spark, dpd_cutoff=30, mob_cutoff=6) -> DataFrame:
    print("📌 Building label store…")
    loans = spark.read.parquet("datamart/silver/loans_clean")

    label_df = (
        loans
        .filter(col("mob") == mob_cutoff)
        .withColumn("label",
            when(col("dpd") >= dpd_cutoff, 1).otherwise(0).cast(IntegerType()))
        .withColumn("label_def", lit(f"{dpd_cutoff}dpd_{mob_cutoff}mob").cast(StringType()))
        .select("Customer_ID", "loan_id", "label", "snapshot_date")
    )

    os.makedirs("datamart/gold/label_store", exist_ok=True)
    label_df.write.mode("overwrite").parquet("datamart/gold/label_store")
    print("✅ Saved label store to datamart/gold/label_store")

    return label_df

# Helpers

def cap_by_quantile(df: DataFrame, colname: str, lower_q=0.01, upper_q=0.95) -> DataFrame:
    q_low, q_high = df.approxQuantile(colname, [lower_q, upper_q], 0.01)
    return df.withColumn(
        colname,
        when(col(colname) < q_low, q_low)
        .when(col(colname) > q_high, q_high)
        .otherwise(col(colname))
    )

def multi_snapshot_join(
    feat_df: DataFrame,
    loans_df: DataFrame,
    dpd_cutoff: int = 30,
    mob_cutoff: int = 6
) -> DataFrame:
    """
    Join a feature source to loans in a multi-snapshot fashion:
    for each feat_date, find loans at feat_date + mob months,
    attach label, and union all slices.
    """
    feat_dates = [r.snapshot_date for r in feat_df.select("snapshot_date").distinct().collect()]
    loan_dates = [r.snapshot_date for r in loans_df.select("snapshot_date").distinct().collect()]
    rows = []

    for fd in sorted(feat_dates):
        ld = fd + relativedelta(months=mob_cutoff)
        if ld not in loan_dates:
            continue

        f_snap = feat_df\
            .filter(col("snapshot_date") == fd)\
            .withColumnRenamed("snapshot_date", "feature_snapshot_date")

        l_snap = loans_df\
            .filter((col("snapshot_date") == ld) & (col("mob") == mob_cutoff))\
            .withColumn("label", when(col("dpd") >= dpd_cutoff, 1).otherwise(0).cast(IntegerType()))\
            .select("Customer_ID", "label")\
            .withColumn("label_snapshot_date", lit(ld))

        rows.append(f_snap.join(l_snap, on="Customer_ID", how="inner"))

    if not rows:
        raise RuntimeError("No valid multi-snapshot slices found")

    out = rows[0]
    for other in rows[1:]:
        out = out.unionByName(other)

    return out.cache()

# Source-specific processing

def process_financials(df: DataFrame) -> DataFrame:
    # 1) Impute & filter
    df = df\
      .withColumn("Credit_Mix",        coalesce(col("Credit_Mix"), lit("Unknown")))\
      .withColumn("Payment_Behaviour", coalesce(col("Payment_Behaviour"), lit("Unknown")))\
      .withColumn("Changed_Credit_Limit", coalesce(col("Changed_Credit_Limit"), lit(0.0)))\
      .filter(col("Monthly_Balance").isNotNull())

    # median debt
    med = df.approxQuantile("Outstanding_Debt", [0.5], 0.01)[0]
    df = df.withColumn("Outstanding_Debt", coalesce(col("Outstanding_Debt"), lit(med)))

    # 2) Parse & derive
    df = df\
      .withColumn("history_years",  regexp_extract("Credit_History_Age", r"(\d+)\s+Years", 1).cast("int"))\
      .withColumn("history_months", regexp_extract("Credit_History_Age", r"(\d+)\s+Months",1).cast("int"))\
      .withColumn("credit_history_months",
                  col("history_years")*12 + coalesce(col("history_months"), lit(0)))\
      .drop("Credit_History_Age","history_years","history_months")\
      .withColumn("num_loan_types", size(col("Loan_Types_Array")))

    # 3) Cap outliers
    for c in ["Num_Bank_Accounts","Num_Credit_Card","Interest_Rate","Num_of_Delayed_Payment","Num_Credit_Inquiries"]:
        df = cap_by_quantile(df, c)

    # 4) Log & ratios
    df = df\
      .withColumn("log_Annual_Income",   log1p(col("Annual_Income")))\
      .withColumn("debt_to_income",      col("Outstanding_Debt")/(col("Annual_Income")+lit(1.0)))\
      .withColumn("emi_to_salary",       col("Total_EMI_per_month")/(col("Monthly_Inhand_Salary")+lit(1.0)))\
      .withColumn("investment_rate",     col("Amount_invested_monthly")/(col("Monthly_Inhand_Salary")+lit(1.0)))\
      .withColumn("has_credit_limit_change", (col("Changed_Credit_Limit")!=0).cast(IntegerType()))\
      .withColumn("balance_to_debt",     (col("Monthly_Balance")+lit(1.0))/(col("Outstanding_Debt")+lit(1.0)))\
      .withColumn("inq_per_loan",        col("Num_Credit_Inquiries")/(col("Num_of_Loan")+lit(1.0)))

    # 5) One-hot encode
    cats = ["Credit_Mix","Payment_Behaviour","Payment_of_Min_Amount"]
    idxs = [StringIndexer(inputCol=c, outputCol=c+"_idx", handleInvalid="keep") for c in cats]
    ohs  = [OneHotEncoder(inputCol=c+"_idx", outputCol=c+"_vec") for c in cats]
    pipe = Pipeline(stages=idxs+ohs)
    df = pipe.fit(df).transform(df)
    return df.drop(*cats, *[c+"_idx" for c in cats])

def process_attributes(df: DataFrame) -> DataFrame:
    df = df\
      .withColumn("Age",        when((col("Age")>=18)&(col("Age")<=75), col("Age")).otherwise(None))\
      .withColumn("Occupation", coalesce(col("Occupation"), lit("Unknown")))

    # bin age
    df = df.withColumn("Age_group",
         when(col("Age")<25, "<25")
        .when(col("Age")<40, "25–39")
        .when(col("Age")<60, "40–59")
        .otherwise("60+"))

    cats = ["Occupation","Age_group"]
    idxs = [StringIndexer(inputCol=c, outputCol=c+"_idx", handleInvalid="keep") for c in cats]
    ohs  = [OneHotEncoder(inputCol=c+"_idx", outputCol=c+"_vec") for c in cats]
    pipe = Pipeline(stages=idxs+ohs)
    df = pipe.fit(df).transform(df)
    return df.drop(*cats, *[c+"_idx" for c in cats])

def process_clickstream(df: DataFrame) -> DataFrame:
    # just cap extremes
    for i in range(1,21):
        df = cap_by_quantile(df, f"fe_{i}")
    return df

def build_feature_store(spark, dpd_cutoff=30, mob_cutoff=6) -> DataFrame:
    print("📌 Building unified gold-ML feature store…")

    # read silver tables
    fin = spark.read.parquet("datamart/silver/financials_clean")
    att = spark.read.parquet("datamart/silver/attributes_clean")
    clk = spark.read.parquet("datamart/silver/clickstream_clean")
    lms = spark.read.parquet("datamart/silver/loans_clean")

    # multi-snapshot + per-source processing
    fin_all = process_financials(multi_snapshot_join(fin, lms, dpd_cutoff, mob_cutoff))
    att_all = process_attributes(multi_snapshot_join(att, lms, dpd_cutoff, mob_cutoff))
    clk_all = process_clickstream(multi_snapshot_join(clk, lms, dpd_cutoff, mob_cutoff))

    # join all feature slices
    feat = (
        fin_all
        .join(att_all,
              on=["Customer_ID", "feature_snapshot_date", "label_snapshot_date", "label"],
              how="inner")
        .join(clk_all,
              on=["Customer_ID", "feature_snapshot_date", "label_snapshot_date", "label"],
              how="inner")
        .cache()
    )

    # drop cols
    cols_to_drop = [
        # target & its window date
        "label", "label_snapshot_date",
        # PII
        "Name", "SSN",
        # raw vs engineered
        "Annual_Income",
        "Monthly_Inhand_Salary",
        "Outstanding_Debt",
        "Total_EMI_per_month",
        "Amount_invested_monthly",
        "Age",
        "Num_of_Loan",
        # intermediate / original strings
        "Type_of_Loan",
        "Loan_Types_Array",
        "Credit_History_Age",
        "Credit_Mix",
        "Payment_Behaviour",
        "Payment_of_Min_Amount"
    ]
    # only drop those actually present
    drop_list = [c for c in cols_to_drop if c in feat.columns]
    final_feat = feat.drop(*drop_list)

    # persist the ML-ready feature store
    os.makedirs("datamart/gold/feature_store", exist_ok=True)
    final_feat.write.mode("overwrite") \
        .parquet("datamart/gold/feature_store")

    return final_feat
