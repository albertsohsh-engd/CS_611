# utils/silver_processing.py
from pyspark.sql.functions import (
    col, regexp_replace, when, split, expr, ceil, datediff
)
from pyspark.sql.types import (
    DoubleType, IntegerType, StringType, DateType
)


# CLEAN FINANCIALS

def clean_financials_table(spark):
    print("Cleaning financials…")
    df = spark.read.parquet("datamart/bronze/financials")
    df = df.replace(["_", "NA", "na", "N/A"], None)

    # strip non-numeric noise
    df = df.withColumn("Annual_Income", regexp_replace("Annual_Income", "[^0-9.]", "")) \
           .withColumn("Num_of_Loan",   regexp_replace("Num_of_Loan",   "[^0-9]" , "")) \
           .withColumn("Num_of_Delayed_Payment",
                       regexp_replace("Num_of_Delayed_Payment", "[^0-9]", "")) \
           .withColumn("Amount_invested_monthly",
                       regexp_replace("Amount_invested_monthly", "[^0-9.]", ""))

    # enforce schema
    cast_map = {
        "Annual_Income": DoubleType(), "Monthly_Balance": DoubleType(),
        "Outstanding_Debt": DoubleType(), "Amount_invested_monthly": DoubleType(),
        "Changed_Credit_Limit": DoubleType(), "Total_EMI_per_month": DoubleType(),
        "Credit_Utilization_Ratio": DoubleType(),
        "Num_Bank_Accounts": IntegerType(), "Num_Credit_Card": IntegerType(),
        "Interest_Rate": IntegerType(), "Num_of_Loan": IntegerType(),
        "Num_of_Delayed_Payment": IntegerType(), "Num_Credit_Inquiries": IntegerType()
    }
    for c, t in cast_map.items():
        df = df.withColumn(c, col(c).cast(t))

    # keep only valid payment behaviours
    valid_pb = [
        "High_spent_Large_value_payments", "High_spent_Medium_value_payments",
        "High_spent_Small_value_payments", "Low_spent_Large_value_payments",
        "Low_spent_Medium_value_payments", "Low_spent_Small_value_payments"
    ]
    df = df.withColumn("Payment_Behaviour",
                       when(col("Payment_Behaviour").isin(valid_pb),
                            col("Payment_Behaviour")).otherwise(None))

    # split Type_of_Loan
    df = (df.withColumn("Loan_Types_Array",
                        expr("transform(split(Type_of_Loan, ', |, and '), x -> lower(trim(x)))")))

    df = df.filter(col("Customer_ID").isNotNull() & col("snapshot_date").isNotNull())
    df.write.mode("overwrite").parquet("datamart/silver/financials_clean")
    print("✅  Saved Silver → financials_clean")
    return df


# 2.  CLEAN ATTRIBUTES

def clean_attributes_table(spark):
    print("Cleaning attributes…")
    df = spark.read.parquet("datamart/bronze/attributes") \
           .replace(["_", "NA", "na", "N/A", "_______"], None)

    # Age
    df = df.withColumn("Age", regexp_replace("Age", "[^0-9]", "").cast(IntegerType()))
    df = df.withColumn("Age", when((col("Age") > 0) & (col("Age") < 100), col("Age")))

    # SSN regex filter
    df = df.withColumn("SSN",
                       when(col("SSN").rlike("^\\d{3}-\\d{2}-\\d{4}$"), col("SSN")))

    df = df.filter(col("Customer_ID").isNotNull() & col("snapshot_date").isNotNull())
    df.write.mode("overwrite").parquet("datamart/silver/attributes_clean")
    print("✅  Saved Silver → attributes_clean")
    return df


# 3.  CLEAN CLICKSTREAM

def clean_clickstream_table(spark):
    print("Cleaning clickstream…")
    df = spark.read.parquet("datamart/bronze/clickstream") \
           .replace(["_", "NA", "na", "N/A"], None)

    for i in range(1, 21):
        df = df.withColumn(f"fe_{i}", col(f"fe_{i}").cast(IntegerType()))

    df = df.filter(col("Customer_ID").isNotNull() & col("snapshot_date").isNotNull())
    df.write.partitionBy("snapshot_date").mode("overwrite") \
      .parquet("datamart/silver/clickstream_clean")
    print("✅  Saved Silver → clickstream_clean")
    return df


# 4.  CLEAN LOANS

def clean_loans_table(spark):
    print("Cleaning loans…")
    df = spark.read.parquet("datamart/bronze/loans")

    schema_map = {
        "loan_id": StringType(), "Customer_ID": StringType(),
        "loan_start_date": DateType(), "tenure": IntegerType(),
        "installment_num": IntegerType(), "loan_amt": DoubleType(),
        "due_amt": DoubleType(), "paid_amt": DoubleType(),
        "overdue_amt": DoubleType(), "balance": DoubleType(),
        "snapshot_date": DateType()
    }
    for c, t in schema_map.items():
        df = df.withColumn(c, col(c).cast(t))

    df = (df.withColumn("mob", col("installment_num"))
             .withColumn("installments_missed",
                         ceil(col("overdue_amt") / col("due_amt")))
             .withColumn("first_missed_date",
                         when(col("installments_missed") > 0,
                              expr("add_months(snapshot_date, -installments_missed)")))
             .withColumn("dpd",
                         when(col("overdue_amt") > 0,
                              datediff(col("snapshot_date"), col("first_missed_date")))
                         .otherwise(0).cast(IntegerType()))
             .filter(col("Customer_ID").isNotNull() & col("snapshot_date").isNotNull()))

    df.write.partitionBy("snapshot_date").mode("overwrite") \
      .parquet("datamart/silver/loans_clean")
    print("✅  Saved Silver → loans_clean")
    return df
