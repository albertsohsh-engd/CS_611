"""Silver layer — Steps 05 (schema enforcement), 06 (validation), 07 (cleaning), 09 (reusable transforms),
08 (post-cleaning EDA vs the Bronze baseline).

Converted from notebooks/dataprocessing_silver.ipynb (the notebook is the development version; this module is the
callable version used by the pipeline).

Usage from a notebook / pipeline:
    from util import dataprocessing_bronze as bronze, dataprocessing_silver as silver
    spark = bronze.get_spark()
    silver.run(spark)                       # 05 -> 06 -> 07 -> 09 -> 06 -> write -> decisions -> 08
    silver.run(spark, eda=False)            # skip step 08
    silver.read_silver(spark, "features_financials")

Command line:
    python util/dataprocessing_silver.py [--skip-eda] [--no-charts] [--tests]
"""
import os
import sys
import shutil
import uuid
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yaml
from pyspark.sql import Window, functions as F, types as T

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from util import dataprocessing_bronze as bronze      # Bronze reader, contract and profiling rules (reused)


# =============================================================================================
# Configuration
# =============================================================================================
PROJECT_DIR    = bronze.PROJECT_DIR
SILVER_DIR     = os.path.join(PROJECT_DIR, "datamart", "silver")
REPORT_DIR     = os.path.join(SILVER_DIR, "_reports")
QUARANTINE_DIR = os.path.join(SILVER_DIR, "_quarantine")
LOG_PATH       = os.path.join(SILVER_DIR, "_silver_log.csv")
SILVER_EDA_DIR = os.path.join(SILVER_DIR, "_eda")
SILVER_CHART_DIR = os.path.join(SILVER_EDA_DIR, "charts")
BRONZE_EDA_DIR = os.path.join(PROJECT_DIR, "datamart", "bronze", "_eda")
BRONZE_AUDIT_DIR = os.path.join(PROJECT_DIR, "datamart", "bronze", "_audit")

with open(os.path.join(PROJECT_DIR, "config", "silver_schema.yaml"), encoding="utf-8") as _f:
    SCHEMA = yaml.safe_load(_f)
TABLES = list(SCHEMA["tables"])

# Plausible ranges: Bronze profiling rules, with Silver overrides
SILVER_RANGE_RULES = {
    **bronze.RANGE_RULES,
    ("features_financials", "Delay_from_due_date"):      (-30, 365),   # negative = paid early (accepted)
    ("features_financials", "Credit_Utilization_Ratio"): (0, 100),
}

# Allowed values of categorical columns (anything else is a validation failure)
VALID_CATEGORIES = {
    ("features_attributes", "Occupation"): [
        "Accountant", "Architect", "Developer", "Doctor", "Engineer", "Entrepreneur", "Journalist", "Lawyer",
        "Manager", "Mechanic", "Media_Manager", "Musician", "Scientist", "Teacher", "Writer"],
    ("features_financials", "Credit_Mix"): ["Good", "Standard", "Bad"],
    ("features_financials", "Payment_of_Min_Amount"): ["Yes", "No"],
    ("features_financials", "Payment_Behaviour"): [
        "High_spent_Large_value_payments", "High_spent_Medium_value_payments", "High_spent_Small_value_payments",
        "Low_spent_Large_value_payments", "Low_spent_Medium_value_payments", "Low_spent_Small_value_payments"],
}

# Formats that must hold in Silver
SILVER_FORMAT_RULES = {
    ("lms_loan_daily", "loan_id"): r"CUS_0x[0-9a-f]+_\d{4}_\d{2}_\d{2}",
    **{(t, "Customer_ID"): r"CUS_0x[0-9a-f]+" for t in TABLES},
    ("features_financials", "Credit_History_Age"): r"\d+ Years? and \d+ Months?",
}

# Columns that get a <column>_invalid flag when cleaning sets an unusable value to null
CLEANING_FLAG_COLUMNS = {
    "features_attributes": ["Age", "Occupation"],
    "features_financials": ["Annual_Income", "Num_Bank_Accounts", "Num_Credit_Card", "Interest_Rate", "Num_of_Loan",
                            "Num_of_Delayed_Payment", "Changed_Credit_Limit", "Num_Credit_Inquiries", "Credit_Mix",
                            "Payment_of_Min_Amount", "Total_EMI_per_month", "Amount_invested_monthly",
                            "Payment_Behaviour", "Monthly_Balance"],
}

# Cross-field rules applied after the column-level cleaning: (table, target column, condition, audit action).
# Where the condition is true the target is impossible given another field -> null + <target>_invalid flag.
CROSS_FIELD_NULL_RULES = [
    ("features_financials", "Total_EMI_per_month",
     lambda: F.col("Total_EMI_per_month") > F.col("Monthly_Inhand_Salary"),   # instalment > take-home pay
     "exceeds_monthly_salary_to_null"),
]

KNOWN_LOAN_TYPES = ["Auto Loan", "Credit-Builder Loan", "Debt Consolidation Loan", "Home Equity Loan",
                    "Mortgage Loan", "Not Specified", "Payday Loan", "Personal Loan", "Student Loan"]

DATA_START = "2023-01-01"           # earliest valid snapshot_date
NULL_PCT_DEFAULT  = (5, 30)         # null % thresholds (warn, fail) for feature columns
NULL_PCT_CRITICAL = (0, 0)

# Step 08 tolerances
TOL_08 = {"row_loss_pct": 0.0, "null_pct_change_pp": 1.0, "median_change_pct": 5.0, "psi": 0.10,
          "category_shift_pp": 2.0}
PSI_BINS = 10

NULLING_ACTIONS = ["invalid_token_to_null", "out_of_range_to_null", "placeholder_to_null",
                   "exceeds_monthly_salary_to_null"]


def get_spark(app_name="silver"):
    return bronze.get_spark(app_name)


# =============================================================================================
# Helpers: strict casting, column cleaning rules, step-09 parsers
# =============================================================================================
NUM_RX  = f"^{bronze.NUMERIC_RE}$"
DATE_RX = f"^{bronze.DATE_RE}$"
TRAILING_UNDERSCORE_RX = r"^([+-]?(\d+(\.\d*)?|\.\d+))_+$"
PLACEHOLDER_RX = f"^(?:{bronze.PLACEHOLDER_RE})$"
CHA_RX = r"^\s*(\d+)\s+Years?\s+and\s+(\d+)\s+Months?\s*$"


def _col(c):
    return F.col(c) if isinstance(c, str) else c


def safe_cast(c, typ):
    c = _col(c)
    if typ == "string":
        return c
    if typ == "date":
        return F.when(c.rlike(DATE_RX), F.to_date(c, "yyyy-MM-dd"))
    num = F.when(c.rlike(NUM_RX), c.cast("double"))
    if typ == "double":
        return num
    if typ == "int":
        return F.when((num == F.floor(num)) & num.between(-2147483648, 2147483647), num.cast("long")).cast("int")
    raise ValueError(f"unknown type {typ}")


def clean_numeric_col(c, typ, rng=None):
    raw = F.trim(F.col(c))
    stripped = F.regexp_replace(raw, TRAILING_UNDERSCORE_RX, "$1")      # '32_' -> '32'
    cast = safe_cast(stripped, typ)
    token = raw.isNotNull() & cast.isNull()                              # '_', '__10000__', ...
    if rng:
        lo, hi = rng
        out_of_range = cast.isNotNull() & ((cast < lo) | (cast > hi))
    else:
        out_of_range = F.lit(False)
    return dict(value=F.when(~out_of_range, cast),
                underscore=raw.isNotNull() & (stripped != raw) & cast.isNotNull(),
                token=token, out_of_range=out_of_range, invalid=token | out_of_range)


def clean_category_col(c, codes=()):
    raw = F.trim(F.col(c))
    placeholder = raw.isNotNull() & (raw.rlike(PLACEHOLDER_RX) | raw.isin(list(codes)))
    return dict(value=F.when(~placeholder, raw), placeholder=placeholder, invalid=placeholder)


def credit_history_months(c):
    c = F.trim(_col(c))
    years, months = F.regexp_extract(c, CHA_RX, 1), F.regexp_extract(c, CHA_RX, 2)
    ok = c.rlike(CHA_RX) & (months.cast("int") < 12)
    return F.when(ok, years.cast("int") * 12 + months.cast("int"))


def loan_type_list(c):
    c = F.trim(_col(c))
    norm = F.regexp_replace(c, r",\s*and\s+", ", ")          # 'A, B, and C' -> 'A, B, C'
    norm = F.regexp_replace(norm, r"\s+and\s+", ", ")        # 'A and B'     -> 'A, B'
    parts = F.transform(F.split(norm, ","), lambda x: F.trim(x))
    return F.when(c.isNull() | (c == ""), F.array().cast("array<string>")).otherwise(
        F.filter(parts, lambda x: x != ""))


# =============================================================================================
# Step 05 — Schema enforcement
# =============================================================================================
def check_drift(table, df, schema=SCHEMA):
    spec = schema["tables"][table]["columns"]
    findings = []
    missing_req = [c for c, s in spec.items() if s.get("required") and c not in df.columns]
    if missing_req:
        raise ValueError(f"{table}: required columns missing from Bronze: {missing_req}")
    for c, s in spec.items():
        if not s.get("required") and c not in df.columns:
            df = df.withColumn(c, F.lit(None).cast("string"))
            findings.append(dict(table=table, column=c, finding="optional column missing", status="WARN",
                                 action="added as null"))
    for c in df.columns:
        if c not in spec and not c.startswith("_"):
            findings.append(dict(table=table, column=c, finding="unexpected column", status="WARN",
                                 action="dropped from Silver"))
    return df, findings


def enforce_schema(table, df, schema=SCHEMA):
    spec = schema["tables"][table]["columns"]
    df, drift = check_drift(table, df, schema)
    for c, s in spec.items():
        if s.get("pii"):
            drift.append(dict(table=table, column=c, finding="PII column", status="INFO", action="not loaded"))
    keep = [c for c, s in spec.items() if not s.get("pii")]
    df = df.select(*keep, *schema["lineage_columns"])

    reasons, stats = [], []
    for c in keep:
        s = spec[c]
        bad_cast = F.col(c).isNotNull() & safe_cast(c, s["type"]).isNull()
        stats += [F.sum(F.col(c).isNull().cast("int")).alias(f"{c}|nulls"),
                  F.sum(bad_cast.cast("int")).alias(f"{c}|cast_failures")]
        if s.get("critical"):
            reasons.append(F.when(F.col(c).isNull(), F.lit(f"{c}: null"))
                            .when(bad_cast, F.concat(F.lit(f"{c}: not a valid {s['type']} '"), F.col(c), F.lit("'"))))
    df = df.withColumn("_quarantine_reason", F.concat_ws("; ", *reasons) if reasons else F.lit(""))

    agg = df.agg(F.count("*").alias("rows"), *stats).first().asDict()
    violations = []
    for c in keep:
        s, nulls, fails = spec[c], agg[f"{c}|nulls"], agg[f"{c}|cast_failures"]
        if nulls or fails:
            examples = ""
            if fails:
                ex = (df.filter(F.col(c).isNotNull() & safe_cast(c, s["type"]).isNull())
                        .groupBy(c).count().orderBy(F.desc("count")).limit(5).collect())
                examples = "; ".join(f"{r[c]} ({r['count']:,})" for r in ex)
            violations.append(dict(
                table=table, column=c, type=s["type"], critical=bool(s.get("critical")),
                nulls=nulls, cast_failures=fails, pct=(nulls + fails) / agg["rows"] * 100, examples=examples,
                action="quarantine row" if s.get("critical") else "pass raw value to cleaning (step 07)"))

    passed = df.filter(F.col("_quarantine_reason") == "").drop("_quarantine_reason")
    quarantined = df.filter(F.col("_quarantine_reason") != "")
    return passed, quarantined, violations, drift


def cast_all(table, df):
    # Type-cast view of the passed rows, no cleaning: shows what validation finds in the raw values
    spec = SCHEMA["tables"][table]["columns"]
    return df.select(*[safe_cast(c, s["type"]).alias(c) for c, s in spec.items() if not s.get("pii")],
                     *SCHEMA["lineage_columns"])


# =============================================================================================
# Step 06 — Validation rules
# =============================================================================================
# A rule is either a row-level expression ("expr": evaluated for all such rules of a table in ONE Spark pass)
# or a function that needs a window / join ("check": returns the frame with a boolean _viol column).
def viol(expr):
    return dict(expr=expr)


def unique_check(keys):
    return dict(check=lambda df, frames: df.withColumn("_viol", F.count("*").over(Window.partitionBy(*keys)) > 1))


def ledger_check(target):
    def fn(df, frames):
        w = Window.partitionBy("loan_id").orderBy("installment_num").rowsBetween(Window.unboundedPreceding, 0)
        d = df.withColumn("_cum_paid", F.sum("paid_amt").over(w)).withColumn("_cum_due", F.sum("due_amt").over(w))
        expected = (F.col("loan_amt") - F.col("_cum_paid")) if target == "balance" else (F.col("_cum_due") - F.col("_cum_paid"))
        return d.withColumn("_viol", F.abs(F.col(target) - expected) > 0.01)
    return dict(check=fn)


def fk_check(col, parent, parent_col):
    def fn(df, frames):
        keys = frames[parent].select(F.col(parent_col).alias("_fk_key")).distinct()
        return (df.join(keys, F.col(col) == F.col("_fk_key"), "left")
                  .withColumn("_viol", F.col("_fk_key").isNull() & F.col(col).isNotNull()))
    return dict(check=fn)


def build_rules(frames):
    as_of = frames["lms_loan_daily"].agg(F.max("snapshot_date")).first()[0]
    rules = []

    def add(table, rule, kind, columns, test, warn, fail):
        rules.append(dict(rule_id=f"V{len(rules) + 1:03d}", table=table, rule=rule, kind=kind,
                          columns=columns, warn_pct=warn, fail_pct=fail, **test))

    for t, tspec in SCHEMA["tables"].items():
        keys = tspec["keys"]
        for k in keys:
            add(t, f"{k} not null", "key", [k], viol(F.col(k).isNull()), 0, 0)
        add(t, f"one row per {' + '.join(keys)}", "key", keys, unique_check(keys), 0, 0)
        for c, s in tspec["columns"].items():
            if s.get("pii") or c in keys:
                continue
            add(t, f"{c} null %", "null_pct", [c], viol(F.col(c).isNull()),
                *(NULL_PCT_CRITICAL if s.get("critical") else NULL_PCT_DEFAULT))
        add(t, f"snapshot_date in [{DATA_START}, {as_of}]", "date", ["snapshot_date"],
            viol((F.col("snapshot_date") < F.lit(DATA_START).cast("date")) | (F.col("snapshot_date") > F.lit(as_of))), 0, 0)

    for (t, c), (lo, hi) in SILVER_RANGE_RULES.items():
        add(t, f"{c} in [{lo:,}, {hi:,}]", "range", [c],
            viol(F.col(c).isNotNull() & ((F.col(c) < lo) | (F.col(c) > hi))), 0, 1)
    for (t, c), values in VALID_CATEGORIES.items():
        add(t, f"{c} in allowed set", "category", [c], viol(F.col(c).isNotNull() & ~F.col(c).isin(values)), 0, 1)
    for (t, c), rx in SILVER_FORMAT_RULES.items():
        add(t, f"{c} format", "format", [c], viol(F.col(c).isNotNull() & ~F.col(c).rlike(f"^{rx}$")), 0, 0)

    lms = "lms_loan_daily"
    add(lms, "snapshot_date >= loan_start_date", "date", ["snapshot_date", "loan_start_date"],
        viol(F.col("snapshot_date") < F.col("loan_start_date")), 0, 0)
    add(lms, "installment_num = months since loan_start_date", "cross_field", ["installment_num"],
        viol(F.months_between("snapshot_date", "loan_start_date") != F.col("installment_num")), 0, 0)
    add(lms, "balance = loan_amt - cumulative paid_amt", "cross_field", ["balance"], ledger_check("balance"), 0, 0)
    add(lms, "overdue_amt = cumulative due_amt - cumulative paid_amt", "cross_field", ["overdue_amt"],
        ledger_check("overdue_amt"), 0, 0)

    fin = "features_financials"
    add(fin, "Num_of_Loan = number of entries in Type_of_Loan", "cross_field", ["Num_of_Loan", "Type_of_Loan"],
        viol(F.col("Num_of_Loan").isNotNull() & (F.col("Num_of_Loan") != F.size(loan_type_list("Type_of_Loan")))),
        0.1, 1)
    add(fin, "Total_EMI_per_month <= Monthly_Inhand_Salary", "cross_field", ["Total_EMI_per_month", "Monthly_Inhand_Salary"],
        viol(F.col("Total_EMI_per_month") > F.col("Monthly_Inhand_Salary")), 0, 1)
    ratio = F.col("Monthly_Inhand_Salary") * 12 / F.col("Annual_Income")
    add(fin, "12 x Monthly_Inhand_Salary / Annual_Income in [0.05, 1.5]", "cross_field",
        ["Monthly_Inhand_Salary", "Annual_Income"], viol(ratio.isNotNull() & ~ratio.between(0.05, 1.5)), 0.5, 2)

    for t in ["feature_clickstream", "features_attributes", "features_financials"]:
        add(t, "Customer_ID exists in lms_loan_daily", "foreign_key", ["Customer_ID"],
            fk_check("Customer_ID", lms, "Customer_ID"), 0, 0)
    for t in ["features_attributes", "features_financials"]:
        add(lms, f"loan customer has a row in {t}", "foreign_key", ["Customer_ID"],
            fk_check("Customer_ID", t, "Customer_ID"), 0, 0)
    add(lms, "loan customer has clickstream history (BRZ-001, accepted)", "coverage", ["Customer_ID"],
        fk_check("Customer_ID", "feature_clickstream", "Customer_ID"), 0, 50)
    return rules


def violating_rows(r, frames):
    df = frames[r["table"]]
    if "expr" in r:
        return df.filter(F.coalesce(r["expr"], F.lit(False)))
    return r["check"](df, frames).filter(F.col("_viol"))


def run_rules(frames, rules, stage):
    totals = {t: df.count() for t, df in frames.items()}
    counts = {}
    for t, df in frames.items():                      # all row-level rules of a table in one pass
        simple = [r for r in rules if r["table"] == t and "expr" in r]
        if simple:
            counts.update(df.agg(*[F.sum(F.coalesce(r["expr"], F.lit(False)).cast("int")).alias(r["rule_id"])
                                   for r in simple]).first().asDict())
    for r in rules:                                   # window / join rules one by one
        if "check" in r:
            counts[r["rule_id"]] = violating_rows(r, frames).count()

    report, samples = [], []
    for r in rules:
        n = counts[r["rule_id"]] or 0
        pct = n / totals[r["table"]] * 100
        status = "FAIL" if pct > r["fail_pct"] else "WARN" if pct > r["warn_pct"] else "PASS"
        report.append(dict(stage=stage, rule_id=r["rule_id"], table=r["table"], kind=r["kind"], rule=r["rule"],
                           violations=n, pct=round(pct, 3), warn_pct=r["warn_pct"], fail_pct=r["fail_pct"],
                           status=status))
        if n:
            keys = SCHEMA["tables"][r["table"]]["keys"]
            show = list(dict.fromkeys(keys + r["columns"]))
            for row in violating_rows(r, frames).select(*show).limit(5).collect():
                samples.append(dict(stage=stage, rule_id=r["rule_id"], table=r["table"], rule=r["rule"],
                                    sample=str({k: str(row[k]) for k in show})))
    return pd.DataFrame(report), pd.DataFrame(samples)


# =============================================================================================
# Step 07 — Cleaning
# =============================================================================================
def clean_table(table, df):
    tspec = SCHEMA["tables"][table]
    flags_for = set(CLEANING_FLAG_COLUMNS.get(table, []))
    select, audit = [], {}
    for c, s in tspec["columns"].items():
        if s.get("pii"):
            continue
        invalid = None
        if s["type"] in ("int", "double"):
            r = clean_numeric_col(c, s["type"], SILVER_RANGE_RULES.get((table, c)))
            value, invalid = r["value"], r["invalid"]
            audit[(c, "trailing_underscore_corrected")] = r["underscore"]
            audit[(c, "invalid_token_to_null")] = r["token"]
            audit[(c, "out_of_range_to_null")] = r["out_of_range"]
        elif s["type"] == "date":
            value = safe_cast(F.trim(F.col(c)), "date")
            audit[(c, "invalid_date_to_null")] = F.col(c).isNotNull() & value.isNull()
        elif (table, c) in VALID_CATEGORIES:
            r = clean_category_col(c, bronze.KNOWN_CODES.get((table, c), []))
            value, invalid = r["value"], r["placeholder"]
            audit[(c, "placeholder_to_null")] = r["placeholder"]
        else:
            value = F.trim(F.col(c))
            audit[(c, "whitespace_trimmed")] = F.col(c) != value
        select.append(value.alias(c))
        if c in flags_for:
            select.append(F.coalesce(invalid, F.lit(False)).alias(f"{c}_invalid"))

    counts = df.agg(*[F.sum(e.cast("int")).alias(f"{c}|{a}") for (c, a), e in audit.items()]).first().asDict()
    rows_in = df.count()
    audit_rows = [dict(table=table, column=k.split("|")[0], action=k.split("|")[1], rows_affected=v or 0,
                       pct=(v or 0) / rows_in * 100) for k, v in counts.items() if v]

    cleaned = df.select(*select, F.col("_source_file"), F.col("_source_row_num").cast("long"),
                        F.col("_batch_id").alias("_bronze_batch_id"))

    # Cross-field rules on the cleaned values (flag first, then null the target)
    for t, target, condition, action in CROSS_FIELD_NULL_RULES:
        if t != table:
            continue
        cond = F.coalesce(condition(), F.lit(False))
        n = cleaned.filter(cond).count()
        if n:
            audit_rows.append(dict(table=table, column=target, action=action, rows_affected=n, pct=n / rows_in * 100))
        cleaned = (cleaned.withColumn(f"{target}_invalid", F.col(f"{target}_invalid") | cond)
                          .withColumn(target, F.when(~cond, F.col(target))))

    # Deduplicate by key: keep the latest Bronze batch, then the last row in the source file
    w = Window.partitionBy(*tspec["keys"]).orderBy(F.desc("_bronze_batch_id"), F.desc("_source_row_num"))
    deduped = cleaned.withColumn("_rank", F.row_number().over(w)).filter("_rank = 1").drop("_rank")
    return deduped, audit_rows, rows_in


# =============================================================================================
# Step 09 — Reusable transforms
# =============================================================================================
def add_transforms(table, df):
    if table != "features_financials":
        return df
    return (df.withColumn("credit_history_age_months", credit_history_months("Credit_History_Age"))
              .withColumn("type_of_loan_list", loan_type_list("Type_of_Loan"))
              .withColumn("num_loan_types", F.size("type_of_loan_list"))
              .withColumn("type_of_loan_unknown",
                          F.exists("type_of_loan_list", lambda x: ~x.isin(KNOWN_LOAN_TYPES))))


# =============================================================================================
# Write / read Silver
# =============================================================================================
def write_parquet_atomic(df, out_path, partition_col="snapshot_date"):
    tmp = out_path + "__tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    df.repartition(partition_col).write.mode("overwrite").partitionBy(partition_col).parquet(tmp)
    shutil.rmtree(out_path, ignore_errors=True)
    os.replace(tmp, out_path)


def write_silver(frames, quarantine_frames):
    batch_id = f"silver_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{uuid.uuid4().hex[:6]}"
    silver_ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    log = []
    for t, df in frames.items():
        out = df.withColumn("_silver_batch_id", F.lit(batch_id)).withColumn("_silver_ts", F.lit(silver_ts))
        write_parquet_atomic(out, os.path.join(SILVER_DIR, t))
        q_path = os.path.join(QUARANTINE_DIR, t)
        n_q = quarantine_frames[t].count()
        shutil.rmtree(q_path, ignore_errors=True)
        if n_q:
            quarantine_frames[t].write.mode("overwrite").parquet(q_path)
        log.append(dict(silver_batch_id=batch_id, table=t, silver_rows=df.count(), quarantined=n_q,
                        columns=len(out.columns), silver_ts=silver_ts))
    log = pd.DataFrame(log)
    old = pd.read_csv(LOG_PATH) if os.path.exists(LOG_PATH) else pd.DataFrame()
    pd.concat([old, log], ignore_index=True).to_csv(LOG_PATH, index=False)
    return log


def read_silver(spark, table):
    return spark.read.parquet(os.path.join(SILVER_DIR, table))


# =============================================================================================
# Build: 05 -> 06 (pre) -> 07 -> 09 -> 06 (post) -> write
# =============================================================================================
def build_silver(spark):
    for d in (SILVER_DIR, REPORT_DIR, QUARANTINE_DIR):
        os.makedirs(d, exist_ok=True)

    # Step 05
    bronze_frames, passed, quarantined = {}, {}, {}
    all_violations, all_drift = [], []
    for t in TABLES:
        bronze_frames[t] = bronze.read_bronze(spark, t).cache()
        p, q, v, d = enforce_schema(t, bronze_frames[t])
        passed[t], quarantined[t] = p.cache(), q.cache()
        all_violations += v
        all_drift += d
    schema_violations, schema_drift = pd.DataFrame(all_violations), pd.DataFrame(all_drift)

    # Step 06 before cleaning
    typed_pre = {t: cast_all(t, passed[t]).cache() for t in TABLES}
    validation_pre, samples_pre = run_rules(typed_pre, build_rules(typed_pre), "pre_clean")

    # Step 07 + 09
    cleaned, cleaning_audit, table_log = {}, [], []
    for t in TABLES:
        df, audit_rows, rows_in = clean_table(t, passed[t])
        cleaned[t] = add_transforms(t, df).cache()
        cleaning_audit += audit_rows
        table_log.append(dict(table=t, bronze_rows=bronze_frames[t].count(), quarantined=quarantined[t].count(),
                              rows_into_cleaning=rows_in, duplicates_removed=rows_in - cleaned[t].count(),
                              silver_rows=cleaned[t].count()))
    cleaning_audit, before_after = pd.DataFrame(cleaning_audit), pd.DataFrame(table_log)

    # Step 06 after cleaning (acceptance: no FAIL)
    validation_post, samples_post = run_rules(cleaned, build_rules(cleaned), "post_clean")
    if (validation_post["status"] == "FAIL").any():
        raise AssertionError(f"Silver validation failed after cleaning:\n"
                             f"{validation_post[validation_post['status'] == 'FAIL']}")

    write_log = write_silver(cleaned, quarantined)

    validation_report = pd.concat([validation_pre, validation_post], ignore_index=True)
    schema_violations.to_csv(os.path.join(REPORT_DIR, "schema_violations.csv"), index=False)
    schema_drift.to_csv(os.path.join(REPORT_DIR, "schema_drift.csv"), index=False)
    validation_report.to_csv(os.path.join(REPORT_DIR, "validation_report.csv"), index=False)
    pd.concat([samples_pre, samples_post], ignore_index=True).to_csv(
        os.path.join(REPORT_DIR, "validation_failed_samples.csv"), index=False)
    cleaning_audit.to_csv(os.path.join(REPORT_DIR, "cleaning_audit.csv"), index=False)
    before_after.to_csv(os.path.join(REPORT_DIR, "cleaning_row_counts.csv"), index=False)

    return dict(silver=cleaned, quarantined=quarantined, schema_violations=schema_violations,
                schema_drift=schema_drift, validation_report=validation_report, cleaning_audit=cleaning_audit,
                row_counts=before_after, write_log=write_log)


# =============================================================================================
# Tests (unit tests of the rules + Silver integrity)
# =============================================================================================
def run_unit_tests(spark):
    # Step 05: edge cases
    test_schema = {
        "lineage_columns": ["_source_file", "_source_row_num", "_batch_id"],
        "tables": {"t": {"keys": ["id"], "columns": {
            "id":     {"type": "string", "required": True, "critical": True},
            "d":      {"type": "date",   "required": True, "critical": True},
            "n":      {"type": "int",    "required": True},
            "x":      {"type": "double", "required": True},
            "opt":    {"type": "string", "required": False},
            "secret": {"type": "string", "required": False, "pii": True}}}}}
    rows = [("A",  "2024-01-01", "12",    "1.5", "s1", "f", "1", "b"),
            ("B",  "2024-13-01", "12.0",  "2",   "s2", "f", "2", "b"),
            (None, "2024-01-01", "12_",   "1e3", "s3", "f", "3", "b"),
            ("C",  "2024-02-01", "12.5",  "NaN", "s4", "f", "4", "b"),
            ("D",  "2024-02-01", "1,000", "_",   "s5", "f", "5", "b")]
    cols = ["id", "d", "n", "x", "secret", "_source_file", "_source_row_num", "_batch_id"]
    test_df = spark.createDataFrame(rows, cols).withColumn("unexpected", F.lit("z"))
    p, q, v, drift = enforce_schema("t", test_df, test_schema)
    assert sorted(r["id"] for r in p.collect()) == ["A", "C", "D"], "wrong rows passed"
    assert sorted(r["_quarantine_reason"] for r in q.collect()) == ["d: not a valid date '2024-13-01'", "id: null"]
    assert "secret" not in p.columns and "unexpected" not in p.columns and "opt" in p.columns
    viol_ = {r["column"]: r for r in v}
    assert viol_["n"]["cast_failures"] == 3 and viol_["x"]["cast_failures"] == 2, viol_
    try:
        enforce_schema("t", test_df.drop("n"), test_schema)
        raise AssertionError("missing required column was not rejected")
    except ValueError as e:
        assert "required columns missing" in str(e)

    # Step 07: cleaning rules
    def apply(fn_result, values):
        df = spark.createDataFrame([(v,) for v in values], T.StructType([T.StructField("v", T.StringType())]))
        return [tuple(r) for r in df.select(fn_result["value"].alias("value"),
                                            fn_result["invalid"].alias("invalid")).collect()]
    assert apply(clean_numeric_col("v", "int", (14, 100)), ["32", "32_", "-500", "8678", "_", "__10000__", " 45 ", None]) \
        == [(32, False), (32, False), (None, True), (None, True), (None, True), (None, True), (45, False), (None, False)]
    assert apply(clean_numeric_col("v", "double"), ["52312.68_", "__-333333333333333333333333333__", "_", "1.5"]) \
        == [(52312.68, False), (None, True), (None, True), (1.5, False)]
    assert apply(clean_category_col("v", ["NM"]), ["Lawyer", "_______", "_", "!@9#%8", "NM", "Yes", None]) \
        == [("Lawyer", False), (None, True), (None, True), (None, True), (None, True), ("Yes", False), (None, False)]

    # Step 09: parsers
    cases = spark.createDataFrame(
        [("16 Years and 3 Months",), ("0 Years and 0 Months",), ("1 Year and 1 Month",), ("10 Years and 11 Months",),
         ("5 Years and 12 Months",), ("abc",), ("",), (None,)], "s string")
    assert [r[0] for r in cases.select(credit_history_months("s")).collect()] == [195, 0, 13, 131, None, None, None, None]
    loans = spark.createDataFrame(
        [(None,), ("",), ("Auto Loan",), ("Credit-Builder Loan, and Home Equity Loan",),
         ("Not Specified, Home Equity Loan, Credit-Builder Loan, and Payday Loan",), ("Auto Loan and Payday Loan",),
         ("Mortgage Loan, Mortgage Loan",), ("Space Loan",)], "Type_of_Loan string")
    parsed = add_transforms("features_financials",
                            loans.withColumn("Credit_History_Age", F.lit(None).cast("string"))).collect()
    assert [r["num_loan_types"] for r in parsed] == [0, 0, 1, 2, 4, 2, 2, 1]
    assert [r["type_of_loan_unknown"] for r in parsed] == [False] * 7 + [True]
    print("Unit tests passed: schema enforcement, cleaning rules, step-09 parsers")


SPARK_TYPES = {"string": T.StringType(), "int": T.IntegerType(), "double": T.DoubleType(), "date": T.DateType()}


def content_checksum(df):
    cols = sorted(c for c in df.columns if not c.startswith("_silver"))
    return df.select(F.sum(F.hash(*[F.col(c).cast("string") for c in cols]).cast("long"))).first()[0], df.count()


def run_integrity_tests(spark, row_counts):
    for t in TABLES:
        s = read_silver(spark, t)
        tspec = SCHEMA["tables"][t]
        row = row_counts.set_index("table").loc[t]
        assert row["bronze_rows"] == s.count() + row["quarantined"] + row["duplicates_removed"], f"{t}: rows do not reconcile"
        keys = tspec["keys"]
        assert s.groupBy(*keys).count().filter("count > 1").count() == 0, f"{t}: duplicate keys"
        assert s.filter(F.greatest(*[F.col(k).isNull().cast("int") for k in keys]) == 1).count() == 0, f"{t}: null keys"
        types = {f.name: f.dataType for f in s.schema.fields}
        for c, spec in tspec["columns"].items():
            if spec.get("pii"):
                assert c not in types, f"{t}: PII column {c} present"
            else:
                assert types[c] == SPARK_TYPES[spec["type"]], f"{t}.{c}: {types[c]} != {spec['type']}"
        for c in CLEANING_FLAG_COLUMNS.get(t, []):
            assert s.filter(F.col(f"{c}_invalid") & F.col(c).isNotNull()).count() == 0, f"{t}.{c}: flag set on a value"
        assert s.filter(F.col("_source_file").isNull() | F.col("_source_row_num").isNull()).count() == 0, f"{t}: lineage"
    print("Integrity tests passed: rows reconcile, keys unique, types match the schema contract, no PII, "
          "flags consistent, lineage complete")


# =============================================================================================
# Decision register — Bronze findings -> Silver decisions
# =============================================================================================
FIN, ATT, CLK, LMS = "features_financials", "features_attributes", "feature_clickstream", "lms_loan_daily"
NULL_FLAG = "set to null + {c}_invalid flag; imputed later on training data only (step 18)"

DECISIONS = {
    # (table, column, category): (decision, action, verified_by)
    (CLK, "Customer_ID", "coverage"): ("defer", "No Silver change. Gold as-of join leaves clickstream features null "
                                       "for uncovered loans (contract feature_availability).", "rule: clickstream coverage (WARN, accepted)"),
    (ATT, "Age", "invalid_token"): ("correct", "trailing underscore stripped, cast to int", "cleaning audit + rule: Age in range"),
    (ATT, "Age", "out_of_range"): ("transform", NULL_FLAG.format(c="Age"), "rule: Age in [14, 100] PASS after cleaning"),
    (ATT, "Name", "privacy"): ("exclude", "not loaded into Silver", "integrity test 3 (no PII)"),
    (ATT, "SSN", "privacy"): ("exclude", "not loaded into Silver", "integrity test 3 (no PII)"),
    (ATT, "SSN", "format"): ("exclude", "column not loaded (PII), so the invalid format is moot", "integrity test 3"),
    (ATT, "Occupation", "placeholder"): ("transform", NULL_FLAG.format(c="Occupation"), "rule: Occupation in allowed set"),
    (FIN, "Annual_Income", "invalid_token"): ("correct", "trailing underscore stripped, cast to double", "cleaning audit"),
    (FIN, "Annual_Income", "out_of_range"): ("transform", NULL_FLAG.format(c="Annual_Income") +
                                             " (12 x monthly salary is only 0.03-10% of these incomes)", "rules: range + salary/income"),
    (FIN, "Credit_Mix", "placeholder"): ("transform", NULL_FLAG.format(c="Credit_Mix"), "rule: Credit_Mix in allowed set"),
    (FIN, "Payment_Behaviour", "placeholder"): ("transform", NULL_FLAG.format(c="Payment_Behaviour"), "rule: allowed set"),
    (FIN, "Payment_of_Min_Amount", "placeholder"): ("transform", NULL_FLAG.format(c="Payment_of_Min_Amount") +
                                                    "; NM = not recorded (NM rows mix Yes/No profiles)", "rule: allowed set"),
    (FIN, "Type_of_Loan", "missing"): ("transform", "empty = no loans (1,363 of 1,426 empty rows have Num_of_Loan = 0): "
                                       "type_of_loan_list = [], num_loan_types = 0; raw text kept null", "step 09 tests"),
    (FIN, "Type_of_Loan", "structure"): ("transform", "parsed to type_of_loan_list + num_loan_types (step 09)", "step 09 tests"),
    (FIN, "Credit_History_Age", "structure"): ("transform", "parsed to credit_history_age_months (step 09)", "step 09 tests"),
    (FIN, "Amount_invested_monthly", "invalid_token"): ("transform", NULL_FLAG.format(c="Amount_invested_monthly") +
                                                        " ('__10000__' sentinel)", "cleaning audit"),
    (FIN, "Changed_Credit_Limit", "invalid_token"): ("transform", NULL_FLAG.format(c="Changed_Credit_Limit"), "cleaning audit"),
    (FIN, "Monthly_Balance", "invalid_token"): ("transform", NULL_FLAG.format(c="Monthly_Balance"), "cleaning audit"),
    (FIN, "Outstanding_Debt", "invalid_token"): ("correct", "trailing underscore stripped, cast to double", "cleaning audit"),
    (FIN, "Num_of_Delayed_Payment", "invalid_token"): ("correct", "trailing underscore stripped, cast to int", "cleaning audit"),
    (FIN, "Num_of_Loan", "invalid_token"): ("correct", "trailing underscore stripped, cast to int", "cleaning audit"),
    (FIN, "Num_of_Loan", "out_of_range"): ("transform", NULL_FLAG.format(c="Num_of_Loan") +
                                           "; num_loan_types (step 09) carries the same information (99.97% agreement)",
                                           "rules: range + Num_of_Loan = entries in Type_of_Loan"),
    (FIN, "Delay_from_due_date", "out_of_range"): ("accept", "negative values (-5..-1) = paid before the due date; "
                                                   "Silver range [-30, 365]", "rule: Delay_from_due_date in range"),
    **{(FIN, c, "out_of_range"): ("transform", NULL_FLAG.format(c=c), f"rule: {c} in range PASS after cleaning")
       for c in ["Interest_Rate", "Num_Bank_Accounts", "Num_Credit_Card", "Num_Credit_Inquiries",
                 "Num_of_Delayed_Payment", "Total_EMI_per_month"]},
    **{(t, "snapshot_date", "coverage"): ("defer", "No Silver change. Contract as-of rule (snapshot_date <= "
                                          "loan_start_date) means later loans use the latest available snapshot.",
                                          "contract test: feature_coverage_at_scoring_time") for t in [CLK, ATT, FIN]},
    (LMS, "loan_amt", "constant"): ("accept", "kept in Silver for ledger checks; not a predictor (contract)", "rule: ledger balance"),
    (LMS, "tenure", "constant"): ("accept", "kept in Silver; not a predictor (contract)", "—"),
    (LMS, "snapshot_date", "sampling_gap"): ("accept", "portfolio ramp-up (2023) and run-off after the last loans "
                                             "start in 2025-01; explained by the origination pattern", "Bronze chart 01/06"),
}


def build_decision_register():
    issues = pd.read_csv(os.path.join(BRONZE_EDA_DIR, "raw_eda_issues.csv"))
    register = issues[["issue_id", "table", "column", "category", "severity", "finding"]].copy()
    mapped = register.apply(lambda r: DECISIONS.get((r["table"], r["column"], r["category"])), axis=1)
    register["decision"] = mapped.map(lambda d: d[0] if d else "UNMAPPED")
    register["action"] = mapped.map(lambda d: d[1] if d else "")
    register["verified_by"] = mapped.map(lambda d: d[2] if d else "")

    audit = pd.read_csv(os.path.join(BRONZE_AUDIT_DIR, "audit_report.csv"))
    for _, a in audit[audit["status"] != "PASS"].iterrows():
        register.loc[len(register)] = dict(
            issue_id=f"AUD-{a['table']}-{a['check']}", table=a["table"], column="snapshot_date", category=a["check"],
            severity="medium", finding=a["observed"], decision="defer",
            action="No Silver change; historical extract. Contract as-of join uses the latest snapshot on or before loan start.",
            verified_by="contract test: feature_coverage_at_scoring_time")

    # Finding raised by step 08 (not in the Bronze register)
    register.loc[len(register)] = dict(
        issue_id="S08-001", table=FIN, column="Total_EMI_per_month", category="cross_field", severity="medium",
        finding="Instalment larger than monthly take-home salary (residual cluster 1,689-49,942 seen in the step-08 "
                "Bronze-vs-Silver overlay; EMI is <= 15% of salary for 99% of other customers)",
        decision="transform", action=NULL_FLAG.format(c="Total_EMI_per_month") + " (cross-field rule)",
        verified_by="rule: Total_EMI_per_month <= Monthly_Inhand_Salary PASS after cleaning")

    register.to_csv(os.path.join(REPORT_DIR, "silver_decisions.csv"), index=False)
    unmapped = register[register["decision"] == "UNMAPPED"]
    if not unmapped.empty:
        raise AssertionError(f"Findings without a decision: {unmapped[['issue_id', 'table', 'column', 'category']].to_dict('records')}")
    return register


# =============================================================================================
# Step 08 — Post-cleaning EDA: Silver vs Bronze baseline
# =============================================================================================
def to_profile_strings(s):
    # Silver typed values -> strings, so the Bronze profiling function measures them identically
    return pd.Series([None if v is None or (isinstance(v, float) and np.isnan(v)) else str(v) for v in s],
                     index=s.index, dtype=object)


def numeric_values(s):
    v = s.dropna().astype(str)
    return pd.to_numeric(v[v.str.fullmatch(bronze.NUMERIC_RE)])


def psi(expected, actual, bins=PSI_BINS):
    if len(expected) == 0 or len(actual) == 0:
        return np.nan
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:                                     # constant column
        return 0.0 if (actual == edges[0]).all() else np.inf
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.clip(np.histogram(expected, edges)[0] / len(expected), 1e-6, None)
    a = np.clip(np.histogram(actual, edges)[0] / len(actual), 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


def category_shift(bronze_raw, silver_pd, t, c):
    # Largest change in the share of a VALID category (shares among valid values only)
    valid = VALID_CATEGORIES[(t, c)]
    b = bronze_raw[t][c][bronze_raw[t][c].isin(valid)].value_counts(normalize=True)
    s = silver_pd[t][c].dropna().value_counts(normalize=True)
    return float((s.reindex(valid).fillna(0) - b.reindex(valid).fillna(0)).abs().max() * 100)


def run_step08(spark, cleaning_audit, register, charts=True):
    os.makedirs(SILVER_CHART_DIR, exist_ok=True)
    bronze_raw = bronze.load_raw(spark)
    silver_pd = {t: read_silver(spark, t).drop("_silver_batch_id", "_silver_ts").toPandas() for t in TABLES}
    bronze_baseline = pd.read_csv(os.path.join(BRONZE_EDA_DIR, "bronze_profile_baseline.csv"))
    bronze_parts = pd.read_csv(os.path.join(BRONZE_EDA_DIR, "bronze_partition_counts.csv"), dtype={"snapshot_date": str})

    # Silver profile with the Bronze profiling function
    common = {t: [c for c in bronze_raw[t].columns if c in silver_pd[t].columns] for t in TABLES}
    silver_profile = pd.DataFrame([bronze.profile_column(t, c, to_profile_strings(silver_pd[t][c]))
                                   for t in TABLES for c in common[t]])
    silver_profile["table_duplicate_rows"] = silver_profile["table"].map(
        {t: int(silver_pd[t][common[t]].astype(str).duplicated().sum()) for t in TABLES})
    silver_profile.to_csv(os.path.join(SILVER_EDA_DIR, "silver_profile_baseline.csv"), index=False)

    # Row loss and duplicates
    silver_parts = pd.concat([
        silver_pd[t].assign(snapshot_date=silver_pd[t]["snapshot_date"].astype(str)).groupby("snapshot_date").size()
        .rename("silver_rows").reset_index().assign(table=t) for t in TABLES])
    parts = bronze_parts.rename(columns={"rows": "bronze_rows"}).merge(silver_parts, on=["table", "snapshot_date"],
                                                                       how="outer").fillna(0)
    parts["lost"] = parts["bronze_rows"] - parts["silver_rows"]
    dups_b = bronze_baseline.groupby("table")["table_duplicate_rows"].first()
    dups_s = silver_profile.groupby("table")["table_duplicate_rows"].first()
    row_loss = (parts.groupby("table")
                .agg(bronze_rows=("bronze_rows", "sum"), silver_rows=("silver_rows", "sum"),
                     partitions=("snapshot_date", "count"),
                     partitions_with_loss=("lost", lambda s: int((s != 0).sum())))
                .reindex(TABLES).reset_index())
    row_loss["row_loss_pct"] = (row_loss["bronze_rows"] - row_loss["silver_rows"]) / row_loss["bronze_rows"] * 100
    row_loss["duplicates_bronze"] = row_loss["table"].map(dups_b)
    row_loss["duplicates_silver"] = row_loss["table"].map(dups_s)
    row_loss["status"] = np.where((row_loss["row_loss_pct"] <= TOL_08["row_loss_pct"])
                                  & (row_loss["duplicates_silver"] <= row_loss["duplicates_bronze"]), "PASS", "INVESTIGATE")

    # Column comparison
    stats = ["inferred_type", "null_pct", "distinct", "min", "p01", "p50", "p99", "max", "mean", "std", "skew", "iqr_outliers"]
    cmp = bronze_baseline[["table", "column", *stats]].merge(silver_profile[["table", "column", *stats]],
                                                             on=["table", "column"], suffixes=("_bronze", "_silver"))
    num = lambda s: pd.to_numeric(s, errors="coerce")
    cmp["null_pct_change_pp"] = cmp["null_pct_silver"] - cmp["null_pct_bronze"]
    cmp["median_change_pct"] = np.where(num(cmp["p50_bronze"]).abs() > 0,
                                        (num(cmp["p50_silver"]) - num(cmp["p50_bronze"])) / num(cmp["p50_bronze"]).abs() * 100,
                                        np.nan)
    cmp["psi"] = [psi(numeric_values(bronze_raw[t][c]), numeric_values(silver_pd[t][c])) if it == "numeric" else np.nan
                  for t, c, it in zip(cmp["table"], cmp["column"], cmp["inferred_type_bronze"])]
    cmp["category_shift_pp"] = [category_shift(bronze_raw, silver_pd, t, c) if (t, c) in VALID_CATEGORIES else np.nan
                                for t, c in zip(cmp["table"], cmp["column"])]

    audit_text = (cleaning_audit.assign(txt=lambda d: d["action"] + " " + d["pct"].round(2).astype(str) + "%")
                  .groupby(["table", "column"])["txt"].agg("; ".join))
    decision_text = (register.groupby(["table", "column"])
                     .apply(lambda g: ", ".join(f"{i} {d}" for i, d in zip(g["issue_id"], g["decision"])),
                            include_groups=False))

    def assess(r):
        reasons = []
        if abs(r["null_pct_change_pp"]) > TOL_08["null_pct_change_pp"]:
            reasons.append(f"null % {r['null_pct_bronze']:.2f} -> {r['null_pct_silver']:.2f}")
        if pd.notna(r["median_change_pct"]) and abs(r["median_change_pct"]) > TOL_08["median_change_pct"]:
            reasons.append(f"median {r['median_change_pct']:+.1f}%")
        if pd.notna(r["psi"]) and r["psi"] > TOL_08["psi"]:
            reasons.append(f"PSI {r['psi']:.3f}")
        if pd.notna(r["category_shift_pp"]) and r["category_shift_pp"] > TOL_08["category_shift_pp"]:
            reasons.append(f"category share shift {r['category_shift_pp']:.2f} pp")
        key = (r["table"], r["column"])
        explanation = "; ".join(x for x in [audit_text.get(key, ""), decision_text.get(key, "")] if x)
        status = "PASS" if not reasons else ("EXPLAINED" if key in audit_text.index else "INVESTIGATE")
        return pd.Series(dict(material_change="; ".join(reasons), status=status,
                              explanation=explanation if reasons else ""))

    cmp = pd.concat([cmp, cmp.apply(assess, axis=1)], axis=1)
    cmp.to_csv(os.path.join(SILVER_EDA_DIR, "bronze_vs_silver_comparison.csv"), index=False)

    # Flag consistency: flag rate == share nulled by cleaning
    nulled = cleaning_audit[cleaning_audit["action"].isin(NULLING_ACTIONS)].groupby(["table", "column"])["pct"].sum()
    flags = pd.DataFrame([dict(table=t, column=c, flag=f"{c}_invalid",
                               flag_rate_pct=silver_pd[t][f"{c}_invalid"].mean() * 100,
                               nulled_by_cleaning_pct=nulled.get((t, c), 0.0))
                          for t, cols in CLEANING_FLAG_COLUMNS.items() for c in cols])
    assert np.allclose(flags["flag_rate_pct"], flags["nulled_by_cleaning_pct"]), flags

    # Action log
    log_rows = []
    for _, r in cmp[cmp["status"] != "PASS"].iterrows():
        for reason in r["material_change"].split("; "):
            log_rows.append(dict(table=r["table"], column=r["column"], change=reason, status=r["status"],
                                 explanation=r["explanation"],
                                 action="accept — expected result of the recorded cleaning decision"
                                        if r["status"] == "EXPLAINED" else "investigate before Gold"))
    for _, r in row_loss[row_loss["status"] != "PASS"].iterrows():
        log_rows.append(dict(table=r["table"], column="*", change=f"row loss {r['row_loss_pct']:.2f}%",
                             status="INVESTIGATE", explanation="", action="investigate before Gold"))
    action_log = pd.DataFrame(log_rows, columns=["table", "column", "change", "status", "explanation", "action"])
    action_log.to_csv(os.path.join(SILVER_EDA_DIR, "silver_vs_bronze_action_log.csv"), index=False)

    if charts:
        make_step08_charts(bronze_raw, silver_pd, cmp)

    assert (row_loss["status"] == "PASS").all(), "unexpected row loss or new duplicates"
    assert (cmp["status"] != "INVESTIGATE").all(), cmp.loc[cmp["status"] == "INVESTIGATE", ["table", "column", "material_change"]]
    return dict(comparison=cmp, row_loss=row_loss, flags=flags, action_log=action_log, silver_profile=silver_profile)


# ---------------------------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------------------------
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]   # fixed categorical order (validated palette)


def _plt():
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "axes.titlecolor": INK, "axes.titlesize": 11,
        "axes.titleweight": "bold", "axes.titlelocation": "left", "xtick.color": INK_2, "ytick.color": INK_2,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID,
        "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
        "legend.fontsize": 8, "font.size": 9})
    return plt


def plot_cleaning_actions(cleaning_audit):
    plt = _plt()
    actions = {"trailing_underscore_corrected": "Corrected (underscore stripped)",
               "invalid_token_to_null": "Invalid token → null",
               "placeholder_to_null": "Placeholder / code → null",
               "out_of_range_to_null": "Out of range → null",
               "exceeds_monthly_salary_to_null": "Cross-field rule → null"}
    a = cleaning_audit[cleaning_audit["action"].isin(actions)]
    pv = a.pivot_table(index=["table", "column"], columns="action", values="pct", aggfunc="sum").fillna(0)
    pv = pv.reindex(columns=[k for k in actions if k in pv.columns])
    pv = pv.loc[pv.sum(axis=1).sort_values().index]
    names = [f"{t.replace('features_', '')} · {c}" for t, c in pv.index]

    fig, ax = plt.subplots(figsize=(11, 0.32 * len(pv) + 1.2))
    left = np.zeros(len(pv))
    for color, k in zip(SERIES, pv.columns):
        ax.barh(names, pv[k], left=left, color=color, label=actions[k], height=0.7, edgecolor=SURFACE, linewidth=1)
        left += pv[k].values
    for y, v in enumerate(pv.sum(axis=1)):
        ax.text(v + 0.3, y, f"{v:.1f}%" if v >= 0.1 else "<0.1%", va="center", fontsize=8, color=INK_2)
    ax.set_xlabel("% of rows")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right")
    ax.set_title("Silver cleaning actions by column — % of rows changed")
    fig.tight_layout()
    os.makedirs(os.path.join(REPORT_DIR, "charts"), exist_ok=True)
    fig.savefig(os.path.join(REPORT_DIR, "charts", "silver_cleaning_actions.png"), dpi=130, bbox_inches="tight")
    return fig


def make_step08_charts(bronze_raw, silver_pd, cmp, close=True):
    plt = _plt()
    figs = []

    # 1. NULL % dumbbell
    d = cmp[(cmp["null_pct_bronze"] > 0) | (cmp["null_pct_silver"] > 0)].copy().sort_values("null_pct_silver")
    d["name"] = d["table"].str.replace("features_", "") + " · " + d["column"]
    fig, ax = plt.subplots(figsize=(10, 0.36 * len(d) + 1.4))
    y = np.arange(len(d))
    ax.hlines(y, d["null_pct_bronze"], d["null_pct_silver"], color=GRID, linewidth=3, zorder=1)
    ax.scatter(d["null_pct_bronze"], y, color=SERIES[0], s=55, zorder=2, label="Bronze (raw)", edgecolor=SURFACE, linewidth=1.5)
    ax.scatter(d["null_pct_silver"], y, color=SERIES[1], s=55, zorder=3, label="Silver (cleaned)", edgecolor=SURFACE, linewidth=1.5)
    for yi, v in zip(y, d["null_pct_silver"]):
        ax.text(v + 0.4, yi, f"{v:.1f}%", va="center", fontsize=8, color=INK_2)
    ax.set_yticks(y, d["name"])
    ax.set_xlabel("% of rows null")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right")
    ax.set_title("NULL % — Bronze baseline vs Silver (invalid values now null)")
    fig.tight_layout()
    fig.savefig(os.path.join(SILVER_CHART_DIR, "08_1_null_pct_bronze_vs_silver.png"), dpi=130, bbox_inches="tight")
    figs.append(fig)

    # 2. Distributions
    fields = [("features_attributes", "Age"), ("features_financials", "Annual_Income"),
              ("features_financials", "Num_Bank_Accounts"), ("features_financials", "Interest_Rate"),
              ("features_financials", "Num_of_Loan"), ("features_financials", "Total_EMI_per_month")]
    to_sym = lambda v: np.sign(v) * np.log10(1 + np.abs(v))
    from_sym = lambda v: np.sign(v) * (10 ** np.abs(v) - 1)
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.6))
    for ax, (t, c) in zip(axes.flat, fields):
        b, s = numeric_values(bronze_raw[t][c]), numeric_values(silver_pd[t][c])
        lo, hi = SILVER_RANGE_RULES[(t, c)]
        ax.axvspan(lo, hi, color=SERIES[0], alpha=0.06, lw=0)
        bins = from_sym(np.linspace(to_sym(min(b.min(), 0)), to_sym(b.max()), 60))
        ax.hist(b, bins=bins, histtype="step", color=SERIES[0], linewidth=1.6, label="Bronze")
        ax.hist(s, bins=bins, histtype="step", color=SERIES[1], linewidth=1.6, label="Silver")
        ax.set_xscale("symlog", linthresh=1)
        ax.set_yscale("log")
        r = cmp.set_index(["table", "column"]).loc[(t, c)]
        ax.set_title(c, pad=16)
        ax.text(0, 1.01, f"median {float(r['p50_bronze']):,.0f} → {float(r['p50_silver']):,.0f} · "
                         f"max {float(r['max_bronze']):,.0f} → {float(r['max_silver']):,.0f} · PSI {r['psi']:.3f}",
                transform=ax.transAxes, ha="left", va="bottom", fontsize=7.5, color=INK_2)
        ax.set_xlabel("value (symmetric log scale)")
    for ax in axes[:, 0]:
        ax.set_ylabel("rows (log)")
    axes[0, 0].legend(loc="upper right")
    fig.suptitle("Distributions — Bronze vs Silver (shaded band = plausible range)", x=0.01, ha="left",
                 fontweight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(SILVER_CHART_DIR, "08_2_distributions_bronze_vs_silver.png"), dpi=130, bbox_inches="tight")
    figs.append(fig)

    # 3. Category shares
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), gridspec_kw={"height_ratios": [2.3, 1]})
    for ax, (t, c) in zip(axes.flat, list(VALID_CATEGORIES)):
        b = bronze_raw[t][c].fillna("<NULL>").value_counts(normalize=True) * 100
        s = silver_pd[t][c].fillna("<NULL>").value_counts(normalize=True) * 100
        order = VALID_CATEGORIES[(t, c)][::-1] + [v for v in b.index if v not in VALID_CATEGORIES[(t, c)]] + ["<NULL>"]
        order = list(dict.fromkeys(order))
        labels = [v if v in VALID_CATEGORIES[(t, c)] or v == "<NULL>" else f"'{v}'" for v in order]
        yy = np.arange(len(order))
        ax.barh(yy + 0.2, b.reindex(order).fillna(0), height=0.38, color=SERIES[0], label="Bronze")
        ax.barh(yy - 0.2, s.reindex(order).fillna(0), height=0.38, color=SERIES[1], label="Silver")
        ax.set_yticks(yy, labels)
        ax.set_title(f"{c}  (max valid-share shift {category_shift(bronze_raw, silver_pd, t, c):.2f} pp)")
        ax.set_xlabel("% of rows")
        ax.grid(axis="y", visible=False)
    axes[0, 0].legend(loc="lower right")
    fig.suptitle("Category shares — Bronze vs Silver", x=0.01, ha="left", fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(os.path.join(SILVER_CHART_DIR, "08_3_categories_bronze_vs_silver.png"), dpi=130, bbox_inches="tight")
    figs.append(fig)

    if close:
        for f in figs:
            plt.close(f)
    return figs


# =============================================================================================
# Entry point
# =============================================================================================
def run(spark=None, eda=True, charts=True, tests=False):
    """Run the Silver layer: 05 -> 06 -> 07 -> 09 -> 06 (acceptance) -> write -> decision register -> 08."""
    spark = spark or get_spark()
    if tests:
        run_unit_tests(spark)
    out = build_silver(spark)
    if tests:
        run_integrity_tests(spark, out["row_counts"])
    out["decisions"] = build_decision_register()
    if charts:
        import matplotlib.pyplot as plt
        plt.close(plot_cleaning_actions(out["cleaning_audit"]))
    if eda:
        out["step08"] = run_step08(spark, out["cleaning_audit"], out["decisions"], charts=charts)
    return out


if __name__ == "__main__":
    import argparse
    import matplotlib
    matplotlib.use("Agg")

    parser = argparse.ArgumentParser(description="Silver layer: schema, validation, cleaning, transforms, EDA")
    parser.add_argument("--skip-eda", action="store_true", help="skip step 08 (Bronze-vs-Silver comparison)")
    parser.add_argument("--no-charts", action="store_true", help="do not write charts")
    parser.add_argument("--tests", action="store_true", help="run unit and integrity tests")
    args = parser.parse_args()

    result = run(eda=not args.skip_eda, charts=not args.no_charts, tests=args.tests)
    v = result["validation_report"]
    print(result["row_counts"].to_string(index=False))
    print("\nValidation before cleaning:", v[v["stage"] == "pre_clean"]["status"].value_counts().to_dict())
    print("Validation after cleaning: ", v[v["stage"] == "post_clean"]["status"].value_counts().to_dict())
    print("Decisions:", result["decisions"]["decision"].value_counts().to_dict())
    if "step08" in result:
        print("Step 08:", result["step08"]["comparison"]["status"].value_counts().to_dict())
