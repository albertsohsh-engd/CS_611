"""Bronze layer — Steps 02 (ingest as-is), 03 (source audit), 04 (raw EDA profiling).

Converted from notebooks/dataprocessing_bronze.ipynb (the notebook is the development
version; this module is the callable version used by the pipeline).

Usage from a notebook / pipeline:
    from util import dataprocessing_bronze as bronze
    spark = bronze.get_spark()
    bronze.run(spark)                          # ingest + audit + EDA outputs
    bronze.run(spark, eda=False)               # ingest + audit only
    bronze.read_bronze(spark, "lms_loan_daily")

Command line:
    python util/dataprocessing_bronze.py [--force] [--skip-eda] [--html-profiles] [--tests]
"""
import os
import sys
import csv
import hashlib
import shutil
import time
import uuid
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yaml
from pyspark.sql import SparkSession, Window, functions as F, types as T


# =============================================================================================
# Configuration
# =============================================================================================
# Project folder: /app inside the Docker container, otherwise the parent of util/
PROJECT_DIR = os.environ.get("PROJECT_DIR") or (
    "/app" if os.path.isdir("/app/data") else os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_DIR    = os.path.join(PROJECT_DIR, "data")
BRONZE_DIR = os.path.join(PROJECT_DIR, "datamart", "bronze")
LOG_PATH   = os.path.join(BRONZE_DIR, "_ingestion_log.csv")
AUDIT_DIR  = os.path.join(BRONZE_DIR, "_audit")
EDA_DIR    = os.path.join(BRONZE_DIR, "_eda")
CHART_DIR  = os.path.join(EDA_DIR, "charts")

# Sources, keys and PII come from the data contract (single source of truth, step 01)
CONTRACT_PATH = os.path.join(PROJECT_DIR, "config", "data_contract.yaml")
with open(CONTRACT_PATH, encoding="utf-8") as _f:
    CONTRACT = yaml.safe_load(_f)

# bronze table name -> (source file in data/, event-time column)
SOURCES = {t: (s["file"], s["event_time"]) for t, s in CONTRACT["sources"].items()}
# Key of each Bronze table: exactly one row per key is expected
KEYS = {t: s["keys"] for t, s in CONTRACT["sources"].items()}
# Personal identifiers
PII_COLUMNS = {(t, c) for t, cols in CONTRACT["excluded_columns"]["pii"].items() for c in cols}

LOG_COLUMNS = ["batch_id", "table", "source_file", "source_sha256", "source_rows", "bronze_rows",
               "event_partitions", "min_event_date", "max_event_date", "schema_version", "schema_hash",
               "ingest_ts", "mode", "status", "duration_s", "message"]

META_COLS = ["_source_row_num", "_source_file", "_source_sha256", "_event_date",
             "_ingest_ts", "_batch_id", "_schema_version", "_schema_hash"]

# --- Step 03 tolerances ---
TOLERANCES = {
    "count_diff_pct":        0.0,   # Bronze vs source rows (total and per partition)
    "null_key_pct":          0.0,
    "dup_key_pct":           0.0,
    "freshness_warn_months": 1,
    "freshness_fail_months": 12,
}
AS_OF_DATE = None   # None -> latest snapshot_date across all Bronze tables

# --- Step 04 profiling rules ---
NUMERIC_RE = r"[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?"
DATE_RE = r"\d{4}-\d{2}-\d{2}"
NUMERIC_MIN_PCT = 50       # a column is profiled as numeric if >= 50% of non-null values parse as numbers
LOW_CARDINALITY = 50       # columns with <= 50 distinct values get a frequency table

# Expected formats (regex, full match)
FORMAT_RULES = {
    ("lms_loan_daily", "loan_id"):           r"CUS_0x[0-9a-f]+_\d{4}_\d{2}_\d{2}",
    ("lms_loan_daily", "loan_start_date"):   DATE_RE,
    ("features_attributes", "SSN"):          r"\d{3}-\d{2}-\d{4}",
    ("features_financials", "Credit_History_Age"): r"\d+ Years and \d+ Months",
    **{(t, "Customer_ID"): r"CUS_0x[0-9a-f]+" for t in SOURCES},
    **{(t, "snapshot_date"): DATE_RE for t in SOURCES},
}

# Plausibility bounds for numeric fields (inclusive). To be confirmed with the data owner.
RANGE_RULES = {
    ("features_attributes", "Age"):                    (14, 100),
    ("features_financials", "Annual_Income"):          (0, 1_000_000),
    ("features_financials", "Num_Bank_Accounts"):      (0, 50),
    ("features_financials", "Num_Credit_Card"):        (0, 50),
    ("features_financials", "Interest_Rate"):          (0, 100),
    ("features_financials", "Num_of_Loan"):            (0, 50),
    ("features_financials", "Delay_from_due_date"):    (0, 365),
    ("features_financials", "Num_of_Delayed_Payment"): (0, 365),
    ("features_financials", "Num_Credit_Inquiries"):   (0, 100),
    ("features_financials", "Total_EMI_per_month"):    (0, 50_000),
    ("lms_loan_daily", "installment_num"):             (0, 10),
    ("lms_loan_daily", "overdue_amt"):                 (0, 10_000),
}

# Tokens standing in for "missing" in categorical columns
PLACEHOLDER_RE = r"[^A-Za-z0-9]*|[^A-Za-z]+"
KNOWN_CODES = {("features_financials", "Payment_of_Min_Amount"): ["NM"]}


# =============================================================================================
# Spark session
# =============================================================================================
def get_spark(app_name="bronze"):
    # Spark must launch its Python processes with this interpreter (on Windows, "python" may not exist)
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
    # Windows only: Spark needs Hadoop's winutils.exe + hadoop.dll (installed in C:\hadoop\bin) to write files
    if os.name == "nt":
        os.environ.setdefault("HADOOP_HOME", r"C:\hadoop")
        hadoop_bin = os.path.join(os.environ["HADOOP_HOME"], "bin")
        if hadoop_bin not in os.environ["PATH"].split(os.pathsep):
            os.environ["PATH"] = hadoop_bin + os.pathsep + os.environ["PATH"]
    spark = (SparkSession.builder
             # cores: all by default; the Docker image sets SPARK_MASTER=local[4] (fewer parallel writes to the
             # mounted folder)
             .master(os.environ.get("SPARK_MASTER", "local[*]"))
             .appName(app_name)
             .config("spark.sql.session.timeZone", "UTC")
             # explicit driver memory: Spark's 1 GB default runs out of heap in Silver inside Docker; override
             # with SPARK_DRIVER_MEMORY (must be set before the first Spark session starts)
             .config("spark.driver.memory", os.environ.get("SPARK_DRIVER_MEMORY", "4g"))
             .getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")
    return spark


# =============================================================================================
# Helpers
# =============================================================================================
def file_sha256(path, chunk_size=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def source_profile(path):
    # Independent of Spark: header, number of data records, and number of empty fields per column.
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        n_rows, empties = 0, [0] * len(header)
        for row in reader:
            n_rows += 1
            for i, v in enumerate(row):
                if v == "":
                    empties[i] += 1
    return header, n_rows, dict(zip(header, empties))


def schema_hash(columns):
    return hashlib.sha256("|".join(columns).encode("utf-8")).hexdigest()[:16]


def read_log():
    if os.path.exists(LOG_PATH):
        return pd.read_csv(LOG_PATH, dtype=str)
    return pd.DataFrame(columns=LOG_COLUMNS, dtype=str)


def append_log(rows):
    log = pd.concat([read_log(), pd.DataFrame(rows, columns=LOG_COLUMNS).astype(str)], ignore_index=True)
    log.to_csv(LOG_PATH, index=False)


def last_loaded(log, table):
    # Most recent successful load of a table, or None.
    done = log[(log["table"] == table) & (log["status"] == "LOADED")]
    return None if done.empty else done.iloc[-1]


def next_schema_version(log, table, current_hash):
    prev = last_loaded(log, table)
    if prev is None:
        return 1
    prev_version = int(prev["schema_version"])
    return prev_version if prev["schema_hash"] == current_hash else prev_version + 1


# =============================================================================================
# Step 02 — Ingestion
# =============================================================================================
def read_source_as_is(spark, path, header):
    # Every column as string; header must match the expected columns; any malformed row fails the load.
    schema = T.StructType([T.StructField(c, T.StringType(), True) for c in header])
    df = (spark.read
          .option("header", True)
          .option("enforceSchema", False)   # validate the file header against `schema`
          .option("mode", "FAILFAST")
          .option("escape", '"')
          .schema(schema)
          .csv(path))

    # Attach the 1-based data-row number in file order, computed in the JVM (no Python workers):
    # monotonically_increasing_id follows file order across partitions; row_number makes it consecutive.
    return (df.withColumn("_mono_id", F.monotonically_increasing_id())
              .withColumn("_source_row_num", F.row_number().over(Window.orderBy("_mono_id")).cast("long"))
              .drop("_mono_id"))


def write_bronze_csv(df, table, event_col, out_path):
    # One CSV per event date, written to a temp folder and swapped in (snapshot overwrite, atomic per table).
    pdf = df.orderBy("_source_row_num").toPandas()
    tmp_path = out_path + "__tmp"
    shutil.rmtree(tmp_path, ignore_errors=True)
    os.makedirs(tmp_path)
    for snap, part in pdf.groupby(event_col, sort=True, dropna=False):
        suffix = snap.replace("-", "_") if isinstance(snap, str) else "unknown"
        part.to_csv(os.path.join(tmp_path, f"bronze_{table}_{suffix}.csv"), index=False)
    shutil.rmtree(out_path, ignore_errors=True)
    os.replace(tmp_path, out_path)


def read_bronze(spark, table):
    # All columns back as strings, exactly as written.
    return (spark.read
            .option("header", True)
            .option("escape", '"')
            .option("inferSchema", False)
            .option("pathGlobFilter", f"bronze_{table}_*.csv")   # filter by name, not a glob in the path
            .csv(os.path.join(BRONZE_DIR, table)))                # (Windows backslashes break path globs)


def ingest_source(spark, table, batch_id, force=False):
    file_name, event_col = SOURCES[table]
    src_path = os.path.join(RAW_DIR, file_name)
    out_path = os.path.join(BRONZE_DIR, table)
    started = time.time()
    ingest_ts = datetime.now(timezone.utc).replace(microsecond=0)

    sha = file_sha256(src_path)
    header, src_rows, _ = source_profile(src_path)
    s_hash = schema_hash(header)
    log = read_log()
    prev = last_loaded(log, table)

    base = dict(batch_id=batch_id, table=table, source_file=file_name, source_sha256=sha,
                source_rows=src_rows, schema_hash=s_hash, ingest_ts=ingest_ts.isoformat())

    # Idempotency: same file already loaded and still on disk -> no-op
    if not force and prev is not None and prev["source_sha256"] == sha and os.path.isdir(out_path):
        row = {**base, "bronze_rows": prev["bronze_rows"], "event_partitions": prev["event_partitions"],
               "min_event_date": prev["min_event_date"], "max_event_date": prev["max_event_date"],
               "schema_version": prev["schema_version"], "mode": "skip", "status": "SKIPPED_UNCHANGED",
               "duration_s": round(time.time() - started, 2),
               "message": f"unchanged since batch {prev['batch_id']}"}
        append_log([row])
        return row

    if event_col not in header:
        raise ValueError(f"{table}: event-time column '{event_col}' not in source header {header}")

    s_version = next_schema_version(log, table, s_hash)
    try:
        df = (read_source_as_is(spark, src_path, header)
              .withColumn("_source_file",    F.lit(file_name))
              .withColumn("_source_sha256",  F.lit(sha))
              .withColumn("_event_date",     F.date_format(F.to_date(F.col(event_col), "yyyy-MM-dd"), "yyyy-MM-dd"))
              .withColumn("_ingest_ts",      F.lit(ingest_ts.strftime("%Y-%m-%dT%H:%M:%SZ")))
              .withColumn("_batch_id",       F.lit(batch_id))
              .withColumn("_schema_version", F.lit(s_version))
              .withColumn("_schema_hash",    F.lit(s_hash)))

        write_bronze_csv(df, table, event_col, out_path)

        stats = (read_bronze(spark, table)
                 .agg(F.count("*").alias("n"),
                      F.countDistinct("_event_date").alias("parts"),
                      F.min("_event_date").alias("lo"),
                      F.max("_event_date").alias("hi"))
                 .first())
        status = "LOADED" if stats["n"] == src_rows else "FAILED_COUNT_MISMATCH"
        row = {**base, "bronze_rows": stats["n"], "event_partitions": stats["parts"],
               "min_event_date": stats["lo"], "max_event_date": stats["hi"],
               "schema_version": s_version, "mode": "snapshot_overwrite", "status": status,
               "duration_s": round(time.time() - started, 2),
               "message": "" if status == "LOADED" else f"source {src_rows} != bronze {stats['n']}"}
    except Exception as e:
        row = {**base, "bronze_rows": "", "event_partitions": "", "min_event_date": "", "max_event_date": "",
               "schema_version": s_version, "mode": "snapshot_overwrite", "status": "FAILED",
               "duration_s": round(time.time() - started, 2),
               "message": f"{type(e).__name__}: {e}"[:500]}
        append_log([row])
        raise

    append_log([row])
    return row


def ingest_all(spark, force=False):
    os.makedirs(BRONZE_DIR, exist_ok=True)
    batch_id = f"bronze_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{uuid.uuid4().hex[:6]}"
    rows = [ingest_source(spark, t, batch_id, force=force) for t in SOURCES]
    return pd.DataFrame(rows)[["batch_id", "table", "status", "source_rows", "bronze_rows",
                               "event_partitions", "min_event_date", "max_event_date",
                               "schema_version", "duration_s", "message"]]


def run_ingestion_tests(spark):
    # Step 02 tests: idempotent reruns and as-is values. NOTE: test 2 performs a forced reload.
    before = {t: read_bronze(spark, t).select("_batch_id").distinct().collect() for t in SOURCES}

    # Test 1: rerun with unchanged files is a no-op
    rerun = ingest_all(spark)
    assert (rerun["status"] == "SKIPPED_UNCHANGED").all(), rerun
    for t in SOURCES:
        assert read_bronze(spark, t).select("_batch_id").distinct().collect() == before[t], \
            f"{t}: data rewritten on no-op rerun"
    print("Test 1 passed: rerun skipped all tables, Bronze untouched")

    # Test 2: forced reload replaces instead of appending
    forced = ingest_all(spark, force=True)
    assert (forced["status"] == "LOADED").all(), forced
    for t in SOURCES:
        df = read_bronze(spark, t)
        src_rows = source_profile(os.path.join(RAW_DIR, SOURCES[t][0]))[1]
        assert df.count() == src_rows, f"{t}: {df.count()} rows != source {src_rows}"
        assert df.select("_batch_id").distinct().count() == 1, f"{t}: rows from more than one batch"
        assert df.select("_source_row_num").distinct().count() == src_rows, f"{t}: duplicate source IDs"
    print("Test 2 passed: forced reload replaced each table, no duplicated loads")

    # Tests 3-5
    for t, (file_name, _) in SOURCES.items():
        path = os.path.join(RAW_DIR, file_name)
        header, _, empties = source_profile(path)
        df = read_bronze(spark, t)

        # Test 3: values identical to the source parsed independently with Python's csv module
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)
            rows = [tuple(v if v != "" else None for v in r) + (i,) for i, r in enumerate(reader, start=1)]
        raw = spark.createDataFrame(rows, T.StructType([T.StructField(c, T.StringType()) for c in header]
                                                       + [T.StructField("_source_row_num", T.LongType())]))
        biz = df.select(*header, F.col("_source_row_num").cast("long"))
        assert biz.exceptAll(raw).count() == 0 and raw.exceptAll(biz).count() == 0, f"{t}: values differ from source"

        # Test 4: nulls in Bronze == empty fields in source
        nulls = df.select([F.sum(F.col(c).isNull().cast("int")).alias(c) for c in header]).first().asDict()
        assert {c: nulls[c] or 0 for c in header} == empties, f"{t}: null/empty mismatch"

        # Test 5: lineage columns fully populated
        meta_nulls = df.select([F.sum(F.col(c).isNull().cast("int")).alias(c) for c in META_COLS]).first().asDict()
        assert all((v or 0) == 0 for v in meta_nulls.values()), f"{t}: null lineage {meta_nulls}"
    print("Tests 3-5 passed: values as-is, empty fields preserved as null, lineage complete")


# =============================================================================================
# Step 03 — Source audit
# =============================================================================================
def month_diff(a, b):
    # Whole months from date string a to date string b ('YYYY-MM-DD')
    return (int(b[:4]) - int(a[:4])) * 12 + (int(b[5:7]) - int(a[5:7]))


def month_range(lo, hi):
    # First-of-month dates from lo to hi inclusive
    y, m, out = int(lo[:4]), int(lo[5:7]), []
    while (y, m) <= (int(hi[:4]), int(hi[5:7])):
        out.append(f"{y:04d}-{m:02d}-01")
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def source_counts_by(path, col):
    # Rows per value of `col`, counted directly from the source file (independent of Spark)
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        i = next(reader).index(col)
        counts = {}
        for row in reader:
            counts[row[i]] = counts.get(row[i], 0) + 1
    return counts


def audit_table(spark, table, as_of):
    file_name, event_col = SOURCES[table]
    src_path = os.path.join(RAW_DIR, file_name)
    keys = KEYS[table]
    df = read_bronze(spark, table).cache()
    checks, exceptions = [], []

    def check(name, status, observed, threshold):
        checks.append(dict(table=table, check=name, status=status, observed=observed, threshold=threshold))

    def exception(name, key, detail):
        exceptions.append(dict(table=table, check=name, key=key, detail=detail))

    # 1. Row count reconciliation
    header, src_rows, _ = source_profile(src_path)
    b_rows = df.count()
    diff_pct = abs(b_rows - src_rows) / max(src_rows, 1) * 100
    check("row_count_reconciliation", "PASS" if diff_pct <= TOLERANCES["count_diff_pct"] else "FAIL",
          f"source={src_rows:,}  bronze={b_rows:,}", "exact match")

    # 2. Per-partition reconciliation
    src_parts = source_counts_by(src_path, event_col)
    b_parts = {r[event_col]: r["count"] for r in df.groupBy(event_col).count().collect()}
    mismatched = [p for p in sorted(set(src_parts) | set(b_parts)) if src_parts.get(p, 0) != b_parts.get(p, 0)]
    for p in mismatched:
        exception("partition_count_reconciliation", p, f"source={src_parts.get(p, 0)}  bronze={b_parts.get(p, 0)}")
    check("partition_count_reconciliation", "PASS" if not mismatched else "FAIL",
          f"{len(src_parts)} partitions, {len(mismatched)} mismatched", "exact match")

    # 3. Bronze built from the current source file
    stale = df.filter(F.col("_source_sha256") != file_sha256(src_path)).count()
    check("bronze_matches_current_source", "PASS" if stale == 0 else "FAIL",
          f"{stale:,} rows from an older file version", "0 stale rows")

    # 4. Freshness
    first_event, last_event = min(b_parts), max(b_parts)
    lag = month_diff(last_event, as_of)
    status = ("FAIL" if lag > TOLERANCES["freshness_fail_months"]
              else "WARN" if lag > TOLERANCES["freshness_warn_months"] else "PASS")
    check("freshness", status, f"latest snapshot {last_event}, {lag} month(s) behind {as_of}",
          f"WARN > {TOLERANCES['freshness_warn_months']}, FAIL > {TOLERANCES['freshness_fail_months']} months")
    if status != "PASS":
        exception("freshness", last_event, f"latest snapshot is {lag} month(s) behind the as-of date {as_of}")

    # 5. Missing partitions (monthly calendar between first and last snapshot)
    expected = month_range(first_event, last_event)
    missing = [m for m in expected if m not in b_parts]
    off_calendar = [p for p in b_parts if p not in expected]
    for m in missing:
        exception("missing_partitions", m, "no rows for this month")
    for p in off_calendar:
        exception("missing_partitions", p, "snapshot_date is not the first of a month")
    check("missing_partitions", "PASS" if not (missing or off_calendar) else "FAIL",
          f"{len(expected)} months {first_event}..{last_event}: {len(missing)} missing, {len(off_calendar)} off-calendar",
          "0 missing")

    # 6. Null keys
    null_rows = df.filter(F.greatest(*[F.col(k).isNull().cast("int") for k in keys]) == 1)
    n_null = null_rows.count()
    pct = n_null / b_rows * 100
    check("null_keys", "PASS" if pct <= TOLERANCES["null_key_pct"] else "FAIL",
          f"{n_null:,} rows ({pct:.2f}%) with a null in {keys}", f"<= {TOLERANCES['null_key_pct']}%")
    for r in null_rows.select("_source_row_num", *keys).limit(50).collect():
        exception("null_keys", f"row {r['_source_row_num']}", str({k: r[k] for k in keys}))

    # 7. Duplicate keys
    dups = df.groupBy(*keys).count().filter("count > 1").cache()
    n_dup_keys = dups.count()
    n_dup_rows = dups.agg(F.sum("count")).first()[0] or 0
    pct = n_dup_rows / b_rows * 100
    check("duplicate_keys", "PASS" if pct <= TOLERANCES["dup_key_pct"] else "FAIL",
          f"{n_dup_keys:,} keys / {n_dup_rows:,} rows ({pct:.2f}%)", f"<= {TOLERANCES['dup_key_pct']}%")
    for r in dups.limit(50).collect():
        exception("duplicate_keys", " | ".join(str(r[k]) for k in keys), f"{r['count']} rows")
    dups.unpersist()

    # 8. Schema drift vs the last successful load
    loaded = read_log().query("table == @table and status == 'LOADED'")
    last = loaded.iloc[-1]
    versions_seen = loaded["schema_hash"].nunique()
    drift = schema_hash(header) != last["schema_hash"]
    check("schema_drift", "FAIL" if drift else ("WARN" if versions_seen > 1 else "PASS"),
          f"schema v{last['schema_version']} ({last['schema_hash']}), {versions_seen} version(s) in log",
          "header unchanged")
    if drift:
        exception("schema_drift", last["schema_hash"], f"current header hash {schema_hash(header)}")

    # 9. Bronze column layout
    expected_cols = header + META_COLS
    ok = df.columns == expected_cols
    check("bronze_columns", "PASS" if ok else "FAIL",
          f"{len(df.columns)} columns" + ("" if ok else f"; missing={sorted(set(expected_cols) - set(df.columns))}, "
                                                         f"extra={sorted(set(df.columns) - set(expected_cols))}"),
          "source header + lineage")

    df.unpersist()
    return checks, exceptions


def run_audit(spark):
    os.makedirs(AUDIT_DIR, exist_ok=True)
    as_of = AS_OF_DATE or max(read_bronze(spark, t).agg(F.max(SOURCES[t][1])).first()[0] for t in SOURCES)

    all_checks, all_exceptions = [], []
    for t in SOURCES:
        c, e = audit_table(spark, t, as_of)
        all_checks += c
        all_exceptions += e

    audit_report = pd.DataFrame(all_checks)
    audit_exceptions = pd.DataFrame(all_exceptions, columns=["table", "check", "key", "detail"])
    audit_report.insert(0, "audit_ts", datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
    audit_report.to_csv(os.path.join(AUDIT_DIR, "audit_report.csv"), index=False)
    audit_exceptions.to_csv(os.path.join(AUDIT_DIR, "audit_exceptions.csv"), index=False)
    return audit_report, audit_exceptions, as_of


# =============================================================================================
# Step 04 — Raw EDA profiling
# =============================================================================================
def load_raw(spark):
    # Bronze business columns as pandas strings (Bronze itself is not modified)
    raw = {}
    for t, (file_name, _) in SOURCES.items():
        header = source_profile(os.path.join(RAW_DIR, file_name))[0]
        raw[t] = read_bronze(spark, t).select(*header).toPandas()
    return raw


def profile_column(table, col, s):
    n = len(s)
    v = s.dropna()
    vc = v.value_counts()
    is_num = v.str.fullmatch(NUMERIC_RE)
    numeric_pct = is_num.mean() * 100 if len(v) else 0.0
    row = dict(table=table, column=col, rows=n,
               null_count=n - len(v), null_pct=(n - len(v)) / n * 100,
               distinct=int(v.nunique()), unique_pct=v.nunique() / n * 100,
               numeric_pct=numeric_pct,
               top_values="; ".join(f"{k} ({c:,})" for k, c in vc.head(5).items()))

    if numeric_pct >= NUMERIC_MIN_PCT:
        x = pd.to_numeric(v[is_num])
        q1, q3 = x.quantile([0.25, 0.75])
        iqr = q3 - q1
        bad = v[~is_num]
        row.update(inferred_type="numeric",
                   non_numeric_count=len(bad), non_numeric_pct=len(bad) / n * 100,
                   non_numeric_examples="; ".join(f"{k} ({c:,})" for k, c in bad.value_counts().head(5).items()),
                   min=x.min(), p01=x.quantile(0.01), p05=x.quantile(0.05), p25=q1, p50=x.median(),
                   p75=q3, p95=x.quantile(0.95), p99=x.quantile(0.99), max=x.max(),
                   mean=x.mean(), std=x.std(), skew=x.skew(),
                   negative_count=int((x < 0).sum()),
                   iqr_outliers=int(((x < q1 - 1.5 * iqr) | (x > q3 + 1.5 * iqr)).sum()))
    elif len(v) and v.str.fullmatch(DATE_RE).mean() >= 0.95:
        row.update(inferred_type="date", min=v.min(), max=v.max())
    else:
        row.update(inferred_type="text")
    return row


def build_profile(raw):
    # Column profile + baselines for step 08 (category frequencies, partition counts)
    os.makedirs(EDA_DIR, exist_ok=True)
    profile = pd.DataFrame([profile_column(t, c, df[c]) for t, df in raw.items() for c in df.columns])

    dup_rows = {t: int(df.duplicated().sum()) for t, df in raw.items()}
    profile["table_duplicate_rows"] = profile["table"].map(dup_rows)
    profile.to_csv(os.path.join(EDA_DIR, "bronze_profile_baseline.csv"), index=False)

    freq = pd.concat([
        df[c].fillna("<NULL>").value_counts().rename_axis("value").reset_index(name="count")
             .assign(table=t, column=c, pct=lambda d: d["count"] / len(df) * 100)
        for t, df in raw.items() for c in df.columns if df[c].nunique() <= LOW_CARDINALITY
    ])[["table", "column", "value", "count", "pct"]]
    freq.to_csv(os.path.join(EDA_DIR, "bronze_category_freq.csv"), index=False)

    partition_counts = pd.concat([
        df.groupby("snapshot_date").size().rename("rows").reset_index().assign(table=t) for t, df in raw.items()
    ])[["table", "snapshot_date", "rows"]]
    partition_counts.to_csv(os.path.join(EDA_DIR, "bronze_partition_counts.csv"), index=False)
    return profile, freq, partition_counts


def build_issues(raw, profile, partition_counts):
    # Suspect fields and source anomalies — findings only, no cleaning decisions
    issues, dq = [], []   # dq: per-column counts by problem type, for the data-quality chart

    def add_issue(table, column, category, finding, count, examples="", severity=None, pct=None):
        # pct defaults to count / table rows; pass it explicitly when the base is different (e.g. customers)
        if pct is None and count is not None:
            pct = count / len(raw[table]) * 100
        if severity is None:
            severity = "high" if pct is not None and pct >= 5 else "medium"
        issues.append(dict(table=table, column=column, category=category, finding=finding,
                           evidence_count=count, evidence_pct=None if pct is None else round(pct, 2),
                           examples=examples, severity=severity, status="open"))

    for _, p in profile.iterrows():
        t, c = p["table"], p["column"]
        s = raw[t][c].dropna()
        counts = dict(missing=int(p["null_count"]), invalid_token=0, bad_format=0, out_of_range=0)

        if p["null_count"] > 0:
            add_issue(t, c, "missing", "Empty values in source", int(p["null_count"]))

        if p["inferred_type"] == "numeric" and p["non_numeric_count"] > 0:
            counts["invalid_token"] = int(p["non_numeric_count"])
            add_issue(t, c, "invalid_token", "Non-numeric tokens in a numeric field",
                      int(p["non_numeric_count"]), p["non_numeric_examples"])

        if p["inferred_type"] == "text" and p["distinct"] <= LOW_CARDINALITY:
            tokens = s[s.str.fullmatch(PLACEHOLDER_RE) | s.isin(KNOWN_CODES.get((t, c), []))]
            if len(tokens):
                counts["invalid_token"] = len(tokens)
                add_issue(t, c, "placeholder", "Placeholder / junk category standing in for missing",
                          len(tokens), "; ".join(f"{k} ({n:,})" for k, n in tokens.value_counts().items()),
                          severity="high" if len(tokens) / len(raw[t]) >= 0.05 else "medium")

        if (t, c) in FORMAT_RULES:
            bad = s[~s.str.fullmatch(FORMAT_RULES[(t, c)])]
            if len(bad):
                counts["bad_format"] = len(bad)
                add_issue(t, c, "format", f"Values not matching expected format /{FORMAT_RULES[(t, c)]}/",
                          len(bad), "; ".join(f"{k} ({n:,})" for k, n in bad.value_counts().head(5).items()))

        if (t, c) in RANGE_RULES:
            lo, hi = RANGE_RULES[(t, c)]
            x = pd.to_numeric(s[s.str.fullmatch(NUMERIC_RE)])
            out = x[(x < lo) | (x > hi)]
            if len(out):
                counts["out_of_range"] = len(out)
                add_issue(t, c, "out_of_range", f"Outside plausible range [{lo:,}, {hi:,}]", len(out),
                          f"min={out.min():,.2f}; max={out.max():,.2f}")
        elif p["inferred_type"] == "numeric" and p["p99"] > 0 and p["max"] > 10 * p["p99"]:
            add_issue(t, c, "outlier", "Extreme values: max more than 10x the 99th percentile",
                      int(p["iqr_outliers"]), f"p99={p['p99']:,.2f}; max={p['max']:,.2f}")

        if p["distinct"] == 1:
            add_issue(t, c, "constant", "Single value in every row (no information)", int(p["rows"]),
                      p["top_values"], severity="low")

        if (t, c) in PII_COLUMNS:
            add_issue(t, c, "privacy", "Personal identifier (PII)", int(p["rows"] - p["null_count"]), "",
                      severity="high")

        dq.append(dict(table=t, column=c, **counts))

    # Structural findings that need parsing rather than a simple rule
    fin = raw["features_financials"]
    add_issue("features_financials", "Credit_History_Age", "structure",
              "Duration stored as text ('X Years and Y Months') — needs parsing to months", len(fin),
              fin["Credit_History_Age"].dropna().iloc[0], severity="medium")
    add_issue("features_financials", "Type_of_Loan", "structure",
              "Multi-valued list in one field ('A, B, and C')", int(fin["Type_of_Loan"].str.contains(",").sum()),
              fin["Type_of_Loan"].dropna().iloc[1], severity="medium")

    # Cross-table coverage
    customers = {t: set(df["Customer_ID"].dropna()) for t, df in raw.items()}
    base = customers["lms_loan_daily"]
    for t in SOURCES:
        missing = len(base - customers[t])
        if missing:
            add_issue(t, "Customer_ID", "coverage", f"Loan customers with no rows in {t} (% of loan customers)",
                      missing, f"{len(customers[t] & base):,} of {len(base):,} loan customers covered",
                      severity="high", pct=missing / len(base) * 100)
    date_ranges = {t: (df["snapshot_date"].min(), df["snapshot_date"].max()) for t, df in raw.items()}
    lms_hi = date_ranges["lms_loan_daily"][1]
    for t, (lo, hi) in date_ranges.items():
        if hi < lms_hi:
            add_issue(t, "snapshot_date", "coverage", f"Snapshots end at {hi}, loan data runs to {lms_hi}",
                      None, f"{t}: {lo}..{hi}", severity="medium")

    # Sampling gaps: partitions far below the table's median volume
    for t, g in partition_counts.groupby("table"):
        med = g["rows"].median()
        low = g[g["rows"] < 0.5 * med]
        if len(low):
            add_issue(t, "snapshot_date", "sampling_gap",
                      f"{len(low)} partition(s) with < 50% of median volume ({med:,.0f} rows)",
                      int(low["rows"].sum()), "; ".join(f"{r.snapshot_date} ({r.rows:,})" for r in low.itertuples()),
                      severity="low")

    issues_df = pd.DataFrame(issues)
    sev_order = {"high": 0, "medium": 1, "low": 2}
    issues_df = (issues_df.sort_values(["severity", "table", "column"],
                                       key=lambda s: s.map(sev_order) if s.name == "severity" else s)
                          .reset_index(drop=True))
    issues_df.insert(0, "issue_id", [f"BRZ-{i:03d}" for i in range(1, len(issues_df) + 1)])
    issues_df.to_csv(os.path.join(EDA_DIR, "raw_eda_issues.csv"), index=False)
    return issues_df, pd.DataFrame(dq), customers


# ---------------------------------------------------------------------------------------------
# Charts (saved as PNG in _eda/charts/; returned figures can be shown in a notebook)
# ---------------------------------------------------------------------------------------------
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]   # fixed categorical order (validated palette)


def _chart_style():
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
        "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.labelsize": 9, "xtick.color": INK_2, "ytick.color": INK_2,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "legend.fontsize": 8, "font.size": 9,
        "lines.linewidth": 2,
    })
    return plt


def _save(fig, name):
    os.makedirs(CHART_DIR, exist_ok=True)
    fig.savefig(os.path.join(CHART_DIR, f"{name}.png"), dpi=130, bbox_inches="tight")
    return fig


def plot_timestamp_coverage(partition_counts):
    plt = _chart_style()
    import matplotlib.ticker as mticker
    fig, axes = plt.subplots(2, 2, figsize=(12, 6), sharex=True)
    for ax, t in zip(axes.flat, SOURCES):
        g = partition_counts[partition_counts["table"] == t]
        x = pd.to_datetime(g["snapshot_date"])
        ax.plot(x, g["rows"], color=SERIES[0], marker="o", markersize=3)
        ax.set_title(t)
        ax.set_ylim(bottom=0)
        ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
        ax.set_xlim(pd.Timestamp("2022-12-01"), pd.Timestamp("2025-12-01"))
    for ax in axes[:, 0]:
        ax.set_ylabel("rows per snapshot")
    fig.suptitle("Timestamp coverage — rows per snapshot_date", x=0.01, ha="left", fontweight="bold", color=INK)
    fig.tight_layout()
    return _save(fig, "01_timestamp_coverage")


def plot_data_quality(raw, dq_df):
    plt = _chart_style()
    labels = {"missing": "Missing", "invalid_token": "Invalid / placeholder token",
              "bad_format": "Bad format", "out_of_range": "Out of plausible range"}
    d = dq_df.copy()
    d["rows"] = d["table"].map({t: len(df) for t, df in raw.items()})
    for k in labels:
        d[k] = d[k] / d["rows"] * 100
    d["total"] = d[list(labels)].sum(axis=1)
    d = d[d["total"] > 0].sort_values("total")
    d["name"] = d["table"].str.replace("features_", "") + " · " + d["column"]

    fig, ax = plt.subplots(figsize=(11, 0.32 * len(d) + 1.2))
    left = np.zeros(len(d))
    for color, (k, lab) in zip(SERIES, labels.items()):
        ax.barh(d["name"], d[k], left=left, color=color, label=lab, height=0.7, edgecolor=SURFACE, linewidth=1)
        left += d[k].values
    for y, v in enumerate(d["total"]):
        ax.text(v + 0.3, y, f"{v:.1f}%" if v >= 0.1 else "<0.1%", va="center", fontsize=8, color=INK_2)
    ax.set_xlabel("% of rows")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right")
    ax.set_title("Data quality by column — % of rows with a problem (raw Bronze)")
    fig.tight_layout()
    return _save(fig, "02_data_quality_by_column")


def plot_numeric_outliers(raw):
    plt = _chart_style()
    fields = [("features_attributes", "Age"), ("features_financials", "Annual_Income"),
              ("features_financials", "Num_Bank_Accounts"), ("features_financials", "Interest_Rate"),
              ("features_financials", "Num_of_Loan"), ("features_financials", "Total_EMI_per_month")]
    # symmetric log: log scale for |x| > 1, linear around 0, so negative values are shown in real units
    to_sym = lambda v: np.sign(v) * np.log10(1 + np.abs(v))
    from_sym = lambda v: np.sign(v) * (10 ** np.abs(v) - 1)

    fig, axes = plt.subplots(2, 3, figsize=(13, 6.4))
    for ax, (t, c) in zip(axes.flat, fields):
        s = raw[t][c].dropna()
        x = pd.to_numeric(s[s.str.fullmatch(NUMERIC_RE)])
        lo, hi = RANGE_RULES[(t, c)]
        ax.axvspan(lo, hi, color=SERIES[0], alpha=0.08, lw=0)
        bins = from_sym(np.linspace(to_sym(min(x.min(), 0)), to_sym(x.max()), 60))
        ax.hist(x, bins=bins, color=SERIES[0], edgecolor=SURFACE, linewidth=0.4)
        ax.set_xscale("symlog", linthresh=1)
        ax.set_yscale("log")
        n_out = int(((x < lo) | (x > hi)).sum())
        ax.set_title(c, pad=16)
        ax.text(0, 1.01, f"{n_out:,} outside [{lo:,}, {hi:,}] · max {x.max():,.0f}",
                transform=ax.transAxes, ha="left", va="bottom", fontsize=8, color=INK_2)
        ax.set_xlabel("value (symmetric log scale)")
    for ax in axes[:, 0]:
        ax.set_ylabel("rows (log)")
    fig.suptitle("Raw numeric distributions — shaded band = plausible range", x=0.01, ha="left",
                 fontweight="bold", color=INK)
    fig.tight_layout()
    return _save(fig, "03_numeric_outliers")


def plot_categorical_frequencies(raw):
    plt = _chart_style()
    cats = [("features_attributes", "Occupation"), ("features_financials", "Credit_Mix"),
            ("features_financials", "Payment_of_Min_Amount"), ("features_financials", "Payment_Behaviour")]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), gridspec_kw={"height_ratios": [2.2, 1]})
    for ax, (t, c) in zip(axes.flat, cats):
        vc = raw[t][c].fillna("<NULL>").value_counts().sort_values()
        is_ph = vc.index.str.fullmatch(PLACEHOLDER_RE) | vc.index.isin(KNOWN_CODES.get((t, c), []))
        names = [f"'{v}'" if p else v for v, p in zip(vc.index, is_ph)]   # quote tokens so '_' stays visible
        ax.barh(names, vc.values, color=[SERIES[1] if p else SERIES[0] for p in is_ph], height=0.7)
        for y, v in enumerate(vc.values):
            ax.text(v, y, f" {v:,} ({v / len(raw[t]) * 100:.1f}%)", va="center", fontsize=7.5, color=INK_2)
        ax.set_title(c)
        ax.grid(axis="y", visible=False)
        ax.set_xlim(0, vc.max() * 1.25)
    handles = [plt.Rectangle((0, 0), 1, 1, color=SERIES[0]), plt.Rectangle((0, 0), 1, 1, color=SERIES[1])]
    fig.legend(handles, ["Valid category", "Placeholder / junk / unclear code"], loc="upper right", ncol=2)
    fig.suptitle("Categorical frequencies (raw Bronze)", x=0.01, ha="left", fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return _save(fig, "04_categorical_frequencies")


def plot_clickstream(raw):
    plt = _chart_style()
    cs = raw["feature_clickstream"]
    fe_cols = [f"fe_{i}" for i in range(1, 21)]
    fig, ax = plt.subplots(figsize=(13, 4.2))
    ax.boxplot([pd.to_numeric(cs[c]) for c in fe_cols], tick_labels=fe_cols, widths=0.6, showfliers=False,
               patch_artist=True, boxprops=dict(facecolor="#cde2fb", edgecolor=SERIES[0]),
               medianprops=dict(color=SERIES[0], linewidth=2), whiskerprops=dict(color=SERIES[0]),
               capprops=dict(color=SERIES[0]))
    ax.axhline(0, color=INK_2, linewidth=0.8)
    ax.set_ylabel("value")
    ax.set_title("Clickstream fe_1 … fe_20 — box = IQR, whiskers = 1.5×IQR (outliers hidden)")
    fig.tight_layout()
    return _save(fig, "05_clickstream_distributions")


def plot_loan_ledger(raw):
    plt = _chart_style()
    import matplotlib.ticker as mticker
    lms = raw["lms_loan_daily"].copy()
    lms["overdue"] = pd.to_numeric(lms["overdue_amt"]) > 0
    starts = lms.drop_duplicates("loan_id").groupby("loan_start_date").size()
    overdue_share = lms.groupby("snapshot_date")["overdue"].mean() * 100

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 3.8), sharex=True)
    ax1.bar(pd.to_datetime(starts.index), starts.values, width=20, color=SERIES[0])
    ax1.set_title("New loans per loan_start_date")
    ax1.set_ylabel("loans")
    ax2.plot(pd.to_datetime(overdue_share.index), overdue_share.values, color=SERIES[0], marker="o", markersize=3)
    ax2.set_title("Share of loan snapshots with overdue_amt > 0")
    ax2.set_ylabel("% of snapshots")
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
    ax2.set_ylim(bottom=0)
    fig.tight_layout()
    return _save(fig, "06_loan_ledger_over_time")


def make_charts(raw, dq_df, partition_counts, close=True):
    figs = [plot_timestamp_coverage(partition_counts), plot_data_quality(raw, dq_df),
            plot_numeric_outliers(raw), plot_categorical_frequencies(raw),
            plot_clickstream(raw), plot_loan_ledger(raw)]
    if close:
        import matplotlib.pyplot as plt
        for f in figs:
            plt.close(f)
    return figs


def coverage_table(raw, customers):
    base = customers["lms_loan_daily"]
    return pd.DataFrame([
        dict(table=t, rows=len(df), customers=df["Customer_ID"].nunique(),
             pct_of_loan_customers=len(customers[t] & base) / len(base) * 100,
             first_snapshot=df["snapshot_date"].min(), last_snapshot=df["snapshot_date"].max(),
             snapshots=df["snapshot_date"].nunique(), rows_per_customer=len(df) / df["Customer_ID"].nunique())
        for t, df in raw.items()])


def html_profiles(raw):
    # ydata-profiling HTML reports: minimal mode, PII columns left out, no sample rows (1-2 minutes)
    import warnings
    from ydata_profiling import ProfileReport

    written = []
    for t, df in raw.items():
        view = df.drop(columns=[c for c in df.columns if (t, c) in PII_COLUMNS])
        for c in view.columns:                         # numeric view for the HTML report only
            num = view[c].dropna().str.fullmatch(NUMERIC_RE)
            if len(num) and num.mean() >= 0.9:
                view[c] = pd.to_numeric(view[c].where(view[c].str.fullmatch(NUMERIC_RE, na=False)))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            report = ProfileReport(view, title=f"Bronze raw profile — {t}", minimal=True, progress_bar=False,
                                   samples={"head": 0, "tail": 0, "random": 0},
                                   duplicates={"head": 0})
            path = os.path.join(EDA_DIR, f"bronze_profile_{t}.html")
            report.to_file(path)
        written.append(path)
    return written


def run_eda(spark, charts=True, html=False):
    raw = load_raw(spark)
    profile, freq, partition_counts = build_profile(raw)
    issues_df, dq_df, customers = build_issues(raw, profile, partition_counts)
    if charts:
        make_charts(raw, dq_df, partition_counts)
    if html:
        html_profiles(raw)
    return dict(raw=raw, profile=profile, freq=freq, partition_counts=partition_counts,
                issues=issues_df, dq=dq_df, coverage=coverage_table(raw, customers))


# =============================================================================================
# Entry point
# =============================================================================================
def run(spark=None, force=False, audit=True, eda=True, charts=True, html=False, tests=False):
    """Run the Bronze layer: ingest (step 02), audit (step 03), raw EDA (step 04)."""
    spark = spark or get_spark()
    out = {"ingestion": ingest_all(spark, force=force)}
    if tests:
        run_ingestion_tests(spark)
    if audit:
        report, exceptions, as_of = run_audit(spark)
        out.update(audit_report=report, audit_exceptions=exceptions, as_of=as_of)
    if eda:
        out.update(run_eda(spark, charts=charts, html=html))
    return out


if __name__ == "__main__":
    import argparse
    import matplotlib
    matplotlib.use("Agg")

    parser = argparse.ArgumentParser(description="Bronze layer: ingest, audit, raw EDA")
    parser.add_argument("--force", action="store_true", help="reload even if source files are unchanged")
    parser.add_argument("--skip-eda", action="store_true", help="ingest + audit only")
    parser.add_argument("--html-profiles", action="store_true", help="also write ydata-profiling HTML reports")
    parser.add_argument("--tests", action="store_true", help="run ingestion tests (includes a forced reload)")
    args = parser.parse_args()

    result = run(force=args.force, eda=not args.skip_eda, html=args.html_profiles, tests=args.tests)
    print(result["ingestion"][["table", "status", "source_rows", "bronze_rows"]].to_string(index=False))
    print("\nAudit:", result["audit_report"]["status"].value_counts().to_dict())
    if "issues" in result:
        print("EDA issues:", result["issues"]["severity"].value_counts().to_dict())
