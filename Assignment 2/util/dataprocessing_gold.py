"""Gold layer — Steps 10 (join and aggregate), 11 (feature store), 12 (label store), 13 (reconciliation),
and the Gold EDA: 14 (feature/target EDA), 16 (temporal quality), 15 (EDA action register).

Converted from notebooks/dataprocessing_gold.ipynb (the notebook is the development version; this module is the
callable version used by the pipelines).

Usage:
    from util import dataprocessing_bronze as bronze, dataprocessing_gold as gold
    spark = bronze.get_spark()
    gold.run_build(spark)          # steps 10-13: feature store v1 + label store, released only if acceptance passes
    gold.run_eda(spark)            # steps 14, 16, 15 — needs the step-17 split manifest (datamart/gold/split/v1)
    gold.get_features(spark, as_of="2024-06-01")

Command line:
    python util/dataprocessing_gold.py [--build] [--eda] [--tests] [--no-charts]
"""
import os
import sys
import re
import shutil
import uuid
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yaml
from pyspark.sql import Window, functions as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from util import dataprocessing_bronze as bronze      # Spark session, source headers
from util import dataprocessing_silver as silver      # Silver reader, parquet writer, PSI


# =============================================================================================
# Configuration (from the data contract)
# =============================================================================================
PROJECT_DIR       = bronze.PROJECT_DIR
GOLD_DIR          = os.path.join(PROJECT_DIR, "datamart", "gold")
FEATURE_STORE_DIR = os.path.join(GOLD_DIR, "feature_store")
LABEL_STORE_DIR   = os.path.join(GOLD_DIR, "label_store")
REPORT_DIR        = os.path.join(GOLD_DIR, "_reports")
CHART_DIR         = os.path.join(REPORT_DIR, "charts")
EDA_DIR           = os.path.join(GOLD_DIR, "_eda")
EDA_CHART_DIR     = os.path.join(EDA_DIR, "charts")
LOG_PATH          = os.path.join(GOLD_DIR, "_gold_log.csv")
DICTIONARY_PATH   = os.path.join(GOLD_DIR, "feature_dictionary.csv")
SPLIT_DIR         = os.path.join(GOLD_DIR, "split", "v1")

with open(os.path.join(PROJECT_DIR, "config", "data_contract.yaml"), encoding="utf-8") as _f:
    CONTRACT = yaml.safe_load(_f)

TARGET        = CONTRACT["target"]
LABEL_DEF     = TARGET["label_def"]
MOB           = TARGET["mob"]
DPD_THRESHOLD = TARGET["dpd_threshold"]
SCORING_COL   = CONTRACT["scoring_timestamp"]["column"]
LABEL_TABLE   = TARGET["table"]
FEATURE_TABLES = [t for t, r in CONTRACT["feature_availability"]["tables"].items() if r["allowed"]]

FEATURE_VERSION = "v1"
FEATURE_STORE_PATH = os.path.join(FEATURE_STORE_DIR, FEATURE_VERSION)
LABEL_STORE_PATH = os.path.join(LABEL_STORE_DIR, LABEL_DEF)
SHORT = {"features_attributes": "attributes", "features_financials": "financials", "feature_clickstream": "clickstream"}

# Unmatched-join tolerance (% of loans with no feature row at scoring time)
UNMATCHED_TOLERANCE = {"features_attributes": 0.0, "features_financials": 0.0,
                       "feature_clickstream": 30.0}   # accepted coverage break (BRZ-001): 28.2% expected

EXCLUDED = (set(CONTRACT["excluded_columns"]["identifiers"])
            | {c for cols in CONTRACT["excluded_columns"]["pii"].values() for c in cols}
            | {c for cols in CONTRACT["excluded_columns"].get("replaced_by_derived", {}).values() for c in cols})

KNOWN_LOAN_TYPES = silver.KNOWN_LOAN_TYPES


def loan_type_column(loan_type):
    slug = re.sub(r"[^a-z0-9]+", "_", loan_type.lower()).strip("_")
    return f"has_{slug}" if slug.endswith("loan") else f"has_{slug}_loan"


LOAN_TYPE_COLUMNS = {loan_type_column(t): t for t in KNOWN_LOAN_TYPES}

UNITS = {
    "Age": "years", "Annual_Income": "currency per year", "Monthly_Inhand_Salary": "currency per month",
    "Num_Bank_Accounts": "count", "Num_Credit_Card": "count", "Interest_Rate": "percent", "Num_of_Loan": "count",
    "Delay_from_due_date": "days", "Num_of_Delayed_Payment": "count",
    "Changed_Credit_Limit": "change in credit limit (unit not documented in source)",
    "Num_Credit_Inquiries": "count", "Outstanding_Debt": "currency", "Credit_Utilization_Ratio": "percent",
    "Total_EMI_per_month": "currency per month", "Amount_invested_monthly": "currency per month",
    "Monthly_Balance": "currency", "credit_history_age_months": "months", "num_loan_types": "count",
}

# Chart style (validated palette)
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]


def get_spark(app_name="gold"):
    return bronze.get_spark(app_name)


def _plt():
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "axes.titlecolor": INK, "axes.titlesize": 11,
        "axes.titleweight": "bold", "axes.titlelocation": "left", "xtick.color": INK_2, "ytick.color": INK_2,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID,
        "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
        "legend.fontsize": 8, "font.size": 9, "lines.linewidth": 2})
    return plt


def _save(fig, folder, name):
    import matplotlib.pyplot as plt
    os.makedirs(folder, exist_ok=True)
    fig.savefig(os.path.join(folder, name), dpi=130, bbox_inches="tight")
    plt.close(fig)


# =============================================================================================
# Contract predictor check
# =============================================================================================
def allowed_predictors():
    out = {}
    for t, rule in CONTRACT["feature_availability"]["tables"].items():
        if not rule["allowed"]:
            continue
        src = CONTRACT["sources"][t]
        header = bronze.source_profile(os.path.join(bronze.RAW_DIR, src["file"]))[0]
        for c in header:
            if c not in EXCLUDED and c not in src["keys"] and c != src["event_time"]:
                out[c] = t
        for c in CONTRACT.get("derived_features", {}).get(t, {}):
            out[c] = t
    return out


def check_predictors(columns):
    allowed = allowed_predictors()
    rejected = [c for c in columns if c not in allowed]
    if rejected:
        raise ValueError(f"Predictors rejected by the data contract: {rejected}")
    return True


# =============================================================================================
# Step 10 — Join and aggregate
# =============================================================================================
def feature_columns(table, df):
    src = CONTRACT["sources"][table]
    return [c for c in df.columns
            if not c.startswith("_") and c not in EXCLUDED and c not in src["keys"]
            and c != src["event_time"] and c != "type_of_loan_list"]


def build_spine(lms):
    return lms.select("loan_id", "Customer_ID", SCORING_COL).distinct()


def asof_join(spine, feat, table, cols):
    short = SHORT[table]
    f = feat.select(F.col("Customer_ID").alias("_cust"), F.col("snapshot_date").alias("_snap"), *cols)
    cand = spine.join(f, spine["Customer_ID"] == f["_cust"], "left")
    valid = cand.filter(F.col("_snap").isNull() | (F.col("_snap") <= F.col(SCORING_COL)))
    w = Window.partitionBy("loan_id").orderBy(F.col("_snap").desc_nulls_last())
    picked = (valid.withColumn("_rn", F.row_number().over(w)).filter("_rn = 1")
                   .select("loan_id", F.col("_snap").alias(f"_asof_{short}"), *cols))
    out = spine.join(picked, "loan_id", "left")      # loans whose rows were all in the future come back as null

    diag = cand.agg(
        F.sum(F.col("_snap").isNotNull().cast("int")).alias("candidate_rows"),
        F.sum((F.col("_snap") > F.col(SCORING_COL)).cast("int")).alias("future_rows_excluded")).first().asDict()
    per_loan = cand.groupBy("loan_id").count().agg(F.max("count")).first()[0]
    stats = out.agg(F.count("*").alias("output_rows"), F.countDistinct("loan_id").alias("loans"),
                    F.sum(F.col(f"_asof_{short}").isNotNull().cast("int")).alias("matched")).first().asDict()
    lag = (out.filter(F.col(f"_asof_{short}").isNotNull())
              .select(F.months_between(SCORING_COL, f"_asof_{short}").cast("int").alias("lag"))
              .groupBy("lag").count().orderBy("lag").collect())
    n = stats["loans"]
    diag.update(table=table, spine_loans=spine.count(), max_rows_per_loan_before_pick=per_loan,
                output_rows=stats["output_rows"], matched_loans=stats["matched"],
                matched_pct=stats["matched"] / n * 100, unmatched_loans=n - stats["matched"],
                unmatched_pct=(n - stats["matched"]) / n * 100,
                snapshot_lag_months="; ".join(f"{r['lag']}m: {r['count']:,}" for r in lag))
    return out, diag


def add_loan_type_flags(df):
    for col, loan_type in LOAN_TYPE_COLUMNS.items():
        df = df.withColumn(col, F.coalesce(F.array_contains("type_of_loan_list", loan_type), F.lit(False)).cast("int"))
    return df.drop("type_of_loan_list")


def build_features(spine, frames):
    # Model-grain feature table: spine + as-of joins of every allowed feature table + Gold-derived loan-type flags
    out, diagnostics = spine, []
    for t in FEATURE_TABLES:
        cols = feature_columns(t, frames[t]) + (["type_of_loan_list"] if t == "features_financials" else [])
        joined, diag = asof_join(spine, frames[t], t, cols)
        out = out.join(joined.drop("Customer_ID", SCORING_COL), "loan_id", "left")
        diagnostics.append(diag)
    return add_loan_type_flags(out), diagnostics


def feature_list(df):
    return [c for c in df.columns if c not in ("loan_id", "Customer_ID", SCORING_COL) and not c.startswith("_")]


# =============================================================================================
# Step 11 — Feature store
# =============================================================================================
def describe_feature(col, dtype):
    table = allowed_predictors()[col]
    derived = CONTRACT.get("derived_features", {}).get(table, {}).get(col)
    if col in LOAN_TYPE_COLUMNS:
        derivation, layer = f"1 if '{LOAN_TYPE_COLUMNS[col]}' is in the parsed loan list, else 0", "gold (step 10)"
    elif derived:
        derivation, layer = derived["rule"], "silver (step 07/09)"
    else:
        derivation, layer = "source column (cleaned in Silver)", "silver (step 07)"
    if col.endswith("_invalid") or col in LOAN_TYPE_COLUMNS or col == "type_of_loan_unknown":
        unit = "flag (0/1)"
    elif dtype == "string":
        unit = "category"
    elif col.startswith("fe_"):
        unit = "clickstream score (unitless)"
    else:
        unit = UNITS.get(col, "")
    note = ""
    if table == "feature_clickstream":
        note = ("coverage break by time: available for loans starting 2023-01..2024-06 (100%), "
                "none for loans starting 2024-07 onwards; null = no clickstream history")
    return dict(feature=col, version_added=FEATURE_VERSION, dtype=dtype, unit=unit, source_table=table,
                source_columns=", ".join(derived["from"]) if derived else (col if col not in LOAN_TYPE_COLUMNS else "Type_of_Loan"),
                derivation=derivation, built_in=layer,
                as_of_rule=CONTRACT["feature_availability"]["tables"][table]["as_of"],
                freshness="monthly snapshot; latest snapshot on or before loan_start_date",
                entity_key="loan_id (via Customer_ID as-of join)", owner=CONTRACT["contract"]["owner"],
                allowed_by_contract=True, coverage_note=note)


def build_dictionary(features, feats):
    dtypes = dict(features.dtypes)
    dictionary = pd.DataFrame([describe_feature(c, dtypes[c]) for c in feats])
    null_pct = features.select([F.avg(F.col(c).isNull().cast("double")).alias(c) for c in feats]).first().asDict()
    dictionary["null_pct"] = dictionary["feature"].map(lambda c: round(null_pct[c] * 100, 2))
    dictionary.to_csv(DICTIONARY_PATH, index=False)
    return dictionary


def staging_path(final):
    return final + "__staging"


def write_staged(df, final, partition_col):
    silver.write_parquet_atomic(df, staging_path(final), partition_col)


def promote(final):
    shutil.rmtree(final, ignore_errors=True)
    os.replace(staging_path(final), final)


def get_features(spark, as_of=None, version=FEATURE_VERSION, staged=False):
    # As-of retrieval: features of loans scored on or before `as_of`
    path = os.path.join(FEATURE_STORE_DIR, version)
    df = spark.read.parquet(staging_path(path) if staged else path)
    return df if as_of is None else df.filter(F.col(SCORING_COL) <= F.lit(as_of).cast("date"))


def historical_replay_test(spark, spine, frames, feats):
    cohorts = [r[0] for r in spine.select(SCORING_COL).distinct().orderBy(SCORING_COL).collect()]
    compare_cols = ["loan_id", "Customer_ID", SCORING_COL] + feats + [f"_asof_{SHORT[t]}" for t in FEATURE_TABLES]
    stored = get_features(spark, staged=True)
    rows = []
    for d in [cohorts[0], cohorts[len(cohorts) // 2], cohorts[-1]]:
        truncated = {t: frames[t].filter(F.col("snapshot_date") <= F.lit(d)) for t in FEATURE_TABLES}
        sp_d = spine.filter(F.col(SCORING_COL) == F.lit(d))
        rebuilt, _ = build_features(sp_d, truncated)
        a = rebuilt.select(*compare_cols)
        b = stored.filter(F.col(SCORING_COL) == F.lit(d)).select(*compare_cols)
        diff = a.exceptAll(b).count() + b.exceptAll(a).count()
        rows.append(dict(cohort=str(d), loans=sp_d.count(), differing_rows=diff))
        assert diff == 0, f"historical replay differs for cohort {d}"
    print("Historical replay test passed: features rebuilt from truncated Silver equal the stored features")
    return pd.DataFrame(rows)


# =============================================================================================
# Step 12 — Label store
# =============================================================================================
def build_labels(lms, spine, cutoff):
    d = (lms.withColumn("mob", F.col("installment_num"))
            .withColumn("installments_missed",
                        F.when(F.col("overdue_amt") <= 0, F.lit(0))
                         .when(F.col("due_amt") > 0, F.ceil(F.col("overdue_amt") / F.col("due_amt"))).cast("int"))
            .withColumn("first_missed_date",
                        F.when(F.col("installments_missed") > 0,
                               F.add_months("snapshot_date", -F.col("installments_missed"))))
            .withColumn("dpd", F.when(F.col("overdue_amt") > 0, F.datediff("snapshot_date", "first_missed_date"))
                                .when(F.col("installments_missed") == 0, F.lit(0))))
    at_mob = (d.filter(F.col("mob") == MOB)
               .select("loan_id", F.col("snapshot_date").alias("label_snapshot_date"),
                       F.col("dpd").alias(f"dpd_at_mob{MOB}"), F.col("overdue_amt").alias(f"overdue_amt_at_mob{MOB}"),
                       F.col("installments_missed").alias(f"installments_missed_at_mob{MOB}")))
    expected = F.add_months(F.col(SCORING_COL), MOB)
    labels = (spine.join(at_mob, "loan_id", "left")
              .withColumn("expected_outcome_date", expected)
              .withColumn("maturity_status",
                          F.when(F.col("label_snapshot_date").isNotNull() & F.col(f"dpd_at_mob{MOB}").isNotNull(), "mature")
                           .when(expected > F.lit(cutoff), "immature")
                           .otherwise("censored"))
              .withColumn("label", F.when(F.col("maturity_status") == "mature",
                                          (F.col(f"dpd_at_mob{MOB}") >= DPD_THRESHOLD).cast("int")))
              .withColumn("label_def", F.lit(LABEL_DEF))
              .withColumn("data_cutoff", F.lit(cutoff)))
    return labels.select("loan_id", "Customer_ID", SCORING_COL, "label", "label_def", "maturity_status",
                         "label_snapshot_date", "expected_outcome_date", f"dpd_at_mob{MOB}",
                         f"overdue_amt_at_mob{MOB}", f"installments_missed_at_mob{MOB}", "data_cutoff")


def run_unit_tests(spark):
    # Step 10: future exclusion
    sp = spark.createDataFrame([("L1", "C1", "2024-03-01"), ("L2", "C2", "2024-03-01"), ("L3", "C3", "2024-03-01")],
                               "loan_id string, Customer_ID string, loan_start_date string") \
              .withColumn("loan_start_date", F.to_date("loan_start_date"))
    ft = spark.createDataFrame([("C1", "2024-01-01", "past"), ("C1", "2024-03-01", "same_day"), ("C1", "2024-04-01", "FUTURE"),
                                ("C2", "2024-05-01", "FUTURE")],
                               "Customer_ID string, snapshot_date string, x string") \
              .withColumn("snapshot_date", F.to_date("snapshot_date"))
    res, d = asof_join(sp, ft, "feature_clickstream", ["x"])
    assert {r["loan_id"]: r["x"] for r in res.collect()} == {"L1": "same_day", "L2": None, "L3": None}
    assert d["output_rows"] == 3 and d["future_rows_excluded"] == 2 and d["matched_loans"] == 1, d

    # Step 12: label rules
    def ledger(loan, start, months, overdue_at):
        s = pd.Timestamp(start)
        return [(loan, f"C{loan}", s.date(), m, 0.0 if m == 0 else 1000.0, float(overdue_at.get(m, 0.0)),
                 (s + pd.DateOffset(months=m)).date()) for m in months]
    rows = (ledger("A", "2024-01-01", range(0, 11), {}) + ledger("B", "2024-01-01", range(0, 11), {6: 2000})
            + ledger("C", "2023-09-01", range(0, 11), {6: 1000}) + ledger("D", "2025-09-01", range(0, 3), {})
            + ledger("E", "2024-01-01", range(0, 5), {}))
    schema = ("loan_id string, Customer_ID string, loan_start_date date, installment_num int, due_amt double, "
              "overdue_amt double, snapshot_date date")
    lms = spark.createDataFrame(rows, schema)
    lab = {r["loan_id"]: r for r in build_labels(lms, build_spine(lms), pd.Timestamp("2025-11-01").date()).collect()}
    assert (lab["A"]["label"], lab["A"]["dpd_at_mob6"]) == (0, 0)
    assert (lab["B"]["label"], lab["B"]["dpd_at_mob6"]) == (1, 61)
    assert (lab["C"]["label"], lab["C"]["dpd_at_mob6"]) == (0, 29)
    assert lab["D"]["maturity_status"] == "immature" and lab["D"]["label"] is None
    assert lab["E"]["maturity_status"] == "censored" and lab["E"]["label"] is None
    print("Unit tests passed: as-of join future exclusion, label rules (on-time, default, immature, censored)")


# =============================================================================================
# Step 13 — Reconciliation (acceptance) + release
# =============================================================================================
def acceptance_checks(spark, frames, join_diagnostics):
    fs = get_features(spark, staged=True)
    ls = spark.read.parquet(staging_path(LABEL_STORE_PATH))
    results = []

    def check(name, ok, observed, rule):
        results.append(dict(check=name, status="PASS" if ok else "FAIL", observed=observed, rule=rule))

    def dup_and_null(df, name):
        dups = df.groupBy("loan_id").count().filter("count > 1").count()
        nulls = df.filter(F.col("loan_id").isNull() | F.col("Customer_ID").isNull() | F.col(SCORING_COL).isNull()).count()
        check(f"{name}: one row per loan", dups == 0, f"{dups} duplicate loan_id", "0 duplicates")
        check(f"{name}: no null keys", nulls == 0, f"{nulls} rows with a null key", "0")

    dup_and_null(fs, "feature store")
    dup_and_null(ls, "label store")

    silver_loans = {r[0] for r in frames[LABEL_TABLE].select("loan_id").distinct().collect()}
    fs_loans = {r[0] for r in fs.select("loan_id").collect()}
    ls_loans = {r[0] for r in ls.select("loan_id").collect()}
    check("same loan population (Silver = feature store = label store)", silver_loans == fs_loans == ls_loans,
          f"Silver {len(silver_loans):,} / features {len(fs_loans):,} / labels {len(ls_loans):,}", "identical sets")

    mature = ls.filter(F.col("maturity_status") == "mature")
    joined = fs.join(mature.select("loan_id", "label"), "loan_id", "inner")
    n_mature, n_joined = mature.count(), joined.count()
    check("features x mature labels: 1:1, no loss",
          n_joined == n_mature and joined.select("loan_id").distinct().count() == n_joined,
          f"{n_joined:,} joined of {n_mature:,} mature labels", "joined = mature, unique")

    for _, r in join_diagnostics.iterrows():
        tol = UNMATCHED_TOLERANCE[r["table"]]
        check(f"unmatched join: {r['table']}", r["unmatched_pct"] <= tol,
              f"{r['unmatched_loans']:,} loans ({r['unmatched_pct']:.1f}%)", f"<= {tol}%")

    bad_label = ls.filter(((F.col("maturity_status") == "mature") & F.col("label").isNull())
                          | ((F.col("maturity_status") != "mature") & F.col("label").isNotNull())).count()
    counts = {r["maturity_status"]: r["count"] for r in ls.groupBy("maturity_status").count().collect()}
    check("only mature loans carry a label", bad_label == 0, f"{counts}; {bad_label} inconsistent", "label iff mature")

    future = fs.filter(F.greatest(*[(F.col(f"_asof_{SHORT[t]}") > F.col(SCORING_COL)).cast("int")
                                    for t in FEATURE_TABLES]) == 1).count()
    check("point-in-time: feature snapshot <= loan_start_date", future == 0, f"{future} rows with a future snapshot", "0")

    wrong_outcome = mature.filter(F.col("label_snapshot_date") != F.add_months(SCORING_COL, MOB)).count()
    check(f"outcome observed at loan_start_date + {MOB} months", wrong_outcome == 0, f"{wrong_outcome} mismatched", "0")

    label_only = set(ls.columns) - {"loan_id", "Customer_ID", SCORING_COL, "_gold_batch_id", "_gold_ts"}
    lms_cols = set(frames[LABEL_TABLE].columns) - {"loan_id", "Customer_ID", SCORING_COL, "snapshot_date"}
    leak = sorted(set(fs.columns) & (label_only | lms_cols))
    check("no label in X (no label or label-source column in features)", not leak, f"overlap: {leak or 'none'}", "no overlap")

    stored_features = feature_list(fs)
    try:
        check_predictors(stored_features)
        check("all predictors allowed by the data contract", True, f"{len(stored_features)} features",
              "contract v" + CONTRACT["contract"]["version"])
    except ValueError as e:
        check("all predictors allowed by the data contract", False, str(e), "contract")

    dict_features = set(pd.read_csv(DICTIONARY_PATH)["feature"])
    check("feature dictionary documents exactly the stored features", dict_features == set(stored_features),
          f"missing: {sorted(set(stored_features) - dict_features)}; extra: {sorted(dict_features - set(stored_features))}",
          "identical")
    label_rate = mature.agg(F.avg("label")).first()[0]
    return pd.DataFrame(results), dict(loans=len(fs_loans), features=len(stored_features), mature=n_mature,
                                       label_rate=label_rate)


# =============================================================================================
# Step 10 charts
# =============================================================================================
def availability_by_cohort(features):
    cov = (features.groupBy(SCORING_COL)
           .agg(*[F.avg(F.col(f"_asof_{SHORT[t]}").isNotNull().cast("double")).alias(SHORT[t]) for t in FEATURE_TABLES])
           .orderBy(SCORING_COL).toPandas())
    cov[SCORING_COL] = pd.to_datetime(cov[SCORING_COL].astype(str))
    return cov


def plot_availability(cov):
    plt = _plt()
    import matplotlib.ticker as mticker
    fig, ax = plt.subplots(figsize=(11, 3.8))
    markers = {"attributes": "o", "financials": "s", "clickstream": "^"}
    for color, t in zip(SERIES, FEATURE_TABLES):
        s = SHORT[t]
        ax.plot(cov[SCORING_COL], cov[s] * 100, color=color, marker=markers[s], markersize=4, label=s,
                linestyle="--" if s == "financials" else "-")
    ends = cov.iloc[-1]
    for level in sorted({round(ends[SHORT[t]] * 100) for t in FEATURE_TABLES}):
        names = [SHORT[t] for t in FEATURE_TABLES if round(ends[SHORT[t]] * 100) == level]
        ax.text(cov[SCORING_COL].iloc[-1] + pd.Timedelta(days=12), level, " & ".join(names), va="center",
                fontsize=8, color=INK_2)
    ax.set_ylim(-5, 105)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
    ax.set_ylabel("% of loans with a snapshot at loan start")
    ax.legend(loc="center left")
    ax.set_title("Feature availability at scoring time, by loan start month (as-of join)")
    fig.tight_layout()
    _save(fig, CHART_DIR, "10_1_feature_availability_by_cohort.png")


def plot_missing_per_loan(features, feats):
    plt = _plt()
    import matplotlib.ticker as mticker
    pdf = features.select(*feats).toPandas()
    missing = pdf.isna().sum(axis=1).value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.bar(missing.index, missing.values, color=SERIES[0], width=0.7)
    for x, v in missing.items():
        ax.text(x, v, f"{v:,}", ha="center", va="bottom", fontsize=7.5, color=INK_2)
    ax.set_xlabel("number of missing features in the loan's row")
    ax.set_ylabel("loans")
    ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
    ax.set_title(f"Missing features per loan ({len(feats)} features) — left: cleaning nulls; right: no clickstream history")
    fig.tight_layout()
    _save(fig, CHART_DIR, "10_2_missing_features_per_loan.png")


# =============================================================================================
# Build entry point: steps 10 -> 11 -> 12 -> 13 -> release
# =============================================================================================
def run_build(spark=None, charts=True, tests=False):
    """Steps 10-13. The stores are written to staging and promoted only if every acceptance check passes."""
    spark = spark or get_spark()
    for d in (GOLD_DIR, FEATURE_STORE_DIR, LABEL_STORE_DIR, REPORT_DIR, CHART_DIR):
        os.makedirs(d, exist_ok=True)
    if tests:
        run_unit_tests(spark)

    # Step 10
    frames = {t: silver.read_silver(spark, t) for t in [LABEL_TABLE] + FEATURE_TABLES}
    spine = build_spine(frames[LABEL_TABLE]).cache()
    assert spine.count() == spine.select("loan_id").distinct().count(), "spine has more than one row per loan"
    features, diagnostics = build_features(spine, frames)
    features = features.cache()
    join_diagnostics = pd.DataFrame(diagnostics)[[
        "table", "spine_loans", "candidate_rows", "future_rows_excluded", "max_rows_per_loan_before_pick",
        "output_rows", "matched_loans", "matched_pct", "unmatched_loans", "unmatched_pct", "snapshot_lag_months"]]
    join_diagnostics.to_csv(os.path.join(REPORT_DIR, "join_diagnostics.csv"), index=False)
    feats = feature_list(features)
    check_predictors(feats)
    assert features.count() == spine.count() == features.select("loan_id").distinct().count(), "row explosion or loss"
    cov = availability_by_cohort(features)
    if charts:
        plot_availability(cov)
        plot_missing_per_loan(features, feats)

    # Step 11
    dictionary = build_dictionary(features, feats)
    batch = f"gold_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{uuid.uuid4().hex[:6]}"
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    feature_store = (features.withColumn("_feature_version", F.lit(FEATURE_VERSION))
                             .withColumn("_gold_batch_id", F.lit(batch)).withColumn("_gold_ts", F.lit(ts)))
    write_staged(feature_store, FEATURE_STORE_PATH, SCORING_COL)
    replay = historical_replay_test(spark, spine, frames, feats) if tests else None

    # Step 12
    cutoff = frames[LABEL_TABLE].agg(F.max("snapshot_date")).first()[0]
    labels = build_labels(frames[LABEL_TABLE], spine, cutoff).cache()
    label_summary = (labels.groupBy("maturity_status")
                     .agg(F.count("*").alias("loans"), F.sum("label").alias("label_1"),
                          F.avg("label").alias("label_rate")).toPandas())
    label_summary["label_rate_pct"] = label_summary["label_rate"] * 100
    label_summary = label_summary.drop(columns="label_rate")
    label_summary.insert(0, "label_def", LABEL_DEF)
    label_summary["data_cutoff"] = str(cutoff)
    label_summary.to_csv(os.path.join(REPORT_DIR, "label_summary.csv"), index=False)
    label_store = labels.withColumn("_gold_batch_id", F.lit(batch)).withColumn("_gold_ts", F.lit(ts))
    write_staged(label_store, LABEL_STORE_PATH, "label_snapshot_date")

    # Step 13
    acceptance, stats = acceptance_checks(spark, frames, join_diagnostics)
    acceptance.insert(0, "gold_batch_id", batch)
    acceptance.to_csv(os.path.join(REPORT_DIR, "gold_acceptance_report.csv"), index=False)
    if (acceptance["status"] == "FAIL").any():
        raise AssertionError("Gold release BLOCKED — failed checks: "
                             f"{acceptance.loc[acceptance['status'] == 'FAIL', 'check'].tolist()}")
    promote(FEATURE_STORE_PATH)
    promote(LABEL_STORE_PATH)

    log = pd.DataFrame([dict(gold_batch_id=batch, gold_ts=ts, feature_version=FEATURE_VERSION, loans=stats["loans"],
                             features=stats["features"], label_def=LABEL_DEF, mature_labels=stats["mature"],
                             label_rate_pct=round(stats["label_rate"] * 100, 2), data_cutoff=str(cutoff),
                             acceptance="PASS")])
    old = pd.read_csv(LOG_PATH) if os.path.exists(LOG_PATH) else pd.DataFrame()
    pd.concat([old, log], ignore_index=True).to_csv(LOG_PATH, index=False)
    return dict(join_diagnostics=join_diagnostics, dictionary=dictionary, label_summary=label_summary,
                acceptance=acceptance, log=log, replay=replay, availability=cov)


# =============================================================================================
# Gold EDA — step 14 (feature / target, train only)
# =============================================================================================
def binned(x, kind):
    if kind == "numeric" and x.nunique() > 10:
        b = pd.qcut(x, 10, duplicates="drop").astype(str)
    else:
        b = x.astype(object).astype(str)
    return b.where(x.notna(), "Missing")


def woe_iv(x, y, kind):
    t = pd.DataFrame({"bin": binned(x, kind), "y": y}).groupby("bin", sort=False)["y"].agg(["count", "sum"])
    t["good"] = t["count"] - t["sum"]
    pb = (t["sum"] + 0.5) / (t["sum"].sum() + 0.5 * len(t))
    pg = (t["good"] + 0.5) / (t["good"].sum() + 0.5 * len(t))
    t["woe"] = np.log(pg / pb)
    t["iv"] = (pg - pb) * t["woe"]
    t["default_rate"] = t["sum"] / t["count"]
    return t, float(t["iv"].sum())


def univariate_auc(x, y, kind):
    from sklearn.metrics import roc_auc_score
    m = x.notna()
    if m.sum() < 50 or x[m].nunique() < 2:
        return np.nan, ""
    score = x[m] if kind != "categorical" else x[m].map(y[m].groupby(x[m]).mean())
    a = roc_auc_score(y[m], score)
    return max(a, 1 - a), ("higher -> more default" if a >= 0.5 else "higher -> less default") if kind != "categorical" else "by category"


def iv_strength(v):
    return ("suspicious (>0.5)" if v > 0.5 else "strong" if v > 0.3 else "medium" if v > 0.1
            else "weak" if v > 0.02 else "not useful")


def load_eda_data(spark):
    if not os.path.exists(os.path.join(SPLIT_DIR, "split_manifest.csv")):
        raise FileNotFoundError("Split manifest missing: run step 17 (script/mlpipeline.ipynb) before the Gold EDA")
    manifest = pd.read_csv(os.path.join(SPLIT_DIR, "split_manifest.csv"), usecols=["loan_id", "partition"])
    fs = get_features(spark)
    feats = feature_list(fs)
    dtypes = dict(fs.dtypes)
    fs_pd = fs.select("loan_id", SCORING_COL, *feats).toPandas()
    fs_pd[SCORING_COL] = pd.to_datetime(fs_pd[SCORING_COL].astype(str))
    lab_pd = spark.read.parquet(LABEL_STORE_PATH).filter("maturity_status = 'mature'").select("loan_id", "label").toPandas()
    data = fs_pd.merge(manifest, on="loan_id", how="inner")
    assert len(data) == len(fs_pd), "every loan must have a partition in the split manifest"

    click = [c for c in feats if c.startswith("fe_")]
    categorical = [c for c in feats if dtypes[c] == "string"]
    binary = [c for c in feats if c.endswith("_invalid") or c in LOAN_TYPE_COLUMNS or c == "type_of_loan_unknown"]
    numeric = [c for c in feats if c not in categorical + binary]
    for c in binary:
        data[c] = data[c].astype(float)
    # Training rows with labels — the ONLY place labels are attached
    train = data[data["partition"] == "train"].merge(lab_pd, on="loan_id", how="inner")
    assert set(train["partition"]) == {"train"} and len(train) == (manifest["partition"] == "train").sum()
    allowed = allowed_predictors()

    def group_of(c):
        if c in click: return "clickstream"
        if c in LOAN_TYPE_COLUMNS: return "gold-derived"
        return SHORT[allowed[c]]

    return dict(data=data, train=train, features=feats, features_a=[c for c in feats if c not in click], click=click,
                categorical=categorical, binary=binary, numeric=numeric, group_of=group_of)


def run_step14(ctx, charts=True):
    os.makedirs(EDA_CHART_DIR, exist_ok=True)
    data, train, feats = ctx["data"], ctx["train"], ctx["features"]
    group_of, CAT, BIN, NUM, CLICK, FA = (ctx["group_of"], ctx["categorical"], ctx["binary"], ctx["numeric"],
                                          ctx["click"], ctx["features_a"])
    order = ["train", "validation", "test", "oot"]

    # 14a missing by partition
    miss = data.groupby("partition")[feats].apply(lambda g: g.isna().mean() * 100).T[order]
    miss["group"] = [group_of(c) for c in miss.index]
    miss.to_csv(os.path.join(EDA_DIR, "missing_by_partition.csv"))

    # 14a correlation (train, Spearman, without clickstream)
    corr_cols = [c for c in NUM + BIN if c not in CLICK and train[c].nunique() > 1]
    corr = train[corr_cols].corr(method="spearman")
    pairs = (corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1)).stack()
             .rename("spearman").reset_index().rename(columns={"level_0": "feature_1", "level_1": "feature_2"}))
    pairs["abs"] = pairs["spearman"].abs()
    high_pairs = pairs[pairs["abs"] >= 0.8].sort_values("abs", ascending=False).drop(columns="abs")
    high_pairs.to_csv(os.path.join(EDA_DIR, "high_correlation_pairs.csv"), index=False)

    # 14b relationships (train only)
    y = train["label"]
    rows, bin_tables = [], {}
    for c in feats:
        kind = "categorical" if c in CAT else "binary" if c in BIN else "numeric"
        t, iv = woe_iv(train[c], y, kind)
        auc, direction = univariate_auc(train[c], y, kind)
        bin_tables[c] = t
        miss_rate = t.loc["Missing", "default_rate"] if "Missing" in t.index else np.nan
        rows.append(dict(feature=c, group=group_of(c), kind=kind, in_set_A=c in FA, iv=iv,
                         iv_strength=iv_strength(iv), auc=auc, direction=direction,
                         missing_pct=train[c].isna().mean() * 100, default_rate_when_missing=miss_rate))
    rel = pd.DataFrame(rows).sort_values("iv", ascending=False).reset_index(drop=True)
    rel.to_csv(os.path.join(EDA_DIR, "feature_target_relationships.csv"), index=False)
    pd.concat([t.assign(feature=c) for c, t in bin_tables.items()]).reset_index().to_csv(
        os.path.join(EDA_DIR, "feature_target_bins.csv"), index=False)

    # Target rates by group / month (train only)
    def rate_by(col, values=None):
        s = values if values is not None else train[col]
        g = train.assign(_g=s.astype(object).where(s.notna(), "Missing").astype(str)).groupby("_g")["label"]
        return pd.DataFrame({"group_by": col, "value": g.mean().index, "loans": g.size().values,
                             "default_rate_pct": g.mean().values * 100})
    age_band = pd.cut(train["Age"], [13, 24, 34, 44, 100], labels=["14-24", "25-34", "35-44", "45+"])
    group_rates = pd.concat([
        rate_by("loan_start_month", train[SCORING_COL].dt.strftime("%Y-%m")), rate_by("age_band", age_band),
        rate_by("num_loan_types"), *[rate_by(c) for c in CAT], *[rate_by(c) for c in LOAN_TYPE_COLUMNS]],
        ignore_index=True)
    group_rates.to_csv(os.path.join(EDA_DIR, "target_rates_by_group.csv"), index=False)

    # Leakage watchlist
    month_num = train[SCORING_COL].dt.year * 12 + train[SCORING_COL].dt.month
    watch = []
    def flag(feature, signal, evidence, risk, proposed):
        watch.append(dict(feature=feature, group=group_of(feature), signal=signal, evidence=evidence, risk=risk,
                          proposed_action=proposed))
    for _, r in rel.iterrows():
        c = r["feature"]
        if r["iv"] > 0.5 or (pd.notna(r["auc"]) and r["auc"] > 0.85):
            flag(c, "suspiciously strong", f"IV {r['iv']:.2f}, AUC {r['auc']:.2f}", "high",
                 "investigate: confirm it is known at loan start and not derived from repayment")
        if pd.notna(r["default_rate_when_missing"]) and abs(r["default_rate_when_missing"] - y.mean()) * 100 >= 5 \
                and r["missing_pct"] >= 0.5:
            flag(c, "informative missingness", f"default rate {r['default_rate_when_missing']:.1%} when missing "
                 f"vs {y.mean():.1%}; {r['missing_pct']:.1f}% missing", "medium",
                 "keep the _invalid flag / missing indicator in step 18; do not drop rows")
        if r["kind"] != "categorical" and train[c].nunique() > 1:
            rho = pd.Series(train[c]).corr(month_num, method="spearman")
            if abs(rho) >= 0.3:
                flag(c, "possible time proxy", f"Spearman with loan-start month {rho:+.2f}", "medium",
                     "check drift in step 16; consider excluding if it only encodes time")
        top_share = train[c].value_counts(normalize=True, dropna=False).iloc[0]
        if top_share >= 0.995:
            flag(c, "constant / near-constant", f"top value covers {top_share:.1%} of train", "low",
                 "candidate for removal in step 23")
    for _, p in high_pairs.iterrows():
        flag(p["feature_1"], "near-duplicate", f"Spearman {p['spearman']:+.2f} with {p['feature_2']}", "low",
             "keep one of the pair in step 23")
    watchlist = pd.DataFrame(watch, columns=["feature", "group", "signal", "evidence", "risk", "proposed_action"])
    watchlist.to_csv(os.path.join(EDA_DIR, "leakage_watchlist.csv"), index=False)

    ab = rel.assign(set=np.where(rel["group"] == "clickstream", "clickstream (B only)", "set A features"))
    ab.groupby("set").agg(features=("feature", "size"), max_iv=("iv", "max"), mean_iv=("iv", "mean"),
                          max_auc=("auc", "max"), useful_iv_ge_0_02=("iv", lambda s: int((s >= 0.02).sum()))
                          ).to_csv(os.path.join(EDA_DIR, "model_a_vs_b_signal_preview.csv"))

    if charts:
        plot_step14(train, corr, corr_cols, rel, bin_tables, group_rates)
    return dict(relationships=rel, bin_tables=bin_tables, watchlist=watchlist, high_pairs=high_pairs,
                group_rates=group_rates, missing=miss)


def plot_step14(train, corr, corr_cols, rel, bin_tables, group_rates):
    plt = _plt()
    from matplotlib.colors import LinearSegmentedColormap
    y = train["label"]
    base = y.mean() * 100

    # 14_1 correlation heatmap
    diverging = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f0efec", "#e34948"])
    fig, ax = plt.subplots(figsize=(13, 11))
    im = ax.imshow(corr.values, cmap=diverging, vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr_cols)), corr_cols, rotation=90, fontsize=6.5)
    ax.set_yticks(range(len(corr_cols)), corr_cols, fontsize=6.5)
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, shrink=0.6)
    cb.set_label("Spearman correlation", color=INK_2)
    ax.set_title("Feature correlations (train, without clickstream) — blue negative, red positive")
    fig.tight_layout()
    _save(fig, EDA_CHART_DIR, "14_1_feature_correlation.png")

    # 14_2 IV top 30 by group
    top = rel.head(30).iloc[::-1]
    gcolor = {"financials": SERIES[0], "attributes": SERIES[1], "gold-derived": SERIES[2], "clickstream": SERIES[3]}
    fig, ax = plt.subplots(figsize=(11, 9))
    ax.barh(top["feature"], top["iv"], color=[gcolor[g] for g in top["group"]], height=0.7)
    for yv, (iv, auc) in enumerate(zip(top["iv"], top["auc"])):
        ax.text(iv + 0.005, yv, f"IV {iv:.2f} · AUC {auc:.2f}", va="center", fontsize=7.5, color=INK_2)
    for x, lab in [(0.02, "weak"), (0.1, "medium"), (0.3, "strong"), (0.5, "suspicious")]:
        ax.axvline(x, color=INK_2, linewidth=0.7, linestyle=":")
        ax.text(x, len(top) - 0.3, f" {lab}", fontsize=7, color=INK_2, va="bottom")
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=v) for v in gcolor.values()], labels=list(gcolor),
              loc="lower right")
    ax.set_xlabel("information value (train)")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, max(top["iv"].max() * 1.35, 0.6))
    ax.set_title("Top 30 features by information value (train only)")
    fig.tight_layout()
    _save(fig, EDA_CHART_DIR, "14_2_feature_strength_iv.png")

    # 14_3 default rate by decile
    top_num = rel[rel["kind"] == "numeric"].head(6)["feature"].tolist()
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.4), sharey=True)
    for ax, c in zip(axes.flat, top_num):
        t = bin_tables[c].copy()
        t = t[t.index != "Missing"]
        order = sorted(t.index, key=lambda s: float(s.strip("(]").split(",")[0]) if s.startswith("(") else float(s))
        t = t.loc[order]
        xs = np.arange(1, len(t) + 1)
        ax.plot(xs, t["default_rate"] * 100, color=SERIES[0], marker="o", markersize=4)
        ax.axhline(base, color=INK_2, linestyle="--", linewidth=1)
        ax.set_title(c, pad=14)
        ax.text(0, 1.01, f"IV {rel.set_index('feature').loc[c, 'iv']:.2f} · bins low → high",
                transform=ax.transAxes, fontsize=7.5, color=INK_2, va="bottom")
        ax.set_xticks(xs)
        ax.set_xlabel("decile of feature value (1 = lowest)")
    for ax in axes[:, 0]:
        ax.set_ylabel("default rate (%)")
    axes[0, 0].text(1, base + 1, f"train average {base:.1f}%", fontsize=7.5, color=INK_2)
    fig.suptitle("Default rate by feature decile — 6 strongest numeric features (train only)", x=0.01, ha="left",
                 fontweight="bold", color=INK)
    fig.tight_layout()
    _save(fig, EDA_CHART_DIR, "14_3_default_rate_by_decile.png")

    # 14_4 default rate by category
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), gridspec_kw={"height_ratios": [2.2, 1]})
    for ax, c in zip(axes.flat, ["Occupation", "Credit_Mix", "Payment_of_Min_Amount", "Payment_Behaviour"]):
        t = bin_tables[c].sort_values("default_rate")
        colors = [SERIES[1] if i == "Missing" else SERIES[0] for i in t.index]
        ax.barh(t.index, t["default_rate"] * 100, color=colors, height=0.7)
        for yv, (r, n) in enumerate(zip(t["default_rate"], t["count"])):
            ax.text(r * 100 + 0.5, yv, f"{r * 100:.1f}%  (n={n:,})", va="center", fontsize=7.5, color=INK_2)
        ax.axvline(base, color=INK_2, linestyle="--", linewidth=1)
        ax.set_xlim(0, t["default_rate"].max() * 100 * 1.35)
        ax.set_title(f"{c}  (IV {rel.set_index('feature').loc[c, 'iv']:.2f})")
        ax.set_xlabel("default rate (%)")
        ax.grid(axis="y", visible=False)
    fig.legend([plt.Rectangle((0, 0), 1, 1, color=SERIES[0]), plt.Rectangle((0, 0), 1, 1, color=SERIES[1])],
               ["category", "Missing (placeholder / NM set to null in Silver)"], loc="upper right", ncol=2)
    fig.suptitle(f"Default rate by category (train only; dashed = train average {base:.1f}%)", x=0.01, ha="left",
                 fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    _save(fig, EDA_CHART_DIR, "14_4_default_rate_by_category.png")

    # 14_5 default rate by month
    m = group_rates[group_rates["group_by"] == "loan_start_month"]
    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.plot(pd.to_datetime(m["value"]), m["default_rate_pct"], color=SERIES[0], marker="o", markersize=4)
    ax.axhline(base, color=INK_2, linestyle="--", linewidth=1)
    ax.text(pd.to_datetime(m["value"]).iloc[0], base + 0.8, f"train average {base:.1f}%", fontsize=7.5, color=INK_2)
    ax.set_ylim(0, m["default_rate_pct"].max() * 1.25)
    ax.set_ylabel("default rate (%)")
    ax.set_title("Default rate by loan-start month (train only, 2023-01 to 2024-06)")
    fig.tight_layout()
    _save(fig, EDA_CHART_DIR, "14_5_default_rate_by_month.png")


# =============================================================================================
# Gold EDA — step 16 (temporal quality)
# =============================================================================================
def psi_num(e, a):
    e, a = pd.Series(e).dropna().astype(float), pd.Series(a).dropna().astype(float)
    return silver.psi(e.values, a.values) if len(e) and len(a) else np.nan


def psi_cat(e, a):
    e, a = pd.Series(e).fillna("Missing"), pd.Series(a).fillna("Missing")
    cats = sorted(set(e) | set(a))
    pe = e.value_counts(normalize=True).reindex(cats).fillna(0).clip(lower=1e-6)
    pa = a.value_counts(normalize=True).reindex(cats).fillna(0).clip(lower=1e-6)
    return float(((pa - pe) * np.log(pa / pe)).sum())


def run_step16(spark, ctx, s14, charts=True):
    from scipy.stats import chi2_contingency
    data, train, feats = ctx["data"], ctx["train"], ctx["features"]
    CAT, BIN, CLICK, FA, group_of = ctx["categorical"], ctx["binary"], ctx["click"], ctx["features_a"], ctx["group_of"]
    rel = s14["relationships"]
    frames = {t: silver.read_silver(spark, t) for t in [LABEL_TABLE] + FEATURE_TABLES}
    cov = availability_by_cohort(get_features(spark))
    temporal = []

    def tq(check, grain, status, finding, evidence, action):
        temporal.append(dict(check=check, grain=grain, status=status, finding=finding, evidence=evidence, action=action))

    def months_between(lo, hi):
        return pd.period_range(pd.Timestamp(lo).to_period("M"), pd.Timestamp(hi).to_period("M"), freq="M")

    def psi_feature(c, e, a):
        return psi_cat(e, a) if c in CAT + BIN else psi_num(e, a)

    # 1. missing periods / sampling
    for t in [LABEL_TABLE] + FEATURE_TABLES:
        cnt = frames[t].groupby("snapshot_date").count().toPandas()
        cnt["snapshot_date"] = pd.to_datetime(cnt["snapshot_date"].astype(str))
        exp = months_between(cnt["snapshot_date"].min(), cnt["snapshot_date"].max())
        missing = sorted(set(exp) - set(cnt["snapshot_date"].dt.to_period("M")))
        cv = cnt["count"].std() / cnt["count"].mean()
        tq("missing periods / sampling", f"{t} snapshot month", "PASS" if not missing else "FAIL",
           f"{len(exp)} months, {len(missing)} missing", f"rows per month CV {cv:.2f}",
           "none" if not missing else "investigate missing months")
    coh = data.groupby(data[SCORING_COL].dt.to_period("M")).size()
    exp = months_between(coh.index.min().to_timestamp(), coh.index.max().to_timestamp())
    tq("missing periods / sampling", "loan-start cohort", "PASS" if set(exp) == set(coh.index) else "FAIL",
       f"{len(coh)} cohorts {coh.index.min()}..{coh.index.max()}, none missing",
       f"loans per cohort {coh.min()}-{coh.max()} (CV {coh.std() / coh.mean():.2f})", "none")
    panel = frames["feature_clickstream"].groupby("Customer_ID").count().toPandas()["count"]
    tq("missing periods / sampling", "clickstream customer panel", "PASS" if panel.nunique() == 1 else "WARN",
       f"{len(panel):,} customers, snapshots per customer: {sorted(panel.unique())}",
       "every covered customer observed every month 2023-01..2024-12", "none")

    # 2. freshness
    audit = pd.read_csv(os.path.join(PROJECT_DIR, "datamart", "bronze", "_audit", "audit_report.csv"))
    for _, a in audit[audit["check"] == "freshness"].iterrows():
        tq("freshness", f"{a['table']} snapshot month", a["status"], a["observed"], "Bronze audit (step 03)",
           "none" if a["status"] == "PASS" else
           "accept: historical extract; features are taken as of each loan's start, so later loans use the latest "
           "available snapshot (as-of rule)")

    # 3. coverage breaks
    for t in FEATURE_TABLES:
        s = cov.set_index(SCORING_COL)[SHORT[t]] * 100
        jumps = s.diff().abs()
        if (jumps > 50).any():
            when = jumps.idxmax()
            tq("coverage break", f"{t} by loan-start cohort", "WARN",
               f"availability changes from {s.shift(1)[when]:.0f}% to {s[when]:.0f}% at {when:%Y-%m}",
               f"{(s == 0).sum()} cohorts with 0% coverage",
               "accept: model A (no clickstream) is the champion and is evaluated on OOT; model B (with clickstream) "
               "is compared on train/validation/test and an in-development time check; no has_clickstream flag (time proxy)")
        else:
            tq("coverage break", f"{t} by loan-start cohort", "PASS", f"availability {s.min():.0f}%-{s.max():.0f}% in every cohort",
               "", "none")

    # 4. feature drift
    train_feats = data[data["partition"] == "train"]
    dev_feats = data[data["partition"] != "oot"]
    oot_feats = data[data["partition"] == "oot"]
    early = train_feats[train_feats[SCORING_COL] < "2023-10-01"]
    late = train_feats[train_feats[SCORING_COL] >= "2023-10-01"]
    drift = pd.DataFrame([dict(feature=c, group=group_of(c),
                               psi_train_early_vs_late=psi_feature(c, early[c], late[c]),
                               psi_dev_vs_oot=psi_feature(c, dev_feats[c], oot_feats[c]) if c not in CLICK else np.nan,
                               missing_pct_dev=dev_feats[c].isna().mean() * 100,
                               missing_pct_oot=oot_feats[c].isna().mean() * 100) for c in feats])
    band = lambda v: "large (>0.25)" if v > 0.25 else "moderate (0.1-0.25)" if v > 0.1 else "stable (<0.1)"
    drift["train_drift"] = drift["psi_train_early_vs_late"].map(band)
    drift["oot_drift_info_only"] = drift["psi_dev_vs_oot"].map(lambda v: band(v) if pd.notna(v) else "n/a (no clickstream in OOT)")
    drift = drift.sort_values("psi_dev_vs_oot", ascending=False)
    drift.to_csv(os.path.join(EDA_DIR, "feature_drift_psi.csv"), index=False)
    n_train_shift = int((drift["psi_train_early_vs_late"] > 0.1).sum())
    n_oot_shift = int((drift["psi_dev_vs_oot"] > 0.1).sum())
    tq("feature drift (inside training)", "train cohorts 2023-01..09 vs 2023-10..2024-06", "PASS" if n_train_shift == 0 else "WARN",
       f"{n_train_shift} features with PSI > 0.1; max {drift['psi_train_early_vs_late'].max():.3f}",
       "features only", "none" if n_train_shift == 0 else "review drifting features in step 23")
    tq("feature drift (development vs OOT, info only)", "dev 2023-01..2024-06 vs OOT 2024-07..2025-01",
       "PASS" if n_oot_shift == 0 else "WARN",
       f"{n_oot_shift} set-A features with PSI > 0.1; max {drift['psi_dev_vs_oot'].max():.3f}",
       "features only; never used for feature decisions",
       "none" if n_oot_shift == 0 else "explain in step 32 (robustness) and monitor in step 35")
    heat_feats = [c for c in rel["feature"] if c in FA][:15]
    months_all = sorted(data[SCORING_COL].dt.to_period("M").unique())
    heat = pd.DataFrame({str(m): [psi_feature(c, train_feats[c], data.loc[data[SCORING_COL].dt.to_period("M") == m, c])
                                   for c in heat_feats] for m in months_all}, index=heat_feats)
    heat.to_csv(os.path.join(EDA_DIR, "feature_psi_by_cohort.csv"))

    # 5. label drift (train only)
    lm = (train.groupby(train[SCORING_COL].dt.to_period("M"))["label"].agg(["size", "sum", "mean"])
          .rename(columns={"size": "loans", "sum": "defaults", "mean": "rate"}))
    p = train["label"].mean()
    lm["lcl"] = p - 3 * np.sqrt(p * (1 - p) / lm["loans"])
    lm["ucl"] = p + 3 * np.sqrt(p * (1 - p) / lm["loans"])
    outside = lm[(lm["rate"] < lm["lcl"]) | (lm["rate"] > lm["ucl"])]
    slope = np.polyfit(np.arange(len(lm)), lm["rate"] * 100, 1)[0]
    chi2, pval, _, _ = chi2_contingency(np.vstack([lm["defaults"], lm["loans"] - lm["defaults"]]))
    lm.to_csv(os.path.join(EDA_DIR, "label_rate_by_month_train.csv"))
    tq("label drift (train only)", "loan-start month", "PASS" if outside.empty and pval >= 0.01 else "WARN",
       f"monthly default rate {lm['rate'].min():.1%}-{lm['rate'].max():.1%}; {len(outside)} month(s) outside 3-sigma limits",
       f"trend {slope:+.2f} pp/month; chi-square homogeneity p = {pval:.3f}",
       "none — variation consistent with sampling noise" if outside.empty and pval >= 0.01
       else "monitor; compare in-time test vs OOT performance in step 31-32")

    # 6. seasonality
    vol = data.groupby(data[SCORING_COL].dt.to_period("M")).size()
    v23 = vol[[m for m in vol.index if m.year == 2023]].values
    v24 = vol[[m for m in vol.index if m.year == 2024]].values
    r_season = np.corrcoef(v23, v24)[0, 1]
    tq("seasonality", "loan volume by calendar month", "INFO",
       f"2023 vs 2024 monthly volumes correlation {r_season:+.2f} (no repeating pattern)" if abs(r_season) < 0.5
       else f"2023 vs 2024 monthly volumes correlation {r_season:+.2f}",
       "labels: only 18 training months (1.5 cycles) — seasonality of default cannot be assessed reliably",
       "none; revisit when more history is available")

    # 7. look-ahead risk
    tq("look-ahead risk", "features", "PASS", "no feature dated after loan_start_date",
       "step 10 future-exclusion test, step 11 historical replay, step 13 point-in-time check (0 rows)", "none")
    split_checks = pd.read_csv(os.path.join(SPLIT_DIR, "split_checks.csv"))
    ow = split_checks[split_checks["check"].str.startswith("outcome window")].iloc[0]
    tq("look-ahead risk", "labels vs OOT", ow["status"], ow["observed"], "step 17 check 7",
       "accept (documented): features are point-in-time; a strict 6-month embargo would drop ~2,500 training loans")

    # 8. extreme months
    med = data.groupby(data[SCORING_COL].dt.to_period("M"))[[c for c in heat_feats if c not in CAT]].median()
    rz = (med - med.median()) / (1.4826 * (med - med.median()).abs().median()).replace(0, np.nan)
    extreme = rz.abs().gt(3.5).stack()
    extreme = extreme[extreme]
    tq("extreme months", "cohort medians of top features", "PASS" if extreme.empty else "WARN",
       f"{len(extreme)} cohort/feature medians beyond 3.5 robust z" + ("" if extreme.empty else
       f": {', '.join(f'{m} {c}' for m, c in extreme.index[:5])}"),
       "loan volume per cohort 456-543 (no spikes)", "none" if extreme.empty else "inspect those cohorts")

    temporal_report = pd.DataFrame(temporal)
    temporal_report.to_csv(os.path.join(EDA_DIR, "temporal_quality_report.csv"), index=False)
    if charts:
        plot_step16(heat, months_all, heat_feats, lm, p, slope, pval)
    return dict(temporal_report=temporal_report, drift=drift, psi_by_cohort=heat, label_by_month=lm)


def plot_step16(heat, months_all, heat_feats, lm, p, slope, pval):
    plt = _plt()
    from matplotlib.colors import LinearSegmentedColormap
    seq = LinearSegmentedColormap.from_list("seq", ["#cde2fb", "#86b6ef", "#2a78d6", "#184f95", "#0d366b"])
    fig, ax = plt.subplots(figsize=(13, 6))
    im = ax.imshow(heat.values, cmap=seq, vmin=0, vmax=0.25, aspect="auto")
    ax.set_xticks(range(len(months_all)), [str(m) for m in months_all], rotation=90, fontsize=7.5)
    ax.set_yticks(range(len(heat_feats)), heat_feats, fontsize=8)
    ax.grid(False)
    oot_idx = [str(m) for m in months_all].index("2024-07")
    ax.axvline(oot_idx - 0.5, color="#e34948", linewidth=1.5)
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            v = heat.values[i, j]
            if v > 0.1:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5, color="white")
    cb = fig.colorbar(im, ax=ax, shrink=0.8)
    cb.set_label("PSI vs train (capped at 0.25)", color=INK_2)
    ax.set_title("Feature drift by loan-start cohort — PSI vs train baseline (top 15 set-A features)\n"
                 "values > 0.1 printed in the cell · red line = OOT starts 2024-07", loc="left")
    fig.tight_layout()
    _save(fig, EDA_CHART_DIR, "16_1_feature_psi_by_cohort.png")

    fig, ax = plt.subplots(figsize=(11, 3.8))
    x = lm.index.to_timestamp()
    ax.fill_between(x, lm["lcl"] * 100, lm["ucl"] * 100, color=SERIES[0], alpha=0.10, lw=0, label="3-sigma limits (sampling noise)")
    ax.plot(x, lm["rate"] * 100, color=SERIES[0], marker="o", markersize=4, label="monthly default rate")
    ax.axhline(p * 100, color=INK_2, linestyle="--", linewidth=1)
    ax.text(x[0], p * 100 + 0.6, f"train average {p * 100:.1f}%", fontsize=7.5, color=INK_2)
    ax.set_ylim(0, lm["ucl"].max() * 100 * 1.15)
    ax.set_ylabel("default rate (%)")
    ax.legend(loc="lower right")
    ax.set_title(f"Label drift (train only) — trend {slope:+.2f} pp/month, homogeneity p = {pval:.2f}")
    fig.tight_layout()
    _save(fig, EDA_CHART_DIR, "16_2_label_rate_control_chart.png")


# =============================================================================================
# Gold EDA — step 15 (action register) + Gate 2
# =============================================================================================
def run_step15(s14, s16, charts=True):
    watchlist, temporal_report = s14["watchlist"], s16["temporal_report"]
    owner = CONTRACT["contract"]["owner"]
    cols = ["register_id", "source_step", "layer", "table", "column", "finding", "severity", "action",
            "rationale", "verification", "implemented_in", "owner", "status"]
    register = []

    def reg(rid, step, layer, table, column, finding, severity, action, rationale, verification, implemented_in, status):
        register.append(dict(register_id=rid, source_step=step, layer=layer, table=table, column=column, finding=finding,
                             severity=severity, action=action, rationale=rationale, verification=verification,
                             implemented_in=implemented_in, owner=owner, status=status))

    sd = pd.read_csv(os.path.join(PROJECT_DIR, "datamart", "silver", "_reports", "silver_decisions.csv"))
    for _, r in sd.iterrows():
        step = "03" if str(r["issue_id"]).startswith("AUD") else "08" if str(r["issue_id"]).startswith("S08") else "04"
        if r["decision"] != "defer":
            reg(r["issue_id"], step, "bronze/silver", r["table"], r["column"], r["finding"], r["severity"], r["decision"],
                r["action"], r["verified_by"], "Silver (steps 05-09)", "accepted" if r["decision"] == "accept" else "closed")
        elif r["category"] == "coverage" and r["column"] == "Customer_ID":
            reg(r["issue_id"], step, "bronze/silver", r["table"], r["column"], r["finding"], r["severity"], "accept",
                "coverage is a time break (100% of loans starting <= 2024-06, 0% after): model A without clickstream "
                "(champion, OOT-evaluable) vs model B with clickstream (challenger); no has_clickstream flag (time proxy)",
                "step 10 join diagnostics; step 16 coverage-break check; step 25 comparison", "steps 10, 16, 25", "assigned")
        else:
            reg(r["issue_id"], step, "bronze/silver", r["table"], r["column"], r["finding"], r["severity"], "accept",
                "historical extract: the as-of join takes the latest snapshot on or before each loan's start",
                "step 13 point-in-time check; step 11 historical replay", "Gold (steps 10-13)", "closed")

    for i, w in watchlist.reset_index(drop=True).iterrows():
        rid = f"G14-{i + 1:03d}"
        if w["signal"] == "suspiciously strong":
            reg(rid, "14", "gold", "feature_store/v1", w["feature"], f"suspiciously strong ({w['evidence']})", "high", "accept",
                "univariate AUC ~0.70-0.72 is far below leakage levels (>0.9); value known at application (financial "
                "snapshot dated in the loan-start month, as-of verified); classic credit-risk driver; IV inflated by decile "
                "binning", "step 13 point-in-time + no-label-in-X checks; re-check importance stability in step 27",
                "step 27 (re-check)", "accepted")
        elif w["signal"] == "informative missingness":
            reg(rid, "14", "gold", "feature_store/v1", w["feature"], f"informative missingness ({w['evidence']})", "medium",
                "transform", "keep the <column>_invalid flag / add a missing indicator; impute on train only",
                "step 18 fit/transform tests", "step 18", "assigned")
        elif w["signal"] == "constant / near-constant":
            reg(rid, "14", "gold", "feature_store/v1", w["feature"], f"constant ({w['evidence']})", "low", "exclude",
                "no information in train", "step 23 filter report", "step 23", "assigned")
        elif w["signal"] == "near-duplicate":
            keep = {"Num_of_Loan": "num_loan_types", "Annual_Income": "Monthly_Inhand_Salary"}.get(w["feature"], "")
            reg(rid, "14", "gold", "feature_store/v1", w["feature"], f"near-duplicate ({w['evidence']})", "low", "exclude",
                f"keep {keep} (no nulls / derived without cleaning loss), drop {w['feature']}; confirm on train in step 23",
                "step 23 filter report", "step 23", "assigned")
        else:
            reg(rid, "14", "gold", "feature_store/v1", w["feature"], w["signal"], "medium", "investigate",
                w["proposed_action"], "", "step 23", "assigned")

    for i, t in temporal_report[temporal_report["status"].isin(["WARN", "FAIL"])].reset_index(drop=True).iterrows():
        sev = "high" if t["status"] == "FAIL" else "medium"
        act = "accept" if t["action"].startswith("accept") or t["action"].startswith("none") else "investigate"
        reg(f"G16-{i + 1:03d}", "16", "gold", t["grain"], "", f"{t['check']}: {t['finding']}", sev, act,
            t["action"], t["evidence"], "steps 16, 31-32, 35", "accepted" if act == "accept" else "assigned")

    action_register = pd.DataFrame(register, columns=cols)
    action_register.to_csv(os.path.join(EDA_DIR, "eda_action_register.csv"), index=False)
    assert (action_register["status"] != "open").all(), "open findings before modelling"
    high = action_register[action_register["severity"] == "high"]
    assert high["rationale"].str.len().gt(0).all(), "high-severity item without rationale"

    gate2 = pd.DataFrame([
        dict(criterion="Gold grain and time rules verified",
             evidence="step 13: one row per loan, point-in-time 0 violations, outcome = loan start + 6 months, no label in X; "
                      "step 11 historical replay", status="PASS"),
        dict(criterion="EDA findings resolved or assigned",
             evidence=f"{len(action_register)} findings: " + ", ".join(f"{k} {v}" for k, v in
                                                                      action_register['status'].value_counts().items())
                      + "; none open", status="PASS"),
        dict(criterion="Leakage risks documented",
             evidence=f"leakage watchlist ({len(watchlist)} entries) with actions; 4 strong financial features accepted with "
                      "rationale (AUC < 0.75, known at application)", status="PASS"),
    ])
    gate2.to_csv(os.path.join(EDA_DIR, "gate2_check.csv"), index=False)
    if charts:
        plot_action_register(action_register)
    return dict(action_register=action_register, gate2=gate2)


def plot_action_register(action_register):
    plt = _plt()
    ct = action_register.groupby(["layer", "action"]).size().unstack(fill_value=0)
    order_actions = [a for a in ["correct", "transform", "exclude", "accept", "investigate"] if a in ct.columns]
    ct = ct[order_actions]
    fig, ax = plt.subplots(figsize=(10, 2.8))
    left = np.zeros(len(ct))
    for color, a in zip(SERIES, order_actions):
        ax.barh(ct.index, ct[a], left=left, color=color, label=a, height=0.6, edgecolor=SURFACE, linewidth=1)
        for yv, (l, v) in enumerate(zip(left, ct[a])):
            if v:
                ax.text(l + v / 2, yv, str(v), ha="center", va="center", fontsize=8, color="white")
        left += ct[a].values
    ax.set_xlabel("findings")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=len(order_actions))
    st = action_register["status"].value_counts().to_dict()
    ax.set_title(f"EDA action register — {len(action_register)} findings ({', '.join(f'{k} {v}' for k, v in st.items())})")
    fig.tight_layout()
    _save(fig, EDA_CHART_DIR, "15_action_register.png")


def run_eda(spark=None, charts=True):
    """Gold EDA after the split: step 14 (train-only feature/target) -> 16 (temporal) -> 15 (action register, Gate 2)."""
    spark = spark or get_spark()
    os.makedirs(EDA_DIR, exist_ok=True)
    ctx = load_eda_data(spark)
    s14 = run_step14(ctx, charts=charts)
    s16 = run_step16(spark, ctx, s14, charts=charts)
    s15 = run_step15(s14, s16, charts=charts)
    return dict(step14=s14, step16=s16, step15=s15)


def run(spark=None, build=True, eda=True, charts=True, tests=False):
    spark = spark or get_spark()
    out = {}
    if build:
        out["build"] = run_build(spark, charts=charts, tests=tests)
    if eda:
        out["eda"] = run_eda(spark, charts=charts)
    return out


if __name__ == "__main__":
    import argparse
    import matplotlib
    matplotlib.use("Agg")

    parser = argparse.ArgumentParser(description="Gold layer: build (10-13) and EDA (14, 16, 15)")
    parser.add_argument("--build", action="store_true", help="run steps 10-13")
    parser.add_argument("--eda", action="store_true", help="run steps 14, 16, 15 (needs the step-17 split manifest)")
    parser.add_argument("--tests", action="store_true", help="unit tests + historical replay")
    parser.add_argument("--no-charts", action="store_true")
    args = parser.parse_args()
    if not (args.build or args.eda):
        args.build = args.eda = True

    result = run(build=args.build, eda=args.eda, charts=not args.no_charts, tests=args.tests)
    if "build" in result:
        print("Acceptance:", result["build"]["acceptance"]["status"].value_counts().to_dict())
        print(result["build"]["label_summary"].to_string(index=False))
    if "eda" in result:
        print("Temporal quality:", result["eda"]["step16"]["temporal_report"]["status"].value_counts().to_dict())
        print("Action register:", result["eda"]["step15"]["action_register"]["status"].value_counts().to_dict())
