"""Monthly production pipeline: the task functions called by the three Airflow DAGs.

Generated from notebooks/pipeline_monthly.ipynb (library cells, unchanged); the notebook also verifies that running
month by month reproduces the full build (Bronze / Silver / Gold) and steps 34-35 (predictions, monitoring).

Command line (Airflow BashOperator / BashSensor):
    python util/pipeline_monthly.py bronze --snapshotdate 2024-07-01
    python util/pipeline_monthly.py silver | gold_features | gold_labels --snapshotdate ...   # data pipeline DAG
    python util/pipeline_monthly.py full_data_validation | register | monitoring_reference # training DAG
    python util/pipeline_monthly.py champion_ready                                          # sensor (exit 0 = ready)
    python util/pipeline_monthly.py score | monitor | governance --snapshotdate ...         # inference DAG
    python util/pipeline_monthly.py backfill_data --start 2023-01-01 --end 2025-11-01       # all months, one session
"""

import os, sys, json, glob, shutil, uuid, argparse, hashlib, warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yaml

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
CODE_DIR = (os.path.dirname(os.path.dirname(os.path.abspath(__file__))) if "__file__" in globals()
            else os.path.abspath(".."))                      # module: util/.. ; notebook: notebooks/..
sys.path.insert(0, CODE_DIR)
from pyspark.sql import functions as F
from util import dataprocessing_bronze as bronze
from util import dataprocessing_silver as silver
from util import dataprocessing_gold as gold

PROJECT_DIR = bronze.PROJECT_DIR                              # data root (PROJECT_DIR env var, /app in Docker)
SCORING_COL = gold.SCORING_COL
REGISTERED_MODEL = "loan_default_30dpd_6mob"
DEP_DIR = os.path.join(PROJECT_DIR, "datamart", "ml", "deployment", "v1")
MON_DIR = os.path.join(DEP_DIR, "monitoring")
GOV_DIR = os.path.join(DEP_DIR, "governance")
PRED_DIR = os.path.join(PROJECT_DIR, "datamart", "gold", "model_predictions", REGISTERED_MODEL)
MLFLOW_DB = os.path.join(PROJECT_DIR, "datamart", "ml", "mlflow", "mlflow.db")

def month_str(snapshotdate):
    return pd.Timestamp(snapshotdate).strftime("%Y-%m-%d")

def now_utc():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def write_partition(df, table_path, part_col, value):
    # Replace exactly one partition folder (<part_col>=<value>) of a partitioned Parquet table; other months untouched
    assert df.select(part_col).distinct().count() == 1, "one partition value per write"
    (df.write.mode("overwrite").option("partitionOverwriteMode", "dynamic")
       .partitionBy(part_col).parquet(table_path))


def bronze_month(spark, snapshotdate):
    D = month_str(snapshotdate)
    os.makedirs(bronze.BRONZE_DIR, exist_ok=True)
    batch_id = f"bronze_{D.replace('-', '')}_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{uuid.uuid4().hex[:6]}"
    ingest_ts = datetime.now(timezone.utc).replace(microsecond=0)
    rows = []
    for table, (file_name, event_col) in bronze.SOURCES.items():
        started = datetime.now()
        src = os.path.join(bronze.RAW_DIR, file_name)
        sha = bronze.file_sha256(src)
        header, src_rows, _ = bronze.source_profile(src)
        s_hash = bronze.schema_hash(header)
        s_version = bronze.next_schema_version(bronze.read_log(), table, s_hash)
        df = (bronze.read_source_as_is(spark, src, header)                 # same columns, same order as the full load
              .withColumn("_source_file", F.lit(file_name))
              .withColumn("_source_sha256", F.lit(sha))
              .withColumn("_event_date", F.date_format(F.to_date(F.col(event_col), "yyyy-MM-dd"), "yyyy-MM-dd"))
              .withColumn("_ingest_ts", F.lit(ingest_ts.strftime("%Y-%m-%dT%H:%M:%SZ")))
              .withColumn("_batch_id", F.lit(batch_id))
              .withColumn("_schema_version", F.lit(s_version))
              .withColumn("_schema_hash", F.lit(s_hash)))
        pdf = df.filter(F.col("_event_date") == D).orderBy("_source_row_num").toPandas()
        out_dir = os.path.join(bronze.BRONZE_DIR, table)
        os.makedirs(out_dir, exist_ok=True)
        if len(pdf):
            path = os.path.join(out_dir, f"bronze_{table}_{D.replace('-', '_')}.csv")
            pdf.to_csv(path + ".tmp", index=False)
            os.replace(path + ".tmp", path)                                  # atomic per month
        rows.append(dict(batch_id=batch_id, table=table, source_file=file_name, source_sha256=sha,
                         source_rows=src_rows, schema_hash=s_hash, ingest_ts=ingest_ts.isoformat(),
                         bronze_rows=len(pdf), event_partitions=int(len(pdf) > 0), min_event_date=D,
                         max_event_date=D, schema_version=s_version, mode="monthly_partition",
                         status="LOADED" if len(pdf) else "NO_DATA",
                         duration_s=round((datetime.now() - started).total_seconds(), 2), message=""))
    bronze.append_log(rows)
    return pd.DataFrame(rows)

def silver_month(spark, snapshotdate):
    D = month_str(snapshotdate)
    suffix = D.replace("-", "_")
    batch_id = f"silver_{D.replace('-', '')}_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{uuid.uuid4().hex[:6]}"
    silver_ts = now_utc()
    cleaned, quarantined, log = {}, {}, []
    for t in silver.TABLES:
        f = os.path.join(bronze.BRONZE_DIR, t, f"bronze_{t}_{suffix}.csv")
        if not os.path.exists(f):
            log.append(dict(table=t, status="NO_DATA", silver_rows=0, quarantined=0))
            continue
        b = spark.read.option("header", True).option("escape", '"').option("inferSchema", False).csv(f)
        passed, quarantined[t], _, _ = silver.enforce_schema(t, b)
        df, _, _ = silver.clean_table(t, passed)
        cleaned[t] = silver.add_transforms(t, df).cache()
    if not cleaned:
        return pd.DataFrame(log)
    # validation: row-level and key rules of this month's tables (cross-table / history rules: full rebuild)
    frames = dict(cleaned)
    frames.setdefault("lms_loan_daily", next(iter(cleaned.values())))    # only used for the date upper bound
    rules = [r for r in silver.build_rules(frames)
             if r["table"] in cleaned and ("expr" in r or r["kind"] == "key")]
    report, _ = silver.run_rules(cleaned, rules, f"monthly_{D}")
    report.insert(0, "snapshot_date", D)
    os.makedirs(silver.REPORT_DIR, exist_ok=True)
    rep_path = os.path.join(silver.REPORT_DIR, "validation_monthly.csv")
    old = pd.read_csv(rep_path) if os.path.exists(rep_path) else pd.DataFrame()
    if len(old):
        old = old[old["snapshot_date"] != D]
    pd.concat([old, report], ignore_index=True).to_csv(rep_path, index=False)
    failed = report[report["status"] == "FAIL"]
    if len(failed):
        raise AssertionError(f"Silver validation failed for {D}:\n{failed[['table', 'rule', 'pct', 'fail_pct']]}")
    for t, df in cleaned.items():
        out = df.withColumn("_silver_batch_id", F.lit(batch_id)).withColumn("_silver_ts", F.lit(silver_ts))
        write_partition(out, os.path.join(silver.SILVER_DIR, t), "snapshot_date", D)
        n_q = quarantined[t].count()
        if n_q:
            quarantined[t].write.mode("overwrite").parquet(os.path.join(silver.QUARANTINE_DIR, t, f"month={D}"))
        log.append(dict(table=t, status="LOADED", silver_rows=df.count(), quarantined=n_q))
    for df in cleaned.values():
        df.unpersist()
    return pd.DataFrame(log)

def gold_features_month(spark, snapshotdate):
    D = month_str(snapshotdate)
    d = F.lit(D).cast("date")
    frames = {t: silver.read_silver(spark, t) for t in [gold.LABEL_TABLE] + gold.FEATURE_TABLES}
    spine = gold.build_spine(frames[gold.LABEL_TABLE].filter((F.col("snapshot_date") == d) & (F.col(SCORING_COL) == d)))
    spine = spine.cache()
    n = spine.count()
    if n == 0:
        return dict(month=D, status="NO_NEW_LOANS", loans=0)
    asof_frames = {t: frames[t].filter(F.col("snapshot_date") <= d) for t in gold.FEATURE_TABLES}  # never the future
    features, _ = gold.build_features(spine, asof_frames)
    features = features.cache()
    feats = gold.feature_list(features)
    gold.check_predictors(feats)
    assert features.count() == n == features.select("loan_id").distinct().count(), "one row per loan"
    for c in [c for c in features.columns if c.startswith("_asof_")]:
        assert features.filter(F.col(c) > F.col(SCORING_COL)).count() == 0, f"{c}: snapshot after loan start"
    out = (features.withColumn("_feature_version", F.lit(gold.FEATURE_VERSION))
                   .withColumn("_gold_batch_id", F.lit(f"gold_{D.replace('-', '')}_{uuid.uuid4().hex[:6]}"))
                   .withColumn("_gold_ts", F.lit(now_utc())))
    write_partition(out, gold.FEATURE_STORE_PATH, SCORING_COL, D)
    features.unpersist(); spine.unpersist()
    return dict(month=D, status="LOADED", loans=n, features=len(feats))

def gold_labels_month(spark, snapshotdate):
    D = month_str(snapshotdate)
    d = F.lit(D).cast("date")
    lms_d = silver.read_silver(spark, gold.LABEL_TABLE).filter(F.col("snapshot_date") == d)
    spine = gold.build_spine(lms_d.filter(F.col("installment_num") == gold.MOB)).cache()
    n = spine.count()
    if n == 0:
        return dict(month=D, status="NO_LABELS_DUE", labels=0)
    labels = gold.build_labels(lms_d, spine, pd.Timestamp(D).date())
    assert labels.count() == n and labels.filter(F.col("maturity_status") != "mature").count() == 0
    out = (labels.withColumn("_gold_batch_id", F.lit(f"gold_{D.replace('-', '')}_{uuid.uuid4().hex[:6]}"))
                 .withColumn("_gold_ts", F.lit(now_utc())))
    write_partition(out, gold.LABEL_STORE_PATH, "label_snapshot_date", D)
    return dict(month=D, status="LOADED", labels=n, label_rate=float(labels.agg(F.avg("label")).first()[0]))


def mlflow_client():
    import mlflow
    from mlflow.tracking import MlflowClient
    uri = "sqlite:///" + MLFLOW_DB.replace("\\", "/")
    mlflow.set_tracking_uri(uri)
    mlflow.set_registry_uri(uri)
    return MlflowClient()

def champion_ready(alias="champion"):
    if not os.path.exists(MLFLOW_DB):
        return False
    try:
        mlflow_client().get_model_version_by_alias(REGISTERED_MODEL, alias)
        return True
    except Exception:
        return False

def resolve_alias(alias="champion"):
    import mlflow
    client = mlflow_client()
    mv = client.get_model_version_by_alias(REGISTERED_MODEL, alias)
    uri = f"models:/{REGISTERED_MODEL}@{alias}"
    pyfunc_model = mlflow.pyfunc.load_model(uri)
    return dict(alias=alias, version=int(mv.version), uri=uri, threshold=float(mv.tags["operating_threshold"]),
                model_sha256=mv.tags["model_sha256"], pyfunc=pyfunc_model, sklearn=mlflow.sklearn.load_model(uri),
                features=pyfunc_model.metadata.get_input_schema().input_names())

def load_scoring_rows(spark, snapshotdate):
    # Gold features of the loans starting in the month (no labels, no split manifest) + generated features
    from util import ml_feature_engineering as fe
    start = pd.Timestamp(snapshotdate)
    end = start + pd.offsets.MonthBegin(1)
    fs = gold.get_features(spark)
    feats = gold.feature_list(fs)
    asof_cols = [c for c in fs.columns if c.startswith("_asof_")]
    rows = (fs.filter((F.col(SCORING_COL) >= F.lit(start.date())) & (F.col(SCORING_COL) < F.lit(end.date())))
              .select("loan_id", "Customer_ID", SCORING_COL, *asof_cols, *feats).toPandas())
    binary = [c for c in feats if c.endswith("_invalid") or c in gold.LOAN_TYPE_COLUMNS or c == "type_of_loan_unknown"]
    for c in binary:                                   # same typing rule as training (gold.load_eda_data)
        rows[c] = rows[c].astype(float)
    return fe.apply_catalog(rows, fe.load_catalog()), asof_cols

def reason_codes(sk_model, X, top=3):
    # Top features pushing each loan's risk UP (SHAP log-odds contributions, summed back to original features)
    import xgboost as xgb
    pre, model = sk_model.named_steps["pre"], sk_model.named_steps["model"]
    contrib = model.get_booster().predict(xgb.DMatrix(pre.transform(X)), pred_contribs=True)[:, :-1]
    cats = sorted([c for c in X.columns if X[c].dtype == object], key=len, reverse=True)
    owner = []
    for name in pre.get_feature_names_out():
        part, col = name.split("__", 1)
        if part == "missing_ind":
            col = col[len("missingindicator_"):]
        elif part == "cat":
            col = next(c for c in cats if col.startswith(c + "_"))
        owner.append(col)
    owner = np.array(owner)
    per_feat = pd.DataFrame({f: contrib[:, owner == f].sum(axis=1) for f in X.columns})
    out = []
    for _, r in per_feat.iterrows():
        up = r[r > 0].sort_values(ascending=False).head(top)
        out.append([f"{f} (+{v:.2f})" for f, v in up.items()] + [""] * (top - len(up)))
    return pd.DataFrame(out, columns=[f"reason_{i + 1}" for i in range(top)])

def prediction_path(snapshotdate, alias="champion"):
    tag = pd.Timestamp(snapshotdate).strftime("%Y_%m_%d")
    return os.path.join(PRED_DIR, f"{REGISTERED_MODEL}_{alias}_predictions_{tag}.parquet")

def score_month(spark, snapshotdate, alias="champion"):
    D = month_str(snapshotdate)
    m = resolve_alias(alias)
    rows, asof_cols = load_scoring_rows(spark, D)
    if rows.empty:
        return dict(month=D[:7], status="NO_NEW_LOANS", loans=0)
    X = rows[m["features"]]
    p = np.asarray(m["pyfunc"].predict(X))[:, 1]             # signature-enforced scoring path
    for c in asof_cols:                                       # as-of rule: no snapshot after the loan start
        snap = pd.to_datetime(rows[c], errors="coerce")
        assert (snap.isna() | (snap <= pd.to_datetime(rows[SCORING_COL]))).all(), f"{c}: snapshot after loan start"
    out = pd.DataFrame({"loan_id": rows["loan_id"], "Customer_ID": rows["Customer_ID"],
                        "loan_start_date": pd.to_datetime(rows[SCORING_COL]).dt.strftime("%Y-%m-%d"),
                        "default_probability": p, "threshold": m["threshold"],
                        "flag_high_risk": (p >= m["threshold"]).astype(int)})
    out = pd.concat([out, reason_codes(m["sklearn"], X)], axis=1)
    out = out.assign(model_name=REGISTERED_MODEL, model_alias=alias, model_version=m["version"],
                     model_sha256=m["model_sha256"], feature_store="gold v1", scored_ts=now_utc())
    assert out["loan_id"].is_unique and out["default_probability"].between(0, 1).all()
    os.makedirs(PRED_DIR, exist_ok=True)
    path = prediction_path(D, alias)
    out.to_parquet(path + ".tmp", index=False)
    os.replace(path + ".tmp", path)
    log = dict(month=D[:7], loans=len(out), model_alias=alias, model_version=m["version"], threshold=m["threshold"],
               mean_probability=float(p.mean()), flagged_share=float(out["flag_high_risk"].mean()),
               file=os.path.relpath(path, PROJECT_DIR).replace("\\", "/"),
               file_sha256=hashlib.sha256(open(path, "rb").read()).hexdigest(), scored_ts=out["scored_ts"].iloc[0])
    log_path = os.path.join(DEP_DIR, "inference_log.csv")                     # upsert this month
    old = pd.read_csv(log_path) if os.path.exists(log_path) else pd.DataFrame()
    if len(old):
        old = old[old["month"] != log["month"]]
    pd.concat([old, pd.DataFrame([log])], ignore_index=True).sort_values("month").to_csv(log_path, index=False)
    return dict(status="SCORED", **{k: v for k, v in log.items() if k != "file_sha256"})


MON_CFG_DEFAULT = {"psi_bins": 10, "psi_watch": 0.10, "psi_alert": 0.25, "missing_increase_alert_pp": 10.0,
                   "flagged_share_alert_pp": 10.0, "pr_auc_drop_alert": 0.05, "calibration_gap_alert_pp": 5.0,
                   "recall_drop_watch_pp": 10.0, "label_delay": "6 months (label = 30+ DPD at month-on-book 6)",
                   "reference": "training partition (step 14 split), features + champion scores"}

def _bin_spec(values, categorical):
    v = values.dropna()
    if categorical:
        return {"type": "categorical", "levels": sorted(v.astype(str).unique().tolist())}
    edges = np.unique(np.quantile(v.astype(float), np.linspace(0, 1, MON_CFG_DEFAULT["psi_bins"] + 1)))
    return {"type": "numeric", "edges": edges[1:-1].tolist()}

def _shares(values, spec):
    n = len(values)
    miss = values.isna()
    if spec["type"] == "categorical":
        s = values[~miss].astype(str)
        counts = [int((s == lv).sum()) for lv in spec["levels"]] + [int((~s.isin(spec["levels"])).sum())]
    else:
        idx = np.searchsorted(spec["edges"], values[~miss].astype(float), side="right")
        counts = np.bincount(idx, minlength=len(spec["edges"]) + 1).tolist()
    return np.array(counts + [int(miss.sum())], dtype=float) / max(n, 1)

def _psi(ref_share, cur_share, eps=1e-4):
    r, c = np.clip(ref_share, eps, None), np.clip(cur_share, eps, None)
    return float(np.sum((c - r) * np.log(c / r)))

def build_monitoring_reference(spark):
    # Frozen training reference for monitoring (training DAG, after registration); same content as step 35
    from util import ml_feature_engineering as fe
    ctx = gold.load_eda_data(spark)
    train_rows = fe.apply_catalog(ctx["data"][ctx["data"]["partition"] == "train"].copy(), fe.load_catalog())
    champ, chall = resolve_alias("champion"), resolve_alias("challenger")
    cat_cols = set(ctx["categorical"])
    ref = {}
    for feats in (champ["features"], chall["features"]):
        for f in feats:
            if f not in ref:
                spec = _bin_spec(train_rows[f], f in cat_cols)
                ref[f] = {**spec, "ref_share": _shares(train_rows[f], spec).tolist(),
                          "ref_missing_pct": float(train_rows[f].isna().mean() * 100)}
    scores = champ["sklearn"].predict_proba(train_rows[champ["features"]])[:, 1]
    score_spec = {"type": "numeric", "edges": np.unique(np.quantile(scores, np.linspace(0, 1, 11)))[1:-1].tolist()}
    final_eval = pd.read_csv(os.path.join(PROJECT_DIR, "datamart", "ml", "model_development", "v1", "final_evaluation.csv"))
    val = final_eval.query("partition == 'validation' and model == 'champion XGB model A'").iloc[0]
    os.makedirs(DEP_DIR, exist_ok=True)
    with open(os.path.join(DEP_DIR, "monitoring_config.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(MON_CFG_DEFAULT, f, sort_keys=False)
    with open(os.path.join(DEP_DIR, "monitoring_reference.json"), "w", encoding="utf-8") as f:
        json.dump({"features": ref, "score": {**score_spec, "ref_share": _shares(pd.Series(scores), score_spec).tolist()},
                   "validation": {k: float(val[k]) for k in ["pr_auc", "recall", "precision", "flagged_share"]},
                   "reference_rows": int(len(train_rows))}, f, indent=1)
    return dict(status="BUILT", inputs=len(ref), reference_rows=len(train_rows), champion_version=champ["version"])

def _load_monitoring_setup():
    with open(os.path.join(DEP_DIR, "monitoring_config.yaml"), encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    with open(os.path.join(DEP_DIR, "monitoring_reference.json"), encoding="utf-8") as f:
        ref = json.load(f)
    return cfg, ref

def monitor_month(spark, snapshotdate):
    from sklearn.metrics import average_precision_score
    D = month_str(snapshotdate)
    cfg, ref = _load_monitoring_setup()
    os.makedirs(MON_DIR, exist_ok=True)
    month_end = (pd.Timestamp(D) + pd.offsets.MonthEnd(0)).strftime("%Y-%m-%d")
    alerts, result = [], dict(month=D[:7])
    def sev(v, watch, alert):
        return "ALERT" if v > alert else "WATCH" if v > watch else "OK"

    # 1) at scoring time: data quality, input drift, score drift, decisions of batch D
    if os.path.exists(prediction_path(D)):
        pred = pd.read_parquet(prediction_path(D))
        rows, _ = load_scoring_rows(spark, D)
        models = {"champion": resolve_alias("champion")["features"], "challenger": resolve_alias("challenger")["features"]}
        drift = []
        for role, feats in models.items():
            for f in feats:
                r = ref["features"][f]
                cur_missing = float(rows[f].isna().mean() * 100)
                p_ = _psi(np.array(r["ref_share"]), _shares(rows[f], r))
                drift.append(dict(month=D[:7], model=role, feature=f, psi=p_, missing_pct=cur_missing,
                                  ref_missing_pct=r["ref_missing_pct"]))
                if role == "challenger" and f in models["champion"]:
                    continue
                s = sev(p_, cfg["psi_watch"], cfg["psi_alert"])
                if s != "OK":
                    alerts.append(dict(month=D[:7], available_on=month_end, model=role, check="input drift (PSI)",
                                       item=f, value=round(p_, 3), severity=s))
                if cur_missing - r["ref_missing_pct"] > cfg["missing_increase_alert_pp"]:
                    alerts.append(dict(month=D[:7], available_on=month_end, model=role, check="missing values",
                                       item=f, value=round(cur_missing, 1), severity="ALERT"))
        pd.DataFrame(drift).to_csv(os.path.join(MON_DIR, f"drift_{D}.csv"), index=False)
        p = pred["default_probability"].values
        score_psi = _psi(np.array(ref["score"]["ref_share"]), _shares(pd.Series(p), ref["score"]))
        flagged = float(pred["flag_high_risk"].mean())
        s = sev(score_psi, cfg["psi_watch"], cfg["psi_alert"])
        if s != "OK":
            alerts.append(dict(month=D[:7], available_on=month_end, model="champion", check="score drift (PSI)",
                               item="score", value=round(score_psi, 3), severity=s))
        if abs(flagged - ref["validation"]["flagged_share"]) * 100 > cfg["flagged_share_alert_pp"]:
            alerts.append(dict(month=D[:7], available_on=month_end, model="champion", check="flagged share",
                               item="score", value=round(flagged, 3), severity="ALERT"))
        pd.DataFrame([dict(month=D[:7], loans=len(pred), scored_on=month_end, score_psi=score_psi,
                           mean_predicted=float(p.mean()), flagged_share=flagged)]).to_csv(
            os.path.join(MON_DIR, f"scoring_{D}.csv"), index=False)
        result.update(scored_loans=len(pred), score_psi=round(score_psi, 4))

    # 2) delayed: performance of the cohort that started 6 months earlier (labels land at D)
    cohort = (pd.Timestamp(D) - pd.DateOffset(months=gold.MOB)).strftime("%Y-%m-%d")
    lab_path = os.path.join(gold.LABEL_STORE_PATH, f"label_snapshot_date={D}")
    if os.path.exists(prediction_path(cohort)) and os.path.exists(lab_path):
        pred = pd.read_parquet(prediction_path(cohort))
        labels = spark.read.parquet(gold.LABEL_STORE_PATH).filter(F.col("label_snapshot_date") == F.lit(D).cast("date")) \
                      .filter("maturity_status = 'mature'").select("loan_id", "label").toPandas()
        lab = pred[["loan_id", "default_probability", "flag_high_risk"]].merge(labels, on="loan_id", how="inner")
        y, pl = lab["label"].values, lab["default_probability"].values
        tp = int(((lab["flag_high_risk"] == 1) & (y == 1)).sum()); fp = int(((lab["flag_high_risk"] == 1) & (y == 0)).sum())
        fn = int(((lab["flag_high_risk"] == 0) & (y == 1)).sum())
        perf = dict(cohort=cohort[:7], labels_available_on=D, labelled_loans=len(lab),
                    pr_auc=average_precision_score(y, pl), recall=tp / max(tp + fn, 1), precision=tp / max(tp + fp, 1),
                    actual_rate=float(y.mean()), mean_predicted=float(pl.mean()))
        perf["calibration_gap_pp"] = (perf["mean_predicted"] - perf["actual_rate"]) * 100
        val = ref["validation"]
        for check, bad, v, s in [
            ("PR-AUC", perf["pr_auc"] < val["pr_auc"] - cfg["pr_auc_drop_alert"], perf["pr_auc"], "ALERT"),
            ("calibration gap (pp)", abs(perf["calibration_gap_pp"]) > cfg["calibration_gap_alert_pp"], perf["calibration_gap_pp"], "ALERT"),
            ("recall at threshold", (val["recall"] - perf["recall"]) * 100 > cfg["recall_drop_watch_pp"], perf["recall"], "WATCH")]:
            if bad:
                alerts.append(dict(month=cohort[:7], available_on=D, model="champion", check=check, item="performance",
                                   value=round(v, 3), severity=s))
        pd.DataFrame([perf]).to_csv(os.path.join(MON_DIR, f"performance_{cohort}.csv"), index=False)
        result.update(performance_cohort=cohort[:7], pr_auc=round(perf["pr_auc"], 4))
    pd.DataFrame(alerts, columns=["month", "available_on", "model", "check", "item", "value", "severity"]).to_csv(
        os.path.join(MON_DIR, f"alerts_{D}.csv"), index=False)
    result["alerts"] = len(alerts)
    return result


GOV_DEFAULT = {"psi_alert": 0.25, "pr_auc_drop": 0.05, "calibration_gap_pp": 5.0, "missing_increase_pp": 10.0,
               "consecutive_months": 2}

def _max_run(flags):
    best = cur = 0
    for f in flags:
        cur = cur + 1 if f else 0
        best = max(best, cur)
    return best

def governance_check(snapshotdate):
    # Step-36 refresh rules on all monitoring up to D: a refit trigger needs 2 consecutive months
    D = month_str(snapshotdate)
    cfg, ref = _load_monitoring_setup()
    need = GOV_DEFAULT["consecutive_months"]
    read = lambda pat: pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(MON_DIR, pat)))
                                  if os.path.basename(f).split("_")[-1][:10] <= D], ignore_index=True) \
        if glob.glob(os.path.join(MON_DIR, pat)) else pd.DataFrame()
    drift, scoring, perf = read("drift_*.csv"), read("scoring_*.csv"), read("performance_*.csv")
    if len(perf):
        perf = perf[perf["labels_available_on"] <= D]
    rules = []
    if len(drift):
        ch = drift[drift["model"] == "champion"].groupby("month")["psi"].max().sort_index()
        rules.append(("champion", "input_drift", (ch > cfg["psi_alert"]).tolist(), "refit"))
        cl = drift[(drift["model"] == "challenger") & drift["feature"].str.startswith("fe_")]
        if len(cl):
            up = (cl["missing_pct"] - cl["ref_missing_pct"]).groupby(cl["month"]).max().sort_index()
            rules.append(("challenger", "data_source", (up > cfg["missing_increase_alert_pp"]).tolist(), "rebuild"))
    if len(scoring):
        sc = scoring.sort_values("month")
        rules.append(("champion", "score_drift", (sc["score_psi"] > cfg["psi_alert"]).tolist(), "refit"))
    if len(perf):
        pf = perf.sort_values("cohort")
        rules.append(("champion", "performance", (pf["pr_auc"] < ref["validation"]["pr_auc"] - cfg["pr_auc_drop_alert"]).tolist(), "refit"))
        rules.append(("champion", "calibration", (pf["calibration_gap_pp"].abs() > cfg["calibration_gap_alert_pp"]).tolist(), "refit"))
    rows = []
    for model, rule, flags, action in rules:
        need_r = 1 if rule == "data_source" else need
        run_len = _max_run(flags)
        status = "TRIGGERED" if run_len >= need_r else "WATCH" if any(flags) else "not triggered"
        rows.append(dict(model=model, trigger=rule, months_flagged=int(sum(flags)), longest_run=run_len,
                         status=status, action=action if status == "TRIGGERED" else ("investigate" if any(flags) else "none")))
    decisions = pd.DataFrame(rows, columns=["model", "trigger", "months_flagged", "longest_run", "status", "action"])
    retrain = bool(((decisions["model"] == "champion") & (decisions["status"] == "TRIGGERED")).any())
    decision = {"as_of": D, "retrain_champion": retrain,
                "triggers": decisions.to_dict(orient="records"), "decided_ts": now_utc(),
                "rule": f"refit when a champion trigger persists {need} consecutive months; challenger data_source -> rebuild"}
    os.makedirs(GOV_DIR, exist_ok=True)
    with open(os.path.join(GOV_DIR, f"decision_{D}.json"), "w", encoding="utf-8") as f:
        json.dump(decision, f, indent=1)
    return decision


def full_data_validation(spark):
    # Training DAG, before training: rebuild and re-validate the FULL history (all cross-table and ledger rules,
    # Bronze audit, Silver decision register, Gold acceptance + reports). Same data as the monthly path (verified).
    import matplotlib
    matplotlib.use("Agg")
    b = bronze.run(spark, force=True, eda=True, charts=True, html=False, tests=False)
    s = silver.run(spark, eda=True, charts=True, tests=False)
    g = gold.run_build(spark, charts=True, tests=False)
    return dict(bronze_audit=b["audit_report"]["status"].value_counts().to_dict(),
                silver_post_clean=s["validation_report"].query("stage == 'post_clean'")["status"].value_counts().to_dict(),
                gold_acceptance=g["acceptance"]["status"].value_counts().to_dict())

def register_models(spark):
    # Training DAG: step 33 (register the signed-off models and set the aliases)
    from util import ml_deployment as dep
    dep.run(spark, steps=["step33"])
    return dep.registry.drop(columns=["model_sha256"]).to_dict(orient="records")

TASKS = {
    "full_data_validation": lambda s, d: full_data_validation(s), "register": lambda s, d: register_models(s),
    "bronze": lambda s, d: bronze_month(s, d), "silver": lambda s, d: silver_month(s, d),
    "gold_features": lambda s, d: gold_features_month(s, d), "gold_labels": lambda s, d: gold_labels_month(s, d),
    "score": lambda s, d: score_month(s, d), "monitor": lambda s, d: monitor_month(s, d),
    "governance": lambda s, d: governance_check(d), "monitoring_reference": lambda s, d: build_monitoring_reference(s),
}
NO_SPARK = {"governance"}

def main(argv=None):
    parser = argparse.ArgumentParser(description="Monthly production pipeline task (Airflow entry point)")
    parser.add_argument("task", choices=sorted(TASKS) + ["champion_ready", "backfill_data"])
    parser.add_argument("--snapshotdate", help="YYYY-MM-DD (the Airflow run date {{ ds }})")
    parser.add_argument("--start", help="backfill_data: first month")
    parser.add_argument("--end", help="backfill_data: last month")
    args = parser.parse_args(argv)
    if args.task == "champion_ready":                          # sensor: exit 0 when @champion exists
        ok = champion_ready()
        print("@champion ready" if ok else "@champion not registered yet")
        sys.exit(0 if ok else 1)
    if args.task in NO_SPARK:
        out = TASKS[args.task](None, args.snapshotdate)
    else:
        spark = bronze.get_spark(f"monthly_{args.task}")
        if args.task == "backfill_data":                       # one Spark session for a range of months
            out = []
            for m in pd.date_range(args.start, args.end, freq="MS"):
                for t in ["bronze", "silver", "gold_features", "gold_labels"]:
                    r = TASKS[t](spark, m)
                    print(m.strftime("%Y-%m"), t, r if isinstance(r, dict) else r.to_dict(orient="records"))
        else:
            if args.task not in ("monitoring_reference", "full_data_validation", "register") and not args.snapshotdate:
                parser.error("--snapshotdate is required")
            out = TASKS[args.task](spark, args.snapshotdate)
    print(json.dumps(out if isinstance(out, dict) else (out.to_dict(orient="records") if hasattr(out, "to_dict") else str(out)),
                     default=str)[:4000])


if __name__ == "__main__":
    main()
