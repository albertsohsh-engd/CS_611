"""ML deployment — section 05: steps 33 (model registry), 34 (batch inference), 35 (monitoring), 36 (governance).

Converted from notebooks/ml_deployment.ipynb (the notebook is the development version with explanations and charts
shown inline; this module is the callable version used by script/mlpipeline.ipynb). The code of each step is the
notebook's code unchanged, wrapped in one function per step; results are kept as module attributes (e.g.
``registry``, ``inference_log``, ``scored``).

Steps
    33 register the signed-off models in MLflow (SQLite store datamart/ml/mlflow/), aliases @champion / @challenger /
       @baseline; from the approved step-30 sign-off and the step-31 evaluation; idempotent
    34 monthly batch inference from models:/loan_default_30dpd_6mob@champion -> datamart/gold/model_predictions/
    35 monitoring replayed on the months after training: input / score drift (PSI), data quality, decisions, and
       performance once labels mature (6 months); alert log
    36 governance policy, refresh triggers applied to the monitoring results, rollback drill on the registry,
       deployment options

Usage:
    from util import dataprocessing_bronze as bronze, ml_deployment as dep
    spark = bronze.get_spark()
    dep.run(spark)                       # steps 33 -> 36
    dep.score_month("2025-01-01")        # score one month with @champion (after run)

Command line:
    python util/ml_deployment.py
"""
import os, sys, json, hashlib, subprocess, warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import joblib

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
import mlflow
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)
from util import dataprocessing_bronze as bronze
from util import dataprocessing_gold as gold
from util import ml_feature_engineering as fe

MD_DIR = os.path.join(PROJECT_DIR, "datamart", "ml", "model_development", "v1")      # section 04 outputs
FINAL_DIR = os.path.join(MD_DIR, "final")
DEP_VERSION = "v1"
DEP_DIR = os.path.join(PROJECT_DIR, "datamart", "ml", "deployment", DEP_VERSION)
MLFLOW_DIR = os.path.join(PROJECT_DIR, "datamart", "ml", "mlflow")
for d in (DEP_DIR, MLFLOW_DIR):
    os.makedirs(d, exist_ok=True)

REGISTERED_MODEL = "loan_default_30dpd_6mob"
EXPERIMENT = "cs611_a2_loan_default"
TRACKING_URI = "sqlite:///" + Path(MLFLOW_DIR, "mlflow.db").as_posix()             # local store, same on Windows / Linux
ARTIFACT_URI = Path(MLFLOW_DIR, "artifacts").as_uri()

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

print("MLflow", mlflow.__version__, "| tracking store:", TRACKING_URI)

import matplotlib
SHOW_CHARTS = False                    # True when called from a notebook that should display the charts


def _show():
    import matplotlib.pyplot as plt
    if SHOW_CHARTS:
        plt.show()
    plt.close("all")


def display(*args, **kwargs):          # notebook display() -> no-op in the module (results are files / attributes)
    if SHOW_CHARTS:
        try:
            from IPython.display import display as _d
            _d(*args, **kwargs)
        except ImportError:
            pass


spark = None


def step33():
    """Step 33 — Register the signed-off models (MLflow model registry)"""
    global APPROVAL, ARTIFACTS, A_RAW_LINEAR, BASELINE_SUMMARY, CONTRACT_VERSION, EVAL, EVAL_NAME, FEATURE_SETS
    global TO_REGISTER, TRUSTED_PACKAGES, X_ex, action, afe, alias, c, cards, check_X, check_ids, checks
    global client, col, ctx, existing, expected, f, feats, features_all, files, final_eval, fn, git_commit
    global group, holdout_pred, hp, info, k, metrics_for, model_sha, mv, n_versions, ok_ref, out, p, p_py, p_sk
    global params, pipe, py, record, reg_checks, reg_rows, registry, run, signature, sk, skops, spec, trusted
    global uri, v
    # 1. Inputs: sign-off approval, final evaluation and baseline summary from section 04
    with open(os.path.join(FINAL_DIR, "signoff_approval.yaml"), encoding="utf-8") as f:
        APPROVAL = yaml.safe_load(f)
    with open(os.path.join(MD_DIR, "final_evaluation.yaml"), encoding="utf-8") as f:
        EVAL = yaml.safe_load(f)
    final_eval = pd.read_csv(os.path.join(MD_DIR, "final_evaluation.csv"))
    with open(os.path.join(MD_DIR, "baseline_summary.json"), encoding="utf-8") as f:
        BASELINE_SUMMARY = json.load(f)
    print(f"Sign-off approved by {APPROVAL['approved_by']}; step-31 evaluation from {EVAL['evaluated_ts']}")

    # 2. What gets registered
    cards = {}
    for fn in sorted(os.listdir(FINAL_DIR)):
        if fn.startswith("model_card_"):
            with open(os.path.join(FINAL_DIR, fn), encoding="utf-8") as f:
                c = yaml.safe_load(f)
            cards[c["role"]] = c
    FEATURE_SETS = fe.load_feature_sets()
    A_RAW_LINEAR = FEATURE_SETS["A_raw"]["linear"]
    EVAL_NAME = {"champion": "champion XGB model A", "challenger": "challenger XGB model B", "baseline": "baseline LR__A_raw"}
    TO_REGISTER = {
        "champion": dict(path=os.path.join(FINAL_DIR, cards["champion"]["artifact"].split("/")[-1]), card=cards["champion"],
                         features=cards["champion"]["features"], threshold=cards["champion"]["operating_threshold"]),
        "challenger": dict(path=os.path.join(FINAL_DIR, cards["challenger"]["artifact"].split("/")[-1]), card=cards["challenger"],
                           features=cards["challenger"]["features"], threshold=cards["challenger"]["operating_threshold"]),
        "baseline": dict(path=os.path.join(MD_DIR, "artifacts", "baseline_logreg_A_raw.joblib"), card=None, features=A_RAW_LINEAR,
                         threshold=float(final_eval.query("model == 'baseline LR__A_raw'")["threshold"].iloc[0])),
    }
    pd.DataFrame([{"alias": k, "file": os.path.relpath(v["path"], MD_DIR), "features": len(v["features"]),
                   "threshold": v["threshold"], "sha256": sha256(v["path"])[:12] + "…"} for k, v in TO_REGISTER.items()])

    # 3. Example input rows (features only, no labels) for the signature, input example and the reload check
    ctx = gold.load_eda_data(spark)
    afe = pd.read_parquet(os.path.join(fe.FE_DIR, "autofe_features.parquet")).drop(columns="partition")
    features_all = ctx["data"].merge(afe, on="loan_id", how="left")
    holdout_pred = pd.read_parquet(os.path.join(MD_DIR, "final_predictions_holdout.parquet"))
    check_ids = holdout_pred[holdout_pred["partition"] == "oot"]["loan_id"].head(500)
    check_X = features_all.set_index("loan_id").loc[check_ids].reset_index()
    print(f"Reload check on {len(check_X)} OOT loans; feature table has {features_all.shape[1] - 2} columns")

    # 4. Register (idempotent: an existing version with the same model checksum is reused)
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_registry_uri(TRACKING_URI)
    client = MlflowClient()
    if client.get_experiment_by_name(EXPERIMENT) is None:
        client.create_experiment(EXPERIMENT, artifact_location=ARTIFACT_URI)
    mlflow.set_experiment(EXPERIMENT)

    def git_commit():
        try:
            return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_DIR,
                                           stderr=subprocess.DEVNULL).decode().strip()
        except Exception:
            return "n/a"

    with open(os.path.join(PROJECT_DIR, "config", "data_contract.yaml"), encoding="utf-8") as f:
        CONTRACT_VERSION = str(yaml.safe_load(f).get("version", "n/a"))

    def metrics_for(alias):
        m = {}
        for _, r in final_eval[final_eval["model"] == EVAL_NAME[alias]].iterrows():
            for k in ["pr_auc", "roc_auc", "brier", "precision", "recall", "f1", "flagged_share"]:
                m[f"{r['partition']}_{k}"] = float(r[k])
        m["cv_pr_auc"] = float(TO_REGISTER[alias]["card"]["cv_pr_auc"]) if TO_REGISTER[alias]["card"] \
            else float(BASELINE_SUMMARY["baseline_cv_pr_auc_mean"])
        return m

    ARTIFACTS = {"governance": [os.path.join(FINAL_DIR, f) for f in ["model_selection_signoff.yaml", "signoff_approval.yaml"]]
                               + [os.path.join(MD_DIR, f) for f in ["metric_contract.yaml", "final_evaluation.yaml",
                                                                    "error_robustness_summary.yaml", "reduced_features.json",
                                                                    "best_params.yaml"]],
                 "charts": [os.path.join(MD_DIR, "charts", f) for f in ["30_1_threshold_selection.png", "30_4_final_shap_importance.png",
                                                                         "31_1_final_pr_auc.png", "31_3_oot_by_month.png",
                                                                         "32_3_input_robustness.png"]]}

    import skops.io
    TRUSTED_PACKAGES = {"numpy", "xgboost"}            # beyond scikit-learn, only these packages' types may be trusted
    existing = {v.tags.get("model_sha256"): v for v in client.search_model_versions(f"name='{REGISTERED_MODEL}'")}
    reg_rows = []
    for alias, spec in TO_REGISTER.items():
        model_sha = sha256(spec["path"])
        pipe = joblib.load(spec["path"])
        if model_sha in existing:
            mv = existing[model_sha]
            action = "reused (same checksum)"
        else:
            X_ex = check_X[spec["features"]].head(5)
            signature = infer_signature(X_ex, pipe.predict_proba(X_ex))
            with mlflow.start_run(run_name=f"{alias}") as run:
                mlflow.set_tags({"role": alias, "label": gold.LABEL_DEF, "split_version": "v1", "feature_store": "gold v1",
                                 "feature_engineering": "v1", "data_contract_version": CONTRACT_VERSION,
                                 "git_commit": git_commit(), "model_sha256": model_sha, "checklist_step": "33",
                                 "oot_evaluable": str(alias != "challenger")})
                params = {"n_features": len(spec["features"]), "operating_threshold": spec["threshold"],
                          "threshold_rule": "max F1 on validation", "class_weighting": "none",
                          "algorithm": type(pipe.named_steps["model"]).__name__}
                if spec["card"]:
                    params.update({f"xgb_{k}": v for k, v in spec["card"]["params"].items()})
                mlflow.log_params(params)
                mlflow.log_metrics(metrics_for(alias))
                mlflow.log_dict({"features": spec["features"]}, "features.json")
                if spec["card"]:
                    mlflow.log_dict(spec["card"], "model_card.yaml")
                for group, files in ARTIFACTS.items():
                    for p in files:
                        mlflow.log_artifact(p, artifact_path=group)
                # skops format (MLflow default, safer than pickle): only explicitly trusted non-sklearn types may load
                trusted = skops.io.get_untrusted_types(data=skops.io.dumps(pipe))
                assert all(t.split(".")[0] in TRUSTED_PACKAGES for t in trusted), f"unexpected types: {trusted}"
                mlflow.set_tag("skops_trusted_types", ", ".join(trusted) or "none")
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    info = mlflow.sklearn.log_model(pipe, name="model", signature=signature, input_example=X_ex,
                                                    pyfunc_predict_fn="predict_proba", registered_model_name=REGISTERED_MODEL,
                                                    serialization_format="skops", skops_trusted_types=trusted)
            mv = client.get_model_version(REGISTERED_MODEL, info.registered_model_version)
            for k, v in {"role": alias, "model_sha256": model_sha, "operating_threshold": str(spec["threshold"]),
                         "oot_evaluable": str(alias != "challenger"), "approved_by": APPROVAL["approved_by"]}.items():
                client.set_model_version_tag(REGISTERED_MODEL, mv.version, k, v)
            action = "registered"
        client.set_registered_model_alias(REGISTERED_MODEL, alias, mv.version)
        reg_rows.append(dict(alias=alias, version=int(mv.version), action=action, run_id=mv.run_id,
                             model_uri=f"models:/{REGISTERED_MODEL}@{alias}", threshold=spec["threshold"],
                             features=len(spec["features"]), model_sha256=model_sha))
    client.update_registered_model(REGISTERED_MODEL, description=(
        "Probability that a new loan reaches 30+ days past due at month-on-book 6 (label 30dpd_6mob), scored at loan start. "
        "@champion = XGBoost without clickstream (deployable, OOT-validated); @challenger = XGBoost with clickstream "
        "(needs the clickstream feed); @baseline = logistic regression reference."))
    registry = pd.DataFrame(reg_rows)
    registry.drop(columns="model_sha256")

    # 5. Check: models loaded back from the registry reproduce the step-31 predictions; aliases point to the right versions
    checks = []
    hp = holdout_pred.set_index("loan_id").loc[check_ids]
    for alias, col in [("champion", "p_champion"), ("challenger", "p_challenger"), ("baseline", "p_baseline")]:
        uri = f"models:/{REGISTERED_MODEL}@{alias}"
        feats = TO_REGISTER[alias]["features"]
        sk = mlflow.sklearn.load_model(uri)
        p_sk = sk.predict_proba(check_X[feats])[:, 1]
        py = mlflow.pyfunc.load_model(uri)
        out = py.predict(check_X[feats])
        p_py = np.asarray(out)[:, 1] if np.asarray(out).ndim == 2 else np.asarray(out)
        expected = hp[col].values
        ok_ref = np.isnan(expected).all() or np.allclose(p_sk, expected, atol=1e-6)     # challenger has no OOT reference
        if np.isnan(expected).all():                                                   # compare with the signed-off file instead
            ok_ref = np.allclose(p_sk, joblib.load(TO_REGISTER[alias]["path"]).predict_proba(check_X[feats])[:, 1], atol=1e-9)
        mv = client.get_model_version_by_alias(REGISTERED_MODEL, alias)
        checks.append(dict(alias=alias, version=int(mv.version), sklearn_matches_signed_off=bool(ok_ref),
                           pyfunc_matches_sklearn=bool(np.allclose(p_py, p_sk, atol=1e-9)),
                           version_sha_matches=mv.tags.get("model_sha256") == sha256(TO_REGISTER[alias]["path"]),
                           signature_inputs=len(py.metadata.get_input_schema().inputs)))
    reg_checks = pd.DataFrame(checks)
    assert reg_checks[["sklearn_matches_signed_off", "pyfunc_matches_sklearn", "version_sha_matches"]].all().all()
    n_versions = len(client.search_model_versions(f"name='{REGISTERED_MODEL}'"))
    print(f"Registry check passed; '{REGISTERED_MODEL}' has {n_versions} version(s)")
    reg_checks

    # 6. Registry record for steps 34-36 (inference loads the alias, not a file path)
    record = {
        "step": 33, "version": DEP_VERSION, "registered_ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "tracking_uri": "sqlite:///datamart/ml/mlflow/mlflow.db (relative to the project folder)",
        "experiment": EXPERIMENT, "registered_model": REGISTERED_MODEL,
        "aliases": {r["alias"]: {"version": int(r["version"]), "model_uri": r["model_uri"], "run_id": r["run_id"],
                                 "operating_threshold": float(r["threshold"]), "n_features": int(r["features"]),
                                 "model_sha256": r["model_sha256"]} for r in reg_rows},
        "inference_uses": f"models:/{REGISTERED_MODEL}@champion",
        "approval": {k: APPROVAL[k] for k in ["approved_by", "approved_ts"]},
        "ui": "mlflow ui --backend-store-uri sqlite:///datamart/ml/mlflow/mlflow.db  (run from the project folder; "
              "in Docker add --host 0.0.0.0 and publish port 5000)",
    }
    with open(os.path.join(DEP_DIR, "model_registry.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(record, f, sort_keys=False)
    registry.drop(columns=["model_sha256"]).to_csv(os.path.join(DEP_DIR, "model_registry.csv"), index=False)
    print(yaml.safe_dump(record["aliases"], sort_keys=False))


def step34():
    """Step 34 — Deploy inference (batch scoring from the registry)"""
    global CATALOG, CHART_DIR, F, GRID, INFERENCE_START, INK, INK_2, PRED_DIR, SCORING_COL, SERIES, SURFACE, ax
    global axes, cmp, d, f, fig, inf_checks, inference_log, load_scoring_rows, log, logs, months, mth, out, plt
    global reason_codes, ref, resolve_alias, score_month, scored, top_reasons, v, x, xi
    from pyspark.sql import functions as F

    PRED_DIR = os.path.join(PROJECT_DIR, "datamart", "gold", "model_predictions", REGISTERED_MODEL)
    CHART_DIR = os.path.join(DEP_DIR, "charts")
    for d in (PRED_DIR, CHART_DIR):
        os.makedirs(d, exist_ok=True)
    with open(os.path.join(gold.SPLIT_DIR, "split_config.yaml"), encoding="utf-8") as f:
        INFERENCE_START = yaml.safe_load(f)["oot_start"]                 # first month after the development period
    SCORING_COL = gold.SCORING_COL
    CATALOG = fe.load_catalog()

    def resolve_alias(alias="champion"):
        # The registry decides which model runs: version, threshold and checksum come from the alias' version tags
        mv = client.get_model_version_by_alias(REGISTERED_MODEL, alias)
        uri = f"models:/{REGISTERED_MODEL}@{alias}"
        pyfunc_model = mlflow.pyfunc.load_model(uri)
        sk_model = mlflow.sklearn.load_model(uri)
        return dict(alias=alias, version=int(mv.version), uri=uri, threshold=float(mv.tags["operating_threshold"]),
                    model_sha256=mv.tags["model_sha256"], pyfunc=pyfunc_model, sklearn=sk_model,
                    features=pyfunc_model.metadata.get_input_schema().input_names())

    def load_scoring_rows(month_start):
        # Gold features of loans starting in the month (no labels, no split manifest) + generated features
        start = pd.Timestamp(month_start)
        end = start + pd.offsets.MonthBegin(1)
        fs = gold.get_features(spark)
        feats = gold.feature_list(fs)
        asof_cols = [c for c in fs.columns if c.startswith("_asof_")]
        rows = (fs.filter((F.col(SCORING_COL) >= F.lit(start.date())) & (F.col(SCORING_COL) < F.lit(end.date())))
                  .select("loan_id", "Customer_ID", SCORING_COL, *asof_cols, *feats).toPandas())
        binary = [c for c in feats if c.endswith("_invalid") or c in gold.LOAN_TYPE_COLUMNS or c == "type_of_loan_unknown"]
        for c in binary:                                   # same typing rule as training (gold.load_eda_data)
            rows[c] = rows[c].astype(float)
        return fe.apply_catalog(rows, CATALOG), asof_cols

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

    def score_month(month_start, alias="champion"):
        m = resolve_alias(alias)
        rows, asof_cols = load_scoring_rows(month_start)
        if rows.empty:
            return None, None
        X = rows[m["features"]]
        p = np.asarray(m["pyfunc"].predict(X))[:, 1]             # signature-enforced scoring path
        for c in asof_cols:                                       # as-of rule: no snapshot after the loan start
            snap = pd.to_datetime(rows[c], errors="coerce")       # missing snapshot (source absent) -> NaT
            ok = snap.isna() | (snap <= pd.to_datetime(rows[SCORING_COL]))
            assert ok.all(), f"{c}: feature snapshot after loan start"
        out = pd.DataFrame({"loan_id": rows["loan_id"], "Customer_ID": rows["Customer_ID"],
                            "loan_start_date": pd.to_datetime(rows[SCORING_COL]).dt.strftime("%Y-%m-%d"),
                            "default_probability": p, "threshold": m["threshold"],
                            "flag_high_risk": (p >= m["threshold"]).astype(int)})
        out = pd.concat([out, reason_codes(m["sklearn"], X)], axis=1)
        out = out.assign(model_name=REGISTERED_MODEL, model_alias=alias, model_version=m["version"],
                         model_sha256=m["model_sha256"], feature_store="gold v1",
                         scored_ts=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
        assert out["loan_id"].is_unique and out["default_probability"].between(0, 1).all()
        month_tag = pd.Timestamp(month_start).strftime("%Y_%m_%d")
        path = os.path.join(PRED_DIR, f"{REGISTERED_MODEL}_{alias}_predictions_{month_tag}.parquet")
        tmp = path + ".tmp"
        out.to_parquet(tmp, index=False)
        os.replace(tmp, path)                                     # atomic: a re-run replaces the month cleanly
        log = dict(month=pd.Timestamp(month_start).strftime("%Y-%m"), loans=len(out), model_alias=alias,
                   model_version=m["version"], threshold=m["threshold"], mean_probability=float(p.mean()),
                   flagged_share=float(out["flag_high_risk"].mean()), file=os.path.relpath(path, PROJECT_DIR).replace("\\", "/"),
                   file_sha256=sha256(path), scored_ts=out["scored_ts"].iloc[0])
        return out, log

    # Run the monthly batches for the period after training
    months = pd.date_range(INFERENCE_START, "2025-01-01", freq="MS")
    logs, scored = [], []
    for mth in months:
        out, log = score_month(mth)
        if out is not None:
            scored.append(out); logs.append(log)
            print(f"{log['month']}: {log['loans']:,} loans scored with @{log['model_alias']} v{log['model_version']} | "
                  f"mean p {log['mean_probability']:.3f} | flagged {log['flagged_share']:.1%}")
    scored = pd.concat(scored, ignore_index=True)
    inference_log = pd.DataFrame(logs)
    inference_log.to_csv(os.path.join(DEP_DIR, "inference_log.csv"), index=False)

    # Checks: the deployed path reproduces the evaluated champion; one row per loan; nothing missing
    ref = pd.read_parquet(os.path.join(MD_DIR, "final_predictions_holdout.parquet"))
    ref = ref[ref["partition"] == "oot"][["loan_id", "p_champion"]]
    cmp = scored.merge(ref, on="loan_id", how="outer", indicator=True)
    inf_checks = pd.DataFrame([
        dict(check="every OOT loan scored exactly once", passed=bool((cmp["_merge"] == "both").all() and scored["loan_id"].is_unique),
             observed=f"{len(scored):,} scored vs {len(ref):,} OOT loans"),
        dict(check="predictions equal the step-31 champion predictions", passed=bool(np.allclose(
            cmp["default_probability"], cmp["p_champion"], atol=1e-6)),
             observed=f"max abs diff {np.abs(cmp['default_probability'] - cmp['p_champion']).max():.2e}"),
        dict(check="probabilities valid, no missing values", passed=bool(scored["default_probability"].between(0, 1).all()
                                                                          and scored["default_probability"].notna().all()),
             observed=f"range {scored['default_probability'].min():.3f}–{scored['default_probability'].max():.3f}"),
        dict(check="threshold and version from the registry alias", passed=bool(
            (scored["threshold"] == TO_REGISTER["champion"]["threshold"]).all()
            and (scored["model_version"] == client.get_model_version_by_alias(REGISTERED_MODEL, "champion").version).all()),
             observed=f"threshold {scored['threshold'].iloc[0]}, version {scored['model_version'].iloc[0]}"),
        dict(check="flag rate matches step 31 at the same threshold", passed=bool(abs(scored["flag_high_risk"].mean()
            - final_eval.query("partition == 'oot' and model == 'champion XGB model A'")["flagged_share"].iloc[0]) < 1e-9),
             observed=f"{scored['flag_high_risk'].mean():.2%} flagged"),
    ])
    inf_checks.to_csv(os.path.join(DEP_DIR, "inference_checks.csv"), index=False)
    assert inf_checks["passed"].all(), inf_checks
    display(inf_checks.style.hide(axis="index"))
    scored[["loan_id", "loan_start_date", "default_probability", "flag_high_risk", "reason_1", "reason_2", "reason_3",
            "model_version"]].head(5).style.hide(axis="index").format({"default_probability": "{:.3f}"})

    # Chart: monthly batches — volume, mean predicted probability and share flagged
    import matplotlib.pyplot as plt
    SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
    SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "axes.titlecolor": INK, "axes.titlesize": 11,
        "axes.titleweight": "bold", "axes.titlelocation": "left", "xtick.color": INK_2, "ytick.color": INK_2,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID,
        "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
        "legend.fontsize": 8, "font.size": 9, "lines.linewidth": 2})
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    x = np.arange(len(inference_log))
    axes[0].bar(x, inference_log["loans"], color=SERIES[0], width=0.6)
    for xi, v in zip(x, inference_log["loans"]):
        axes[0].text(xi, v + 8, f"{v:,}", ha="center", fontsize=8, color=INK_2)
    axes[0].set_title("Loans scored per monthly batch")
    axes[0].set_ylabel("loans")
    axes[1].plot(x, inference_log["mean_probability"], "o-", color=SERIES[0], label="mean predicted default probability")
    axes[1].plot(x, inference_log["flagged_share"], "o-", color=SERIES[1], label=f"share flagged (p ≥ {inference_log['threshold'].iloc[0]:.3f})")
    axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    axes[1].set_ylim(0, 0.5)
    axes[1].set_title("Scores per batch")
    axes[1].legend(loc="lower right")
    for ax in axes:
        ax.set_xticks(x, inference_log["month"])
    fig.suptitle(f"Step 34 — batch inference with @champion (v{inference_log['model_version'].iloc[0]}), loans starting "
                 f"{inference_log['month'].iloc[0]} → {inference_log['month'].iloc[-1]}", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(CHART_DIR, "34_1_batch_inference.png"), dpi=130, bbox_inches="tight")
    _show()

    # Most frequent reasons behind high-risk flags (first reason code of flagged loans)
    top_reasons = (scored.loc[scored["flag_high_risk"] == 1, "reason_1"].str.replace(r" \(\+.*\)$", "", regex=True)
                   .value_counts(normalize=True).head(8).rename("share of flagged loans").to_frame())
    top_reasons.to_csv(os.path.join(DEP_DIR, "inference_top_reasons.csv"))
    print(f"Prediction files in {os.path.relpath(PRED_DIR, PROJECT_DIR)}:", len([f for f in os.listdir(PRED_DIR) if f.endswith('.parquet')]))
    top_reasons.style.format("{:.0%}")


def step35():
    """Step 35 — Monitoring (simulated production on the out-of-time months)"""
    global BoundaryNorm, CAT_COLS, ListedColormap, MON_CFG, MON_FEATURES, REF, SCORE_REF, SCORE_SPEC, VAL_REF
    global _, alert_log, alerts, average_precision_score, ax, axes, bad, band, chall, champ, champ_miss
    global champ_psi, check, cl, click_psi, cmap, cur_missing, f, f_, feat_rows, feats, fig, flagged, fn, fp
    global heat, i, j, lab, labels_all, labels_on, labels_y, log, make_bins, miss, mon_features, mon_month
    global month_rows, mth, norm, p, p_, perf, pl, pred, psi, r, ref_miss, role, rows, s, score_psi, scored_on
    global sev, shares, spec, summary_alerts, tp, train_rows, train_scores, v, val, x, xl, xm, y
    # Configuration (thresholds) and the frozen training reference
    from sklearn.metrics import average_precision_score
    MON_CFG = {"psi_bins": 10, "psi_watch": 0.10, "psi_alert": 0.25, "missing_increase_alert_pp": 10.0,
               "flagged_share_alert_pp": 10.0, "pr_auc_drop_alert": 0.05, "calibration_gap_alert_pp": 5.0,
               "recall_drop_watch_pp": 10.0, "label_delay": "6 months (label = 30+ DPD at month-on-book 6)",
               "reference": "training partition (step 14 split), features + champion scores"}
    with open(os.path.join(DEP_DIR, "monitoring_config.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(MON_CFG, f, sort_keys=False)

    champ = resolve_alias("champion")
    chall = resolve_alias("challenger")
    train_rows = fe.apply_catalog(ctx["data"][ctx["data"]["partition"] == "train"].copy(), CATALOG)
    MON_FEATURES = {"champion": champ["features"], "challenger": chall["features"]}
    CAT_COLS = set(ctx["categorical"])

    def make_bins(values):
        # decile edges from the reference (numeric) or category list (categorical); missing is its own bin
        v = values.dropna()
        if values.name in CAT_COLS:
            return {"type": "categorical", "levels": sorted(v.astype(str).unique().tolist())}
        edges = np.unique(np.quantile(v.astype(float), np.linspace(0, 1, MON_CFG["psi_bins"] + 1)))
        return {"type": "numeric", "edges": edges[1:-1].tolist()}

    def shares(values, spec):
        n = len(values)
        miss = values.isna()
        if spec["type"] == "categorical":
            s = values[~miss].astype(str)
            counts = [int((s == lv).sum()) for lv in spec["levels"]] + [int((~s.isin(spec["levels"])).sum())]   # + unseen
        else:
            idx = np.searchsorted(spec["edges"], values[~miss].astype(float), side="right")
            counts = np.bincount(idx, minlength=len(spec["edges"]) + 1).tolist()
        return np.array(counts + [int(miss.sum())], dtype=float) / max(n, 1)

    def psi(ref_share, cur_share, eps=1e-4):
        r, c = np.clip(ref_share, eps, None), np.clip(cur_share, eps, None)
        return float(np.sum((c - r) * np.log(c / r)))

    REF = {}
    for role, feats in MON_FEATURES.items():
        for f_ in feats:
            if f_ not in REF:
                spec = make_bins(train_rows[f_].rename(f_))
                REF[f_] = {**spec, "ref_share": shares(train_rows[f_], spec).tolist(),
                           "ref_missing_pct": float(train_rows[f_].isna().mean() * 100)}
    train_scores = champ["sklearn"].predict_proba(train_rows[champ["features"]])[:, 1]
    SCORE_SPEC = {"type": "numeric", "edges": np.unique(np.quantile(train_scores, np.linspace(0, 1, 11)))[1:-1].tolist()}
    SCORE_REF = shares(pd.Series(train_scores), SCORE_SPEC)
    VAL_REF = final_eval.query("partition == 'validation' and model == 'champion XGB model A'").iloc[0]
    with open(os.path.join(DEP_DIR, "monitoring_reference.json"), "w", encoding="utf-8") as f:
        json.dump({"features": REF, "score": {**SCORE_SPEC, "ref_share": SCORE_REF.tolist()},
                   "validation": {k: float(VAL_REF[k]) for k in ["pr_auc", "recall", "precision", "flagged_share"]},
                   "reference_rows": int(len(train_rows))}, f, indent=1)
    print(f"Reference: {len(train_rows):,} training loans, {len(REF)} monitored inputs "
          f"({len(MON_FEATURES['champion'])} champion, {len(set(MON_FEATURES['challenger']) - set(MON_FEATURES['champion']))} "
          f"extra challenger); validation PR-AUC {VAL_REF['pr_auc']:.3f}, flagged {VAL_REF['flagged_share']:.1%}")

    # Replay each month: unlabelled checks at scoring time, label-based checks once outcomes mature
    labels_all = (spark.read.parquet(gold.LABEL_STORE_PATH).filter("maturity_status = 'mature'")
                  .select("loan_id", "label", "expected_outcome_date").toPandas())
    def sev(value, watch, alert):
        return "ALERT" if value > alert else "WATCH" if value > watch else "OK"

    feat_rows, month_rows, alerts = [], [], []
    for log in inference_log.itertuples():
        mth = pd.Timestamp(log.month + "-01")
        rows, _ = load_scoring_rows(mth)
        pred = pd.read_parquet(os.path.join(PROJECT_DIR, log.file))
        scored_on = (mth + pd.offsets.MonthEnd(0)).strftime("%Y-%m-%d")             # batch run at month end
        # 1. data quality + input drift, both models
        for role, feats in MON_FEATURES.items():
            for f_ in feats:
                r = REF[f_]
                cur_missing = float(rows[f_].isna().mean() * 100)
                p_ = psi(np.array(r["ref_share"]), shares(rows[f_], r))
                feat_rows.append(dict(month=log.month, model=role, feature=f_, psi=p_, missing_pct=cur_missing,
                                      ref_missing_pct=r["ref_missing_pct"]))
                if role == "challenger" and f_ in MON_FEATURES["champion"]:
                    continue                                                            # reported once, under champion
                s = sev(p_, MON_CFG["psi_watch"], MON_CFG["psi_alert"])
                if s != "OK":
                    alerts.append(dict(month=log.month, available_on=scored_on, model=role, check="input drift (PSI)",
                                       item=f_, value=round(p_, 3), rule=f"> {MON_CFG['psi_watch']} watch / > {MON_CFG['psi_alert']} alert",
                                       severity=s))
                if cur_missing - r["ref_missing_pct"] > MON_CFG["missing_increase_alert_pp"]:
                    alerts.append(dict(month=log.month, available_on=scored_on, model=role, check="missing values",
                                       item=f_, value=round(cur_missing, 1),
                                       rule=f"> training {r['ref_missing_pct']:.1f}% + {MON_CFG['missing_increase_alert_pp']:.0f} pp",
                                       severity="ALERT"))
        # 2. score drift + decisions (champion)
        p = pred["default_probability"].values
        score_psi = psi(SCORE_REF, shares(pd.Series(p), SCORE_SPEC))
        flagged = float(pred["flag_high_risk"].mean())
        for check, val, s in [("score drift (PSI)", score_psi, sev(score_psi, MON_CFG["psi_watch"], MON_CFG["psi_alert"])),
                              ("flagged share", flagged, "ALERT" if abs(flagged - VAL_REF["flagged_share"]) * 100
                               > MON_CFG["flagged_share_alert_pp"] else "OK")]:
            if s != "OK":
                alerts.append(dict(month=log.month, available_on=scored_on, model="champion", check=check, item="score",
                                   value=round(val, 3), rule="see config", severity=s))
        # 3. performance once labels mature (delayed)
        lab = pred[["loan_id", "default_probability", "flag_high_risk"]].merge(labels_all, on="loan_id", how="inner")
        y, pl = lab["label"].values, lab["default_probability"].values
        tp = int(((lab["flag_high_risk"] == 1) & (y == 1)).sum()); fp = int(((lab["flag_high_risk"] == 1) & (y == 0)).sum())
        fn = int(((lab["flag_high_risk"] == 0) & (y == 1)).sum())
        labels_on = pd.to_datetime(lab["expected_outcome_date"].astype(str)).max().strftime("%Y-%m-%d")
        perf = dict(pr_auc=average_precision_score(y, pl), recall=tp / max(tp + fn, 1), precision=tp / max(tp + fp, 1),
                    actual_rate=float(y.mean()), mean_predicted=float(pl.mean()))
        perf["calibration_gap_pp"] = (perf["mean_predicted"] - perf["actual_rate"]) * 100
        for check, bad, val, s in [
            ("PR-AUC", perf["pr_auc"] < VAL_REF["pr_auc"] - MON_CFG["pr_auc_drop_alert"], perf["pr_auc"], "ALERT"),
            ("calibration gap (pp)", abs(perf["calibration_gap_pp"]) > MON_CFG["calibration_gap_alert_pp"], perf["calibration_gap_pp"], "ALERT"),
            ("recall at threshold", (VAL_REF["recall"] - perf["recall"]) * 100 > MON_CFG["recall_drop_watch_pp"], perf["recall"], "WATCH")]:
            if bad:
                alerts.append(dict(month=log.month, available_on=labels_on, model="champion", check=check, item="performance",
                                   value=round(val, 3), rule="see config", severity=s))
        month_rows.append(dict(month=log.month, loans=len(pred), scored_on=scored_on, score_psi=score_psi,
                               mean_predicted=float(p.mean()), flagged_share=flagged, labels_available_on=labels_on,
                               labelled_loans=len(lab), **{k: v for k, v in perf.items() if k != "mean_predicted"}))

    mon_features = pd.DataFrame(feat_rows)
    mon_month = pd.DataFrame(month_rows)
    alert_log = pd.DataFrame(alerts, columns=["month", "available_on", "model", "check", "item", "value", "rule", "severity"])
    mon_features.to_csv(os.path.join(DEP_DIR, "monitoring_feature_drift.csv"), index=False)
    mon_month.to_csv(os.path.join(DEP_DIR, "monitoring_report.csv"), index=False)
    alert_log.to_csv(os.path.join(DEP_DIR, "alert_log.csv"), index=False)
    print("Alerts by model / severity:", alert_log.groupby(["model", "severity"]).size().to_dict() if len(alert_log) else "none")
    mon_month[["month", "loans", "score_psi", "mean_predicted", "flagged_share", "labels_available_on", "pr_auc", "recall",
               "precision", "actual_rate", "calibration_gap_pp"]].style.hide(axis="index").format(
        {"score_psi": "{:.3f}", "mean_predicted": "{:.1%}", "flagged_share": "{:.1%}", "pr_auc": "{:.3f}", "recall": "{:.2f}",
         "precision": "{:.2f}", "actual_rate": "{:.1%}", "calibration_gap_pp": "{:+.1f}"})

    # Alert log, summarised: what monitoring would have raised, and when
    if len(alert_log):
        summary_alerts = (alert_log.groupby(["model", "check", "severity"])
                          .agg(months=("month", lambda m: ", ".join(sorted(set(m)))), items=("item", "nunique"),
                               first_available=("available_on", "min")).reset_index())
    else:
        summary_alerts = pd.DataFrame(columns=["model", "check", "severity", "months", "items", "first_available"])
    summary_alerts.to_csv(os.path.join(DEP_DIR, "alert_summary.csv"), index=False)
    summary_alerts.style.hide(axis="index")

    # Chart 1: input drift heatmap — PSI of every champion input by month (+ the challenger's clickstream inputs)
    from matplotlib.colors import ListedColormap, BoundaryNorm
    champ_psi = mon_features[mon_features["model"] == "champion"].pivot(index="feature", columns="month", values="psi")
    click_psi = mon_features[(mon_features["model"] == "challenger") & mon_features["feature"].str.startswith("fe_")] \
        .pivot(index="feature", columns="month", values="psi")
    champ_psi = champ_psi.loc[champ_psi.max(axis=1).sort_values(ascending=False).index]
    heat = pd.concat([champ_psi, click_psi.loc[sorted(click_psi.index, key=lambda c: int(c.split("_")[1]))]])
    labels_y = [f.replace("__", " · ") + ("  [challenger]" if f.startswith("fe_") else "") for f in heat.index]
    cmap = ListedColormap(["#e8f1fb", "#f6d68a", "#e34948"])
    norm = BoundaryNorm([0, MON_CFG["psi_watch"], MON_CFG["psi_alert"], 1e9], cmap.N)
    fig, ax = plt.subplots(figsize=(10, 0.22 * len(heat) + 1.6))
    ax.imshow(heat.values, aspect="auto", cmap=cmap, norm=norm)
    ax.set_xticks(range(heat.shape[1]), heat.columns)
    ax.set_yticks(range(len(heat)), labels_y, fontsize=7)
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            v = heat.values[i, j]
            ax.text(j, i, f"{v:.2f}" if v < 10 else ">10", ha="center", va="center", fontsize=6,
                    color="white" if v > MON_CFG["psi_alert"] else INK_2)
    ax.axhline(len(champ_psi) - 0.5, color=INK, linewidth=1.2)
    ax.grid(False)
    ax.set_title(f"Step 35 — input drift (PSI vs training): champion inputs above the line, challenger clickstream below\n"
                 f"blue < {MON_CFG['psi_watch']} stable · yellow watch · red > {MON_CFG['psi_alert']} alert", loc="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(CHART_DIR, "35_1_input_drift_heatmap.png"), dpi=130, bbox_inches="tight")
    _show()

    # Chart 2: score drift and decisions (available at scoring) | Chart 3: performance (available 6 months later)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
    x = np.arange(len(mon_month)); xl = mon_month["month"]
    axes[0].plot(x, mon_month["score_psi"], "o-", color=SERIES[0], label="score PSI")
    axes[0].axhline(MON_CFG["psi_watch"], color=SERIES[3], linestyle="--", linewidth=1, label="watch 0.10")
    axes[0].axhline(MON_CFG["psi_alert"], color="#e34948", linestyle=":", linewidth=1.3, label="alert 0.25")
    axes[0].set_ylim(0, 0.36); axes[0].set_title("Score drift (at scoring)"); axes[0].legend(loc="upper left", ncol=3)
    axes[1].plot(x, mon_month["flagged_share"], "o-", color=SERIES[1], label="share flagged")
    band = MON_CFG["flagged_share_alert_pp"] / 100
    axes[1].axhspan(VAL_REF["flagged_share"] - band, VAL_REF["flagged_share"] + band, color=GRID, alpha=0.6,
                    label=f"validation {VAL_REF['flagged_share']:.0%} ± {band:.0%}")
    axes[1].plot(x, mon_month["mean_predicted"], "o-", color=SERIES[0], label="mean predicted")
    axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    axes[1].set_ylim(0, 0.75); axes[1].set_title("Decisions (at scoring)"); axes[1].legend(loc="upper left", ncol=2)
    axes[2].plot(x, mon_month["pr_auc"], "o-", color=SERIES[0], label="PR-AUC")
    axes[2].axhline(VAL_REF["pr_auc"], color=INK_2, linestyle="--", linewidth=1, label=f"validation {VAL_REF['pr_auc']:.3f}")
    axes[2].axhline(VAL_REF["pr_auc"] - MON_CFG["pr_auc_drop_alert"], color="#e34948", linestyle=":", linewidth=1.3,
                    label="alert line (−0.05)")
    axes[2].plot(x, mon_month["recall"], "o-", color=SERIES[2], label="recall at threshold")
    axes[2].set_ylim(0.4, 0.85); axes[2].set_title("Performance (labels arrive ~6 months later)")
    axes[2].legend(loc="lower left")
    for ax in axes:
        ax.set_xticks(x, xl, fontsize=7.5)
    axes[2].set_xticks(x, [f"{m}\n→ {d[:7]}" for m, d in zip(xl, mon_month["labels_available_on"])], fontsize=7)
    fig.suptitle("Step 35 — monitoring the champion month by month (simulated production 2024-07 → 2025-01)", x=0.01,
                 ha="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(CHART_DIR, "35_2_score_and_performance.png"), dpi=130, bbox_inches="tight")
    _show()

    # Chart 4: what monitoring catches for the challenger — clickstream inputs missing in production
    cl = mon_features[(mon_features["model"] == "challenger") & mon_features["feature"].str.startswith("fe_")]
    miss = cl.groupby("month")["missing_pct"].mean()
    ref_miss = cl.groupby("month")["ref_missing_pct"].mean().iloc[0]
    champ_miss = mon_features[mon_features["model"] == "champion"].groupby("month")["missing_pct"].mean()
    fig, ax = plt.subplots(figsize=(9, 3.6))
    xm = np.arange(len(miss))
    ax.plot(xm, champ_miss.values, "o-", color=SERIES[0], label="champion inputs (mean missing %)")
    ax.plot(xm, miss.values, "o-", color="#e34948", label="challenger clickstream inputs (mean missing %)")
    ax.axhline(ref_miss, color=INK_2, linestyle="--", linewidth=1, label=f"clickstream missing in training {ref_miss:.0f}%")
    ax.set_xticks(xm, miss.index); ax.set_ylim(-5, 105); ax.set_ylabel("missing values, %")
    ax.legend(loc="center right", fontsize=8)
    ax.set_title("Step 35 — data-quality alert: the challenger's clickstream feed is empty in production", loc="left",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(CHART_DIR, "35_3_challenger_clickstream_missing.png"), dpi=130, bbox_inches="tight")
    _show()


def step36():
    """Step 36 — Model governance, refresh SOP, rollback and deployment options"""
    global GOV, GOV_LOG, REFRESH_DECISION, _, action, any_flag, baseline_v, champ_drift, champ_triggered, click
    global click_missing_up, dec_rows, drill_rows, evidence, f, flags, gov_record, k, last_label, log_event
    global max_run, mm, model, need, next_refit, p0, p1, p2, rollback_drill, rows_s, rule, rules, run_len
    global sample_month, scores_now, status, trigger_decisions, v, v0, v1, v2
    # 1. Governance policy (machine-readable; read by the trigger evaluation below)
    GOV = {
        "model": REGISTERED_MODEL, "risk_tier": "high (credit decisioning)",
        "roles": {"model_owner": "albertsoh", "independent_validator": "to be assigned (e.g. course reviewer)",
                  "approver": "risk / credit committee", "platform": "ML platform team"},
        "scheduled_refit_months": 6, "training_window_months": 18,
        "triggers": {
            "input_drift": {"metric": "PSI (champion inputs)", "threshold": MON_CFG["psi_alert"], "consecutive_months": 2, "action": "refit"},
            "score_drift": {"metric": "score PSI", "threshold": MON_CFG["psi_alert"], "consecutive_months": 2, "action": "refit"},
            "performance": {"metric": "PR-AUC below validation - 0.05", "consecutive_months": 2, "action": "refit"},
            "calibration": {"metric": "|mean predicted - actual| > 5 pp", "consecutive_months": 2, "action": "refit"},
            "data_source": {"metric": "missing values +10 pp / schema change on a model input", "consecutive_months": 1, "action": "rebuild"},
        },
        "single_month_alert": "watch: investigate and record, no model change",
        "promotion_gates": ["pipeline + stress tests pass", "PR-AUC not lower than champion by > 0.01 on newest held-out months",
                            "calibration gap within +-5 pp", "no new segment weakness (step 32 checks)",
                            "shadow scoring >= 1 month", "model card + approver sign-off"],
        "rollback": "move @champion to the previous version; log; investigate",
        "retention_years": 7,
    }
    with open(os.path.join(DEP_DIR, "governance_policy.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(GOV, f, sort_keys=False, allow_unicode=True)

    GOV_LOG = os.path.join(DEP_DIR, "governance_log.csv")
    def log_event(event, detail, actor="platform (pipeline)"):
        row = pd.DataFrame([dict(ts=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), actor=actor, event=event,
                                 detail=detail)])
        row.to_csv(GOV_LOG, mode="a", header=not os.path.exists(GOV_LOG), index=False)
    print("Policy saved; triggers:", list(GOV["triggers"]))

    # 2. Apply the refresh triggers to the step-35 monitoring results (champion and challenger)
    def max_run(flags):
        best = cur = 0
        for f in flags:
            cur = cur + 1 if f else 0
            best = max(best, cur)
        return best

    mm = mon_month.sort_values("month")
    champ_drift = mon_features[mon_features["model"] == "champion"].groupby("month")["psi"].max().reindex(mm["month"])
    click = mon_features[(mon_features["model"] == "challenger") & mon_features["feature"].str.startswith("fe_")]
    click_missing_up = (click["missing_pct"] - click["ref_missing_pct"]).groupby(click["month"]).max().reindex(mm["month"])
    rules = [
        ("champion", "input_drift", (champ_drift > GOV["triggers"]["input_drift"]["threshold"]).tolist(),
         f"max input PSI per month {champ_drift.min():.3f}–{champ_drift.max():.3f}"),
        ("champion", "score_drift", (mm["score_psi"] > GOV["triggers"]["score_drift"]["threshold"]).tolist(),
         f"score PSI {mm['score_psi'].min():.3f}–{mm['score_psi'].max():.3f}"),
        ("champion", "performance", (mm["pr_auc"] < VAL_REF["pr_auc"] - MON_CFG["pr_auc_drop_alert"]).tolist(),
         "PR-AUC below alert line in: " + (", ".join(mm.loc[mm["pr_auc"] < VAL_REF["pr_auc"] - MON_CFG["pr_auc_drop_alert"], "month"]) or "none")),
        ("champion", "calibration", (mm["calibration_gap_pp"].abs() > MON_CFG["calibration_gap_alert_pp"]).tolist(),
         f"calibration gap {mm['calibration_gap_pp'].min():+.1f} to {mm['calibration_gap_pp'].max():+.1f} pp"),
        ("challenger", "data_source", (click_missing_up > MON_CFG["missing_increase_alert_pp"]).tolist(),
         f"clickstream missing +{click_missing_up.max():.0f} pp vs training"),
    ]
    dec_rows = []
    for model, rule, flags, evidence in rules:
        need = GOV["triggers"][rule]["consecutive_months"]
        run_len, any_flag = max_run(flags), any(flags)
        status = "TRIGGERED" if run_len >= need else "WATCH (single month)" if any_flag else "not triggered"
        action = GOV["triggers"][rule]["action"] if status == "TRIGGERED" else ("investigate, no model change" if any_flag else "none")
        dec_rows.append(dict(model=model, trigger=rule, rule=f"{GOV['triggers'][rule]['metric']} for {need}+ consecutive month(s)",
                             months_flagged=int(sum(flags)), longest_run=run_len, status=status, action=action, evidence=evidence))
    trigger_decisions = pd.DataFrame(dec_rows)
    trigger_decisions.to_csv(os.path.join(DEP_DIR, "refresh_trigger_decisions.csv"), index=False)

    champ_triggered = trigger_decisions.query("model == 'champion' and status == 'TRIGGERED'")
    last_label = mm["labels_available_on"].max()
    next_refit = (pd.Timestamp(last_label) + pd.DateOffset(months=0)).strftime("%Y-%m-%d")
    REFRESH_DECISION = {
        "champion": ("REFIT now (triggered)" if len(champ_triggered) else
                     f"no early refit — keep @champion v{champ['version']}; investigate the single-month PR-AUC alert; "
                     f"scheduled refit when the newest labels are complete ({next_refit})"),
        "challenger": "not deployable while the clickstream feed is missing (rebuild trigger) — stays challenger, re-assess if the feed returns",
    }
    log_event("refresh review", f"champion: {REFRESH_DECISION['champion']} | challenger: {REFRESH_DECISION['challenger']}")
    display(trigger_decisions.style.hide(axis="index"))
    for k, v in REFRESH_DECISION.items():
        print(f"{k}: {v}")

    # 3. Rollback drill on the real registry: promote another version, check inference follows the alias, roll back
    drill_rows = []
    sample_month = inference_log["month"].iloc[-1] + "-01"
    rows_s, _ = load_scoring_rows(sample_month)
    def scores_now():
        m = resolve_alias("champion")
        return m["version"], np.asarray(m["pyfunc"].predict(rows_s[m["features"]]))[:, 1]

    v0, p0 = scores_now()
    baseline_v = int(client.get_model_version_by_alias(REGISTERED_MODEL, "baseline").version)
    try:
        client.set_registered_model_alias(REGISTERED_MODEL, "champion", baseline_v)          # simulated bad promotion
        log_event("DRILL promote", f"@champion v{v0} -> v{baseline_v} (rollback drill)")
        v1, p1 = scores_now()
        drill_rows.append(dict(stage="after simulated promotion", champion_version=v1,
                               scores_changed=bool(not np.allclose(p0, p1)), note="inference follows the alias, no code change"))
    finally:
        client.set_registered_model_alias(REGISTERED_MODEL, "champion", v0)                  # rollback = move alias back
        log_event("DRILL rollback", f"@champion restored to v{v0}")
    v2, p2 = scores_now()
    drill_rows.append(dict(stage="after rollback", champion_version=v2, scores_changed=bool(not np.allclose(p0, p2)),
                           note="identical to before the drill" if np.allclose(p0, p2) else "MISMATCH"))
    rollback_drill = pd.DataFrame(drill_rows)
    rollback_drill.to_csv(os.path.join(DEP_DIR, "rollback_drill.csv"), index=False)
    assert v2 == v0 and np.array_equal(p0, p2), "rollback drill failed to restore the champion"
    assert rollback_drill.iloc[0]["champion_version"] == baseline_v and rollback_drill.iloc[0]["scores_changed"]
    print(f"Rollback drill passed on {len(rows_s)} loans from {sample_month[:7]}: alias move took effect immediately and "
          f"rollback restored v{v0} exactly")
    rollback_drill.style.hide(axis="index")

    # 4. Governance record for the step: what was decided, with evidence
    gov_record = {
        "step": 36, "decided_ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "policy": "datamart/ml/deployment/v1/governance_policy.yaml",
        "refresh_decision": REFRESH_DECISION,
        "triggers": trigger_decisions[["model", "trigger", "status", "action"]].to_dict(orient="records"),
        "rollback_drill": "passed (alias move takes effect immediately; rollback restores the champion exactly)",
        "deployment": {"chosen": "batch scoring from the registry alias, daily schedule recommended",
                       "later_option": "real-time API (mlflow models serve / FastAPI) once features are available online",
                       "not_needed": ["streaming", "embedded / in-database"]},
        "champion": {"alias": "@champion", "version": int(champ["version"]), "threshold": float(champ["threshold"])},
        "approval_required_for": ["any alias move to a new version", "policy changes"],
    }
    with open(os.path.join(DEP_DIR, "governance_decision.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(gov_record, f, sort_keys=False, allow_unicode=True)
    pd.read_csv(GOV_LOG).tail(6).style.hide(axis="index").set_caption("governance_log.csv (latest entries)")


STEPS = ['step33', 'step34', 'step35', 'step36']


def run(spark_session=None, steps=None, show_charts=False):
    """Run the deployment steps in order (default: 33 -> 36). Needs the section-04 outputs: signed-off models,
    the approval and the step-31 evaluation."""
    global spark, SHOW_CHARTS
    SHOW_CHARTS = show_charts
    if not show_charts:
        matplotlib.use("Agg")
    spark = spark_session or bronze.get_spark("ml_deployment")
    for name in (steps or STEPS):
        t0 = datetime.now()
        print(f"--- {name} ---")
        globals()[name]()
        print(f"--- {name} done in {(datetime.now() - t0).total_seconds():.0f}s")
    return sys.modules[__name__]


if __name__ == "__main__":
    run()
