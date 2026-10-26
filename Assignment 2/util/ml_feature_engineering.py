"""ML feature engineering — section 03: step 21 (AutoFE with OpenFE), step 23 (initial feature filtering),
step 18 (train-fitted preprocessing). Steps 19, 20, 22 are not applied (reasons in the notebook).

Converted from notebooks/ml_feature_engineering.ipynb (the notebook is the development version; this module is the
callable version used by script/mlpipeline.ipynb and, later, inference).

Usage:
    from util import dataprocessing_bronze as bronze, ml_feature_engineering as fe
    spark = bronze.get_spark()
    out = fe.run(spark)                               # 21 -> 23 -> 18, with tests
    X = fe.apply_catalog(df, fe.load_catalog())       # inference: add the generated features (no fitting)
    pre = fe.load_preprocessor("A_raw", "tree")       # fitted on train; .transform() only

Command line:
    python util/ml_feature_engineering.py [--no-charts] [--no-tests]
"""
import os
import sys
import json
import copy
import time
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from util import dataprocessing_bronze as bronze
from util import dataprocessing_silver as silver
from util import dataprocessing_gold as gold


# =============================================================================================
# Configuration
# =============================================================================================
PROJECT_DIR = bronze.PROJECT_DIR
FE_VERSION = "v1"
FE_DIR = os.path.join(PROJECT_DIR, "datamart", "ml", "feature_engineering", FE_VERSION)
CHART_DIR = os.path.join(FE_DIR, "charts")
TMP_DIR = os.path.join(PROJECT_DIR, "datamart", "ml", "_tmp")
PREP_DIR = os.path.join(PROJECT_DIR, "datamart", "ml", "preprocessing", FE_VERSION)

# Step 21
MAX_FEATURES = 20                 # configurable bound on generated features kept
SEED = 42
N_JOBS = 4
DUPLICATE_RHO = 0.98              # |Spearman| at or above this with an existing / earlier feature = duplicate formula
OP_NAME = {"+": "add", "-": "sub", "*": "mul", "/": "div", "min": "min", "max": "max"}

# Step 23
FILTER = dict(missing_pct_max=50.0, near_constant_share=0.995, leakage_auc=0.85,
              near_duplicate_rho=0.95, correlation_report_rho=0.8, vif_max=10.0)
PREFER_DROP = {"Num_of_Loan": "num_loan_types", "Annual_Income": "Monthly_Inhand_Salary"}   # step-16 register
MODEL_TYPES = ["tree", "linear"]

CAT_LEVELS = {c: lv for (t, c), lv in silver.VALID_CATEGORIES.items()}
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834"]


def get_spark(app_name="ml_feature_engineering"):
    return bronze.get_spark(app_name)


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


# =============================================================================================
# Shared helpers (also used at inference)
# =============================================================================================
def encode_for_openfe(df, base, categorical):
    # Categories -> fixed integer codes from the Silver allowed sets (a fixed rule, nothing learned)
    out = df[base].copy()
    for c in categorical:
        out[c] = pd.Categorical(out[c], categories=CAT_LEVELS[c]).codes.astype(float)
        out.loc[out[c] < 0, c] = np.nan
    return out.astype(float)


def apply_formula(df, operator, a, b):
    # Row-wise, no fitted statistics: identical on every partition and at inference
    x, y = df[a].astype(float), df[b].astype(float)
    if operator == "+":   return x + y
    if operator == "-":   return x - y
    if operator == "*":   return x * y
    if operator == "/":   return x / y.replace(0, np.nan)          # zero denominator -> null
    if operator == "min": return np.minimum(x, y)
    if operator == "max": return np.maximum(x, y)
    raise ValueError(operator)


def load_catalog():
    return pd.read_csv(os.path.join(FE_DIR, "autofe_catalog.csv"))


def apply_catalog(df, catalog):
    # Adds every generated feature in the catalog to df (inputs are numeric model-A features)
    out = df.copy()
    for _, r in catalog.iterrows():
        out[r["name"]] = apply_formula(out, r["operator"], r["feature_1"], r["feature_2"]).values
    return out


def load_feature_sets():
    # {feature_set: {"tree": [...], "linear": [...]}} — model-specific filtered lists (step 23, V6.6)
    with open(os.path.join(FE_DIR, "filtered_features.json"), encoding="utf-8") as f:
        return json.load(f)["sets"]


def load_preprocessor(feature_set, model_type):
    import joblib
    return joblib.load(os.path.join(PREP_DIR, f"preprocess__{feature_set}__{model_type}.joblib"))


# =============================================================================================
# Step 21 — AutoFE with OpenFE (train only, row-wise operators)
# =============================================================================================
def run_step21(ctx, charts=True):
    from openfe import OpenFE
    from openfe.openfe import get_candidate_features, num_num_operators
    from openfe.utils import tree_to_formula

    os.makedirs(FE_DIR, exist_ok=True)
    os.makedirs(TMP_DIR, exist_ok=True)
    data, train = ctx["data"], ctx["train"]
    base = ctx["features_a"]
    numeric_a = [c for c in ctx["numeric"] if c in base]
    categorical_a = [c for c in ctx["categorical"] if c in base]

    X_train = encode_for_openfe(train, base, categorical_a).reset_index(drop=True)
    y_train = train[["label"]].reset_index(drop=True)
    candidates = [f for f in get_candidate_features(numerical_features=numeric_a, order=1) if f.name in num_num_operators]
    gold.check_predictors(base)

    t0 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ranked = OpenFE().fit(data=X_train, label=y_train, task="classification", candidate_features_list=candidates,
                              categorical_features=categorical_a, n_jobs=N_JOBS, seed=SEED, verbose=False,
                              tmp_save_path=os.path.join(TMP_DIR, "openfe_tmp.feather"))
    gen_seconds = time.time() - t0

    def describe(node):
        a, b = node.children[0].name, node.children[1].name
        return dict(operator=node.name, feature_1=a, feature_2=b, formula=tree_to_formula(node),
                    name=f"afe_{OP_NAME[node.name]}__{a}__{b}")

    ranking = pd.DataFrame([dict(rank=i + 1, **describe(n)) for i, n in enumerate(ranked)])
    ranking.to_csv(os.path.join(FE_DIR, "autofe_ranking_all.csv"), index=False)

    # Test: our implementation reproduces OpenFE's calculation
    for n in ranked[:MAX_FEATURES * 2]:
        d = describe(n)
        ours = apply_formula(X_train, d["operator"], d["feature_1"], d["feature_2"])
        theirs = pd.Series(np.asarray(n.calculate(X_train), dtype=float))
        assert np.allclose(ours.fillna(-9e99), theirs.fillna(-9e99)), d["formula"]
        n.delete()

    # Keep the top MAX_FEATURES, discarding duplicate formulas
    kept, kept_ops, rows = [], {}, []
    base_numeric = X_train[[c for c in base if X_train[c].nunique() > 2]]
    for _, r in ranking.iterrows():
        if len(kept) >= MAX_FEATURES:
            break
        v = apply_formula(X_train, r["operator"], r["feature_1"], r["feature_2"])
        reason = ""
        if v.nunique(dropna=True) <= 1:
            reason = "constant"
        else:
            rho_base = base_numeric.corrwith(v, method="spearman").abs()
            rho_kept = pd.Series({k: v.corr(apply_formula(X_train, *kept_ops[k]), method="spearman")
                                  for k in kept}, dtype=float).abs()
            if rho_base.max() >= DUPLICATE_RHO:
                reason = f"duplicate of base feature {rho_base.idxmax()} (rho {rho_base.max():.3f})"
            elif len(rho_kept) and rho_kept.max() >= DUPLICATE_RHO:
                reason = f"duplicate of {rho_kept.idxmax()} (rho {rho_kept.max():.3f})"
        if not reason:
            kept.append(r["name"])
            kept_ops[r["name"]] = (r["operator"], r["feature_1"], r["feature_2"])
        rows.append(dict(**r, kept=not reason, discard_reason=reason))
    screen = pd.DataFrame(rows)

    # Catalog with train-only strength
    y = train["label"].reset_index(drop=True)
    catalog = screen[screen["kept"]].copy().reset_index(drop=True)
    catalog.insert(0, "feature_id", [f"AFE-{i + 1:02d}" for i in range(len(catalog))])
    iv, auc, nulls = [], [], []
    for _, r in catalog.iterrows():
        v = apply_formula(X_train, r["operator"], r["feature_1"], r["feature_2"])
        iv.append(gold.woe_iv(v, y, "numeric")[1])
        auc.append(gold.univariate_auc(v, y, "numeric")[0])
        nulls.append(v.isna().mean() * 100)
    catalog["iv_train"], catalog["auc_train"], catalog["null_pct_train"] = iv, auc, nulls
    catalog["iv_best_input"] = [max(gold.woe_iv(train[a].reset_index(drop=True), y, "numeric")[1],
                                    gold.woe_iv(train[b].reset_index(drop=True), y, "numeric")[1])
                                for a, b in zip(catalog["feature_1"], catalog["feature_2"])]
    catalog["iv_gain_vs_inputs"] = catalog["iv_train"] - catalog["iv_best_input"]
    catalog["version_added"] = FE_VERSION
    catalog["zero_denominator"] = np.where(catalog["operator"] == "/", "null", "n/a")
    catalog = catalog.drop(columns=["kept", "discard_reason"])
    catalog.to_csv(os.path.join(FE_DIR, "autofe_catalog.csv"), index=False)
    screen.to_csv(os.path.join(FE_DIR, "autofe_screening.csv"), index=False)

    # Candidate matrix: formulas applied to every partition (no fitting)
    X_all = encode_for_openfe(data, base, categorical_a)
    gen = pd.DataFrame({r["name"]: apply_formula(X_all, r["operator"], r["feature_1"], r["feature_2"]).values
                        for _, r in catalog.iterrows()}, index=data.index)
    matrix = pd.concat([data[["loan_id", "partition"]], gen], axis=1)
    matrix.to_parquet(os.path.join(FE_DIR, "autofe_features.parquet"), index=False)

    config = dict(version=FE_VERSION, tool="openfe 0.0.12", base_feature_set="model A (no clickstream)",
                  base_features=len(base), numeric_base_features=len(numeric_a), operators=sorted(num_num_operators),
                  order=1, candidates=len(candidates), ranked=len(ranked), max_features=MAX_FEATURES,
                  kept=int(len(catalog)), duplicate_rho=DUPLICATE_RHO, seed=SEED, rows="train only",
                  train_rows=int(len(train)), generation_seconds=round(gen_seconds, 1),
                  created_ts=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
    with open(os.path.join(FE_DIR, "autofe_config.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    if charts:
        plot_step21(catalog)
    return dict(catalog=catalog, ranking=ranking, screening=screen, matrix=matrix, config=config)


def plot_step21(catalog):
    plt = _plt()
    rel = pd.read_csv(os.path.join(gold.EDA_DIR, "feature_target_relationships.csv"))
    orig = rel[rel["in_set_A"]].nlargest(10, "iv")[["feature", "iv"]].assign(kind="original")
    genr = (catalog.nlargest(15, "iv_train")[["formula", "iv_train"]]
            .rename(columns={"formula": "feature", "iv_train": "iv"}).assign(kind="generated (OpenFE)"))
    both = pd.concat([orig, genr]).sort_values("iv")
    fig, ax = plt.subplots(figsize=(11, 0.32 * len(both) + 1.4))
    ax.barh(both["feature"], both["iv"], color=[SERIES[0] if k == "original" else SERIES[1] for k in both["kind"]], height=0.7)
    for yv, v in enumerate(both["iv"]):
        ax.text(v + 0.005, yv, f"{v:.2f}", va="center", fontsize=7.5, color=INK_2)
    ax.legend([plt.Rectangle((0, 0), 1, 1, color=c) for c in SERIES],
              ["original feature (top 10)", "generated by OpenFE (top 15)"], loc="lower right")
    ax.set_xlabel("information value (train only)")
    ax.grid(axis="y", visible=False)
    ax.set_title("Step 21 — OpenFE generated features vs strongest original features")
    fig.tight_layout()
    os.makedirs(CHART_DIR, exist_ok=True)
    fig.savefig(os.path.join(CHART_DIR, "21_autofe_vs_original_iv.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)


# =============================================================================================
# Step 23 — Initial feature filtering (V6.6: train-only screens, flag rather than drop,
#           model-specific keep/drop, validation impact)
# =============================================================================================
def build_full(ctx, matrix):
    full = ctx["data"].merge(matrix.drop(columns="partition"), on="loan_id", how="left")
    train_full = full[full["partition"] == "train"].merge(ctx["train"][["loan_id", "label"]], on="loan_id")
    assert len(train_full) == len(ctx["train"])
    return full, train_full


def run_step23(spark, ctx, catalog, matrix, charts=True):
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    base, click, cat, binary = ctx["features_a"], ctx["click"], ctx["categorical"], ctx["binary"]
    afe = list(catalog["name"])
    pool = base + afe + click
    set_candidates = {"A_raw": base, "A_fe": base + afe, "B_raw": base + click, "B_fe": base + afe + click}
    full, train_full = build_full(ctx, matrix)
    y_tr = train_full["label"].reset_index(drop=True)

    def origin(c):
        return "generated" if c in afe else "clickstream" if c in click else "original"

    # --- per-feature screens (train only) ---
    rows = []
    for c in pool:
        s = train_full[c].reset_index(drop=True)
        kind = "categorical" if c in cat else "binary" if c in binary else "numeric"
        rows.append(dict(feature=c, origin=origin(c), kind=kind, missing_pct=s.isna().mean() * 100,
                         top_share=s.value_counts(normalize=True, dropna=False).iloc[0],
                         iv=gold.woe_iv(s, y_tr, kind)[1], auc=gold.univariate_auc(s, y_tr, kind)[0]))
    report = pd.DataFrame(rows).set_index("feature")
    report["flag_missing"] = report["missing_pct"] > FILTER["missing_pct_max"]
    report["flag_low_variance"] = report["top_share"] >= FILTER["near_constant_share"]
    report["flag_leakage"] = report["auc"].fillna(0) > FILTER["leakage_auc"]
    hashes, dup_of = {}, {}
    for c in pool:
        key = pd.util.hash_pandas_object(train_full[c].astype(str), index=False).sum()
        if key in hashes:
            dup_of[c] = hashes[key]
        else:
            hashes[key] = c
    report["exact_duplicate_of"] = pd.Series(dup_of)
    report["flag_exact_duplicate"] = report["exact_duplicate_of"].notna()

    # --- correlation report ---
    num_like = [c for c in pool if c not in cat and train_full[c].nunique() > 1]
    rho = train_full[num_like].corr(method="spearman")
    pairs = (rho.where(np.triu(np.ones(rho.shape, dtype=bool), k=1)).stack().rename("spearman").reset_index()
             .rename(columns={"level_0": "feature_1", "level_1": "feature_2"}))
    pairs["abs_rho"] = pairs["spearman"].abs()
    corr_report = pairs[pairs["abs_rho"] >= FILTER["correlation_report_rho"]].sort_values("abs_rho", ascending=False)
    corr_report = corr_report.assign(origin_1=corr_report["feature_1"].map(origin),
                                     origin_2=corr_report["feature_2"].map(origin),
                                     near_duplicate=corr_report["abs_rho"] >= FILTER["near_duplicate_rho"])
    corr_report.to_csv(os.path.join(FE_DIR, "correlation_report.csv"), index=False)
    max_corr = pairs.assign(a=pairs["feature_1"], b=pairs["feature_2"])
    partner = pd.concat([max_corr[["a", "b", "abs_rho"]], max_corr.rename(columns={"a": "b", "b": "a"})[["a", "b", "abs_rho"]]])
    best = partner.sort_values("abs_rho", ascending=False).drop_duplicates("a").set_index("a")
    report["max_abs_rho"] = best["abs_rho"]
    report["max_rho_partner"] = best["b"]
    report["flag_near_duplicate"] = report["max_abs_rho"].fillna(0) >= FILTER["near_duplicate_rho"]

    # --- ACF: not applicable ---
    pd.DataFrame([dict(scope="all features", applicable=False,
                       reason="one row per loan scored at loan_start_date; features are as-of snapshots, "
                              "not ordered series per entity, so lag autocorrelation is undefined",
                       action="none")]).to_csv(os.path.join(FE_DIR, "acf_report.csv"), index=False)

    # --- VIF ---
    def vif_table(features):
        X = train_full[features].astype(float)
        X = X.fillna(X.median())
        X = X.loc[:, X.std() > 0]
        X = (X - X.mean()) / X.std()
        Xc = np.column_stack([np.ones(len(X)), X.values])
        return pd.Series([variance_inflation_factor(Xc, i + 1) for i in range(X.shape[1])], index=X.columns)

    def vif_prune(features):
        feats, removed = list(features), []
        while True:
            v = vif_table(feats)
            high = v[v > FILTER["vif_max"]]
            if high.empty:
                return feats, removed, v
            gen_high = high[[c for c in high.index if origin(c) == "generated"]]
            worst = (gen_high if not gen_high.empty else high).idxmax()
            removed.append((worst, float(high[worst])))
            feats.remove(worst)

    screen_drop = report.index[report[["flag_missing", "flag_low_variance", "flag_leakage",
                                       "flag_exact_duplicate"]].any(axis=1)]
    vif_all = vif_table([c for c in num_like if c not in screen_drop])
    vif_report = vif_all.rename("vif_all_candidates").to_frame()
    vif_report["origin"] = vif_report.index.map(origin)
    vif_report["flag_vif_gt_10"] = vif_report["vif_all_candidates"] > FILTER["vif_max"]
    report["vif"] = vif_report["vif_all_candidates"]
    report["flag_high_vif"] = report["vif"].fillna(0) > FILTER["vif_max"]

    # --- model-specific keep / drop ---
    def near_duplicate_drops(features):
        drops = {}
        fs_pairs = corr_report[corr_report["near_duplicate"] & corr_report["feature_1"].isin(features)
                               & corr_report["feature_2"].isin(features)]
        for _, p in fs_pairs.iterrows():
            a, b = p["feature_1"], p["feature_2"]
            if a in drops or b in drops:
                continue
            if a in PREFER_DROP or b in PREFER_DROP:
                loser = a if a in PREFER_DROP else b
            elif origin(a) != origin(b) and "generated" in (origin(a), origin(b)):
                loser = a if origin(a) == "generated" else b
            else:
                loser = a if report.at[a, "iv"] < report.at[b, "iv"] else b
            drops[loser] = f"near-duplicate of {b if loser == a else a} (|rho| {p['abs_rho']:.3f})"
        return drops

    filtered, decisions, vif_steps = {}, [], []
    for set_name, cands in set_candidates.items():
        common = {c: reason for c in cands for reason in [
            "low variance" if report.at[c, "flag_low_variance"] else
            f"exact duplicate of {report.at[c, 'exact_duplicate_of']}" if report.at[c, "flag_exact_duplicate"] else
            "missing > 50%" if report.at[c, "flag_missing"] else
            "suspected leakage" if report.at[c, "flag_leakage"] else ""] if reason}
        tree_keep = [c for c in cands if c not in common]
        nd = near_duplicate_drops(tree_keep)
        lin_after_nd = [c for c in tree_keep if c not in nd]
        lin_num = [c for c in lin_after_nd if c not in cat and train_full[c].nunique() > 1]
        kept_num, vif_removed, final_vif = vif_prune(lin_num)
        vif_drop = {c: f"VIF {v:,.1f} > {FILTER['vif_max']:.0f} (iterative removal)" for c, v in vif_removed}
        lin_keep = [c for c in lin_after_nd if c not in vif_drop]
        filtered[set_name] = {"tree": tree_keep, "linear": lin_keep}
        for i, (c, v) in enumerate(vif_removed, 1):
            vif_steps.append(dict(feature_set=set_name, order=i, removed=c, vif_at_removal=v, origin=origin(c)))
        for c in cands:
            decisions.append(dict(feature_set=set_name, feature=c, origin=origin(c),
                                  tree="drop" if c in common else "keep",
                                  linear="drop" if (c in common or c in nd or c in vif_drop) else "keep",
                                  reason=common.get(c) or nd.get(c) or vif_drop.get(c) or "",
                                  final_linear_vif=float(final_vif.get(c, np.nan))))
    decisions = pd.DataFrame(decisions)
    pd.DataFrame(vif_steps).to_csv(os.path.join(FE_DIR, "vif_removal_steps.csv"), index=False)
    vif_report.reset_index().rename(columns={"index": "feature"}).to_csv(os.path.join(FE_DIR, "vif_report.csv"), index=False)
    assert not set(filtered["A_raw"]["tree"]) & set(afe + click)

    # --- validation impact (validation labels only) ---
    impact = validation_impact(spark, ctx, full, train_full, set_candidates, filtered)

    # --- final report + filtered_features.json ---
    report["tree_decision"] = np.where(report[["flag_missing", "flag_low_variance", "flag_leakage",
                                               "flag_exact_duplicate"]].any(axis=1), "drop", "keep")
    report["linear_decision"] = decisions.groupby("feature").apply(
        lambda g: "drop" if (g["linear"] == "drop").all() else ("keep" if (g["linear"] == "keep").all() else "depends on set"),
        include_groups=False)
    report["reason"] = decisions[decisions["reason"] != ""].drop_duplicates("feature").set_index("feature")["reason"]
    report = report.reset_index()
    report.to_csv(os.path.join(FE_DIR, "feature_filter_report.csv"), index=False)
    decisions.to_csv(os.path.join(FE_DIR, "feature_filter_decisions_by_set.csv"), index=False)
    with open(os.path.join(FE_DIR, "filtered_features.json"), "w", encoding="utf-8") as f:
        json.dump({"version": FE_VERSION, "filters": FILTER, "baseline_set": "A_raw", "sets": filtered}, f, indent=1)
    with open(os.path.join(FE_DIR, "feature_sets.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump({"version": FE_VERSION, "filters": FILTER, "sets": filtered}, f, sort_keys=False)
    if charts:
        plot_step23(impact)
    return dict(report=report, decisions=decisions, feature_sets=filtered, full=full, impact=impact,
                correlation=corr_report, vif=vif_report)


def reference_model(model_type):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    if model_type == "tree":
        return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, random_state=SEED)
    return LogisticRegression(max_iter=3000, C=1.0)


def validation_impact(spark, ctx, full, train_full, set_candidates, filtered):
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import roc_auc_score, average_precision_score
    cat, binary = ctx["categorical"], ctx["binary"]
    flagged = {c[: -len("_invalid")] for c in ctx["features_a"] if c.endswith("_invalid")}
    val_labels = (spark.read.parquet(gold.LABEL_STORE_PATH).filter("maturity_status = 'mature'")
                  .select("loan_id", "label").toPandas())
    val_full = full[full["partition"] == "validation"].merge(val_labels, on="loan_id")
    assert set(val_full["partition"]) == {"validation"}
    rows = []
    for set_name, cands in set_candidates.items():
        for model_type in MODEL_TYPES:
            for stage, feats in [("before filtering", cands), ("after filtering", filtered[set_name][model_type])]:
                pipe = Pipeline([("pre", build_preprocessor(feats, model_type, cat, binary, flagged)),
                                 ("model", reference_model(model_type))])
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    pipe.fit(train_full[feats], train_full["label"])
                p = pipe.predict_proba(val_full[feats])[:, 1]
                rows.append(dict(feature_set=set_name, model_type=model_type, stage=stage, features=len(feats),
                                 val_roc_auc=roc_auc_score(val_full["label"], p),
                                 val_pr_auc=average_precision_score(val_full["label"], p)))
    impact = pd.DataFrame(rows)
    wide = impact.pivot_table(index=["feature_set", "model_type"], columns="stage",
                              values=["features", "val_roc_auc", "val_pr_auc"]).reset_index()
    wide.columns = [" ".join(c).strip() for c in wide.columns]
    wide["delta_roc_auc"] = wide["val_roc_auc after filtering"] - wide["val_roc_auc before filtering"]
    wide["delta_pr_auc"] = wide["val_pr_auc after filtering"] - wide["val_pr_auc before filtering"]
    wide.to_csv(os.path.join(FE_DIR, "filter_validation_impact.csv"), index=False)
    return wide


def plot_step23(wide):
    plt = _plt()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 3.8))
    lab = [f"{r['feature_set']}\n{r['model_type']}" for _, r in wide.iterrows()]
    x = np.arange(len(wide))
    ax1.bar(x - 0.2, wide["features before filtering"], width=0.38, color=SERIES[0], label="before filtering")
    ax1.bar(x + 0.2, wide["features after filtering"], width=0.38, color=SERIES[1], label="after filtering")
    ax1.set_xticks(x, lab, fontsize=7.5)
    ax1.set_ylabel("features")
    ax1.legend(loc="upper left")
    ax1.set_title("Step 23 — features per set and model type")
    ax2.scatter(x - 0.1, wide["val_pr_auc before filtering"], color=SERIES[0], s=40, label="before filtering", zorder=3)
    ax2.scatter(x + 0.1, wide["val_pr_auc after filtering"], color=SERIES[1], s=40, label="after filtering", zorder=3)
    for xi, (b, a) in enumerate(zip(wide["val_pr_auc before filtering"], wide["val_pr_auc after filtering"])):
        ax2.plot([xi - 0.1, xi + 0.1], [b, a], color=GRID, linewidth=2, zorder=2)
    ax2.set_xticks(x, lab, fontsize=7.5)
    ax2.set_ylabel("validation PR-AUC (reference model)")
    ax2.legend(loc="lower right")
    ax2.set_title("Validation impact of filtering")
    fig.tight_layout()
    os.makedirs(CHART_DIR, exist_ok=True)
    fig.savefig(os.path.join(CHART_DIR, "23_feature_filtering.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)


# =============================================================================================
# Step 18 — Train-fitted preprocessing (one per feature set x model type, on its filtered list)
# =============================================================================================
def build_preprocessor(features, model_type, categorical, binary, flagged):
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer, MissingIndicator
    from sklearn.preprocessing import OneHotEncoder, QuantileTransformer

    cat = [c for c in features if c in categorical]
    bins = [c for c in features if c in binary]
    num = [c for c in features if c not in cat + bins]
    num_flagged = [c for c in num if c in flagged]
    num_other = [c for c in num if c not in flagged]

    def num_pipe():
        steps = [("impute", SimpleImputer(strategy="median"))]
        if model_type == "linear":
            steps.append(("quantile", QuantileTransformer(n_quantiles=1000, output_distribution="normal",
                                                          random_state=SEED)))
        return Pipeline(steps)

    parts = [("num_flagged", num_pipe(), num_flagged),
             ("num_other", num_pipe(), num_other),
             ("missing_ind", MissingIndicator(features="missing-only", error_on_new=False), num_other),
             ("cat", Pipeline([("fill", SimpleImputer(strategy="constant", fill_value="Missing")),
                               ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat),
             ("flags", SimpleImputer(strategy="constant", fill_value=0), bins)]
    return ColumnTransformer([p for p in parts if p[2]], remainder="drop", verbose_feature_names_out=True)


def run_step18(ctx, feature_sets, full, tests=True):
    import joblib
    os.makedirs(PREP_DIR, exist_ok=True)
    for f_old in os.listdir(PREP_DIR):                      # previous preprocessors are replaced
        os.remove(os.path.join(PREP_DIR, f_old))
    cat, binary = ctx["categorical"], ctx["binary"]
    flagged = {c[: -len("_invalid")] for c in ctx["features_a"] if c.endswith("_invalid")}
    part = {p: full[full["partition"] == p] for p in ["train", "validation", "test", "oot"]}
    fitted, rows = {}, []
    for set_name, by_type in feature_sets.items():
        for model_type in MODEL_TYPES:
            feats = by_type[model_type]
            pre = build_preprocessor(feats, model_type, cat, binary, flagged).fit(part["train"][feats])
            fitted[(set_name, model_type)] = pre
            path = os.path.join(PREP_DIR, f"preprocess__{set_name}__{model_type}.joblib")
            joblib.dump(pre, path)
            out_names = list(pre.get_feature_names_out())
            rows.append(dict(feature_set=set_name, model_type=model_type, input_features=len(feats),
                             output_columns=len(out_names), file=os.path.basename(path)))
            with open(path.replace(".joblib", "__columns.json"), "w", encoding="utf-8") as f:
                json.dump({"input": feats, "output": out_names}, f, indent=1)
    manifest = pd.DataFrame(rows)
    manifest.to_csv(os.path.join(PREP_DIR, "preprocessing_manifest.csv"), index=False)
    test_results = preprocessing_tests(fitted, feature_sets, part, cat) if tests else None
    return dict(fitted=fitted, manifest=manifest, tests=test_results)


def preprocessing_tests(fitted, feature_sets, part, categorical):
    results = []
    for (set_name, model_type), pre in fitted.items():
        feats = feature_sets[set_name][model_type]
        tr = part["train"][feats]
        names = [n for n, *_ in pre.transformers_]
        if "num_flagged" in names:
            cols = pre.transformers_[names.index("num_flagged")][2]
            assert np.allclose(pre.named_transformers_["num_flagged"].named_steps["impute"].statistics_,
                               tr[cols].median().values), "not train medians"
        before = copy.deepcopy(pre)
        outs = {p: pre.transform(part[p][feats]) for p in ["train", "validation", "test", "oot"]}
        for (n1, t1, _), (n2, t2, _) in zip(before.transformers_, pre.transformers_):
            objs = [(t1, t2)] + ([(s1, s2) for (_, s1), (_, s2) in zip(t1.steps, t2.steps)] if hasattr(t1, "steps") else [])
            for o1, o2 in objs:
                for attr in ["statistics_", "quantiles_", "categories_", "features_"]:
                    if hasattr(o1, attr):
                        v1, v2 = getattr(o1, attr), getattr(o2, attr)
                        same = (all(np.array_equal(a, b) for a, b in zip(v1, v2)) if isinstance(v1, list)
                                else np.array_equal(v1, v2))
                        assert same, f"{set_name}/{model_type}: {n1}.{attr} changed after transform"
        probe = tr.iloc[[0, 1]].copy()
        for c in [c for c in feats if c in categorical]:
            probe[c] = ["Astronaut_NEW", None]
        for c in [c for c in feats if c not in categorical]:
            probe.iloc[1, probe.columns.get_loc(c)] = np.nan
        p_out = pre.transform(probe)
        widths = {p: o.shape[1] for p, o in outs.items()}
        no_null = all(not np.isnan(np.asarray(o, dtype=float)).any() for o in list(outs.values()) + [p_out])
        deterministic = np.allclose(np.asarray(pre.transform(part["validation"][feats]), dtype=float),
                                    np.asarray(outs["validation"], dtype=float), equal_nan=True)
        assert no_null and len(set(widths.values())) == 1 and deterministic
        results.append(dict(feature_set=set_name, model_type=model_type, columns=widths["train"], fitted_on="train",
                            no_refit=True, unseen_category_ok=True, new_missing_ok=True, no_nulls=no_null,
                            deterministic=deterministic))
    print(f"Preprocessing tests passed for all {len(results)} fitted preprocessors")
    return pd.DataFrame(results)


# =============================================================================================
# Gate 3 + entry point
# =============================================================================================
def gate3(s21, s23, s18):
    split_checks = pd.read_csv(os.path.join(gold.SPLIT_DIR, "split_checks.csv"))
    sets = ", ".join(f"{k} tree {len(v['tree'])}/linear {len(v['linear'])}" for k, v in s23["feature_sets"].items())
    rows = [
        dict(criterion="Time-safe features", status="PASS",
             evidence="all inputs as-of loan_start_date (Gold step 13); generated features are row-wise formulas of "
                      "those inputs; no temporal / time-series features (steps 20, 22 not applied; ACF N/A)"),
        dict(criterion="Train-only transformations and filtering",
             status="PASS" if s18["tests"] is not None else "NOT TESTED",
             evidence=f"OpenFE fitted on train; step-23 screens and VIF on train (validation only for documented "
                      f"impact); {len(s18['manifest'])} preprocessors fitted on train, never refit (tests)"),
        dict(criterion="No holdout leakage", status="PASS" if (split_checks["status"] != "FAIL").all() else "FAIL",
             evidence="test and OOT labels never read in section 03; split manifest v1 frozen"),
        dict(criterion="Reproducible definitions", status="PASS",
             evidence=f"autofe_catalog.csv ({len(s21['catalog'])} formulas, tested); filtered_features.json ({sets})"),
    ]
    g = pd.DataFrame(rows)
    g.to_csv(os.path.join(FE_DIR, "gate3_check.csv"), index=False)
    return g


def run(spark=None, charts=True, tests=True):
    """Section 03: step 21 (OpenFE) -> 23 (filtering, V6.6) -> 18 (preprocessing). Needs Gold and the step-14 split."""
    spark = spark or get_spark()
    ctx = gold.load_eda_data(spark)
    s21 = run_step21(ctx, charts=charts)
    s23 = run_step23(spark, ctx, s21["catalog"], s21["matrix"], charts=charts)
    s18 = run_step18(ctx, s23["feature_sets"], s23["full"], tests=tests)
    return dict(step21=s21, step23=s23, step18=s18, gate3=gate3(s21, s23, s18))


if __name__ == "__main__":
    import argparse
    import matplotlib
    matplotlib.use("Agg")
    parser = argparse.ArgumentParser(description="Section 03: AutoFE (21), filtering (23), preprocessing (18)")
    parser.add_argument("--no-charts", action="store_true")
    parser.add_argument("--no-tests", action="store_true")
    args = parser.parse_args()
    out = run(charts=not args.no_charts, tests=not args.no_tests)
    print(f"Generated features kept: {len(out['step21']['catalog'])}")
    print("Feature sets:", {k: {m: len(v) for m, v in d.items()} for k, d in out["step23"]["feature_sets"].items()})
    print(out["gate3"].to_string(index=False))
