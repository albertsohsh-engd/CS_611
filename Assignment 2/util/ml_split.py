"""Train / validation / test + out-of-time split (step 14).

Converted from notebooks/ml_split.ipynb (moved there from script/mlpipeline.ipynb so the split can run as its own
Airflow task). The code is the notebook's code unchanged, wrapped in step14(); results are kept as module attributes
(``manifest``, ``summary``, ``split_checks``). Writes datamart/gold/split/v1/.

Usage:
    from util import dataprocessing_bronze as bronze, ml_split
    ml_split.run(bronze.get_spark())

Command line:
    python util/ml_split.py
"""
import os, sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yaml
from pyspark.sql import functions as F
from sklearn.model_selection import train_test_split

# Project folder: /app inside the Docker container, otherwise the parent of this notebook's folder (notebooks/)
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)
from util import dataprocessing_bronze as bronze      # Spark session

GOLD_DIR = os.path.join(PROJECT_DIR, "datamart", "gold")
with open(os.path.join(PROJECT_DIR, "config", "data_contract.yaml"), encoding="utf-8") as f:
    CONTRACT = yaml.safe_load(f)
LABEL_DEF = CONTRACT["target"]["label_def"]
SCORING_COL = CONTRACT["scoring_timestamp"]["column"]

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


def step14():
    """Step 14 — Train / validation / test + out-of-time split"""
    global CHART_DIR, GRID, INK, INK_2, ORDER, SERIES, SIZE_TOLERANCE_PP, SPLIT_CONFIG, SPLIT_DIR
    global SPLIT_VERSION, STATUS_STYLE, STRAT_TOLERANCE_PP, SURFACE, again, ax, bottom, c, check, checks, color
    global ct, cust_parts, dev_max, dev_rate, dup, f, fig, frac, gap, labels, load_population, make_split
    global manifest, manifest_out, mticker, n_dev, oot_min, oot_start, overlap, overlap_months, p, plt
    global population, r, same, share, split_checks, summary, train_last_outcome, view
    SPLIT_VERSION = "v1"
    SPLIT_CONFIG = {
        "version": SPLIT_VERSION,
        "oot_start": "2024-07-01",                       # loans starting on/after this date -> OOT holdout
        "fractions": {"train": 0.70, "validation": 0.15, "test": 0.15},   # of the development period
        "seed": 42,
        "stratify_by": "label",
        "group_by": "Customer_ID",
        "label_def": LABEL_DEF,
        "eligibility": "maturity_status == 'mature'",
    }
    STRAT_TOLERANCE_PP = 1.0      # label rate of train/validation/test must be within 1 pp of the development rate
    SIZE_TOLERANCE_PP = 1.0       # partition share within 1 pp of the configured fraction

    SPLIT_DIR = os.path.join(GOLD_DIR, "split", SPLIT_VERSION)
    CHART_DIR = os.path.join(GOLD_DIR, "_reports", "charts")
    os.makedirs(SPLIT_DIR, exist_ok=True)
    os.makedirs(CHART_DIR, exist_ok=True)

    def load_population():
        # Eligible loans: mature label in the label store, present in the feature store
        fs = spark.read.parquet(os.path.join(GOLD_DIR, "feature_store", "v1")).select(
            "loan_id", "Customer_ID", SCORING_COL, F.col("_asof_clickstream").isNotNull().alias("has_clickstream_data"))
        ls = spark.read.parquet(os.path.join(GOLD_DIR, "label_store", LABEL_DEF)).select(
            "loan_id", "label", "maturity_status", "label_snapshot_date")
        pop = fs.join(ls, "loan_id", "inner").toPandas()
        pop[SCORING_COL] = pd.to_datetime(pop[SCORING_COL])
        pop["label_snapshot_date"] = pd.to_datetime(pop["label_snapshot_date"].astype(str))
        return pop[pop["maturity_status"] == "mature"].drop(columns="maturity_status").reset_index(drop=True)


    def make_split(pop, cfg):
        oot = pop[SCORING_COL] >= pd.Timestamp(cfg["oot_start"])
        dev = pop[~oot]
        fr = cfg["fractions"]
        train, rest = train_test_split(dev, train_size=fr["train"], stratify=dev[cfg["stratify_by"]],
                                       random_state=cfg["seed"])
        val, test = train_test_split(rest, train_size=fr["validation"] / (fr["validation"] + fr["test"]),
                                     stratify=rest[cfg["stratify_by"]], random_state=cfg["seed"])
        part = pd.Series("oot", index=pop.index)
        part[train.index], part[val.index], part[test.index] = "train", "validation", "test"
        out = pop.assign(partition=part, split_version=cfg["version"], split_seed=cfg["seed"])
        return out[["loan_id", "Customer_ID", SCORING_COL, "label_snapshot_date", "label", "partition",
                    "has_clickstream_data", "split_version", "split_seed"]]


    population = load_population()
    manifest = make_split(population, SPLIT_CONFIG)
    manifest["created_ts"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"Eligible loans (mature labels): {len(population):,}")
    manifest["partition"].value_counts().reindex(["train", "validation", "test", "oot"]).to_frame("loans")

    ORDER = ["train", "validation", "test", "oot"]
    dev_rate = manifest.loc[manifest["partition"] != "oot", "label"].mean()
    n_dev = (manifest["partition"] != "oot").sum()

    summary = (manifest.groupby("partition")
               .agg(loans=("loan_id", "size"), positives=("label", "sum"), label_rate=("label", "mean"),
                    first_loan_start=(SCORING_COL, "min"), last_loan_start=(SCORING_COL, "max"),
                    cohorts=(SCORING_COL, "nunique"), with_clickstream=("has_clickstream_data", "mean"))
               .reindex(ORDER).reset_index())
    summary["share_of_all_pct"] = summary["loans"] / len(manifest) * 100
    summary["share_of_dev_pct"] = np.where(summary["partition"] != "oot", summary["loans"] / n_dev * 100, np.nan)
    summary["label_rate_pct"] = summary["label_rate"] * 100
    summary["with_clickstream_pct"] = summary["with_clickstream"] * 100
    summary = summary.drop(columns=["label_rate", "with_clickstream"])
    for c in ["first_loan_start", "last_loan_start"]:
        summary[c] = summary[c].dt.strftime("%Y-%m")
    summary.to_csv(os.path.join(SPLIT_DIR, "split_summary.csv"), index=False)
    print(f"Development-period label rate: {dev_rate * 100:.2f}%")
    summary.style.hide(axis="index").format({"loans": "{:,}", "positives": "{:,}", "share_of_all_pct": "{:.1f}%",
                                             "share_of_dev_pct": "{:.1f}%", "label_rate_pct": "{:.2f}%",
                                             "with_clickstream_pct": "{:.0f}%"}, na_rep="—")

    checks = []
    def check(name, status, observed, rule):
        checks.append(dict(check=name, status=status, observed=observed, rule=rule))

    # 1
    dup = manifest["loan_id"].duplicated().sum()
    check("each eligible loan in exactly one partition", "PASS" if dup == 0 and len(manifest) == len(population) else "FAIL",
          f"{len(manifest):,} assigned, {dup} duplicates", "all eligible, no duplicates")

    # 2
    cust_parts = manifest.groupby("Customer_ID")["partition"].nunique()
    overlap = int((cust_parts > 1).sum())
    check("zero customer overlap between partitions", "PASS" if overlap == 0 else "FAIL",
          f"{overlap} customers in more than one partition", "0")

    # 3
    oot_start = pd.Timestamp(SPLIT_CONFIG["oot_start"])
    dev_max = manifest.loc[manifest["partition"] != "oot", SCORING_COL].max()
    oot_min = manifest.loc[manifest["partition"] == "oot", SCORING_COL].min()
    check("no temporal overlap: development before OOT", "PASS" if dev_max < oot_start <= oot_min else "FAIL",
          f"last development loan {dev_max:%Y-%m}, first OOT loan {oot_min:%Y-%m}", f"dev < {oot_start:%Y-%m} <= OOT")

    # 4
    for p in ["train", "validation", "test"]:
        r = manifest.loc[manifest["partition"] == p, "label"].mean()
        gap = abs(r - dev_rate) * 100
        check(f"stratification: {p} label rate", "PASS" if gap <= STRAT_TOLERANCE_PP else "FAIL",
              f"{r * 100:.2f}% vs development {dev_rate * 100:.2f}% (gap {gap:.2f} pp)", f"<= {STRAT_TOLERANCE_PP} pp")

    # 5
    for p, frac in SPLIT_CONFIG["fractions"].items():
        share = (manifest["partition"] == p).sum() / n_dev
        check(f"size: {p}", "PASS" if abs(share - frac) * 100 <= SIZE_TOLERANCE_PP else "FAIL",
              f"{share * 100:.1f}% of development", f"{frac * 100:.0f}% ± {SIZE_TOLERANCE_PP} pp")

    # 6
    again = make_split(load_population(), SPLIT_CONFIG)
    same = again.set_index("loan_id")["partition"].equals(manifest.set_index("loan_id")["partition"].loc[again["loan_id"]])
    check("reproducible with the same seed", "PASS" if same else "FAIL", "identical assignment" if same else "differs", "identical")

    # 7 (reported, not a failure): training outcomes observed after the first OOT scoring date
    train_last_outcome = manifest.loc[manifest["partition"] == "train", "label_snapshot_date"].max()
    overlap_months = (train_last_outcome.year - oot_start.year) * 12 + train_last_outcome.month - oot_start.month + 1
    check("outcome window vs OOT start (reported)", "WARN" if train_last_outcome >= oot_start else "PASS",
          f"training outcomes observed up to {train_last_outcome:%Y-%m}; OOT scoring starts {oot_start:%Y-%m} "
          f"({max(overlap_months, 0)} months overlap)",
          "features are point-in-time (no feature leakage); a strict design would leave a 6-month gap")

    split_checks = pd.DataFrame(checks)
    split_checks.to_csv(os.path.join(SPLIT_DIR, "split_checks.csv"), index=False)
    assert (split_checks["status"] != "FAIL").all(), split_checks[split_checks["status"] == "FAIL"]

    STATUS_STYLE = {"PASS": ("✔ PASS", "#e3f4e3"), "WARN": ("▲ WARN", "#fdf0d2"), "FAIL": ("✖ FAIL", "#f8dede")}
    view = split_checks.assign(status=split_checks["status"].map(lambda s: STATUS_STYLE[s][0]))
    view.style.apply(lambda col: [f"background-color: {STATUS_STYLE[v.split()[-1]][1]}" for v in col],
                     subset=["status"]).hide(axis="index")

    # Save the manifest and the configuration
    manifest_out = manifest.copy()
    for c in [SCORING_COL, "label_snapshot_date"]:
        manifest_out[c] = manifest_out[c].dt.strftime("%Y-%m-%d")
    manifest_out.to_csv(os.path.join(SPLIT_DIR, "split_manifest.csv"), index=False)
    with open(os.path.join(SPLIT_DIR, "split_config.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump({**SPLIT_CONFIG, "eligible_loans": int(len(manifest)),
                        "development_label_rate": round(float(dev_rate), 6),
                        "created_ts": manifest["created_ts"].iloc[0]}, f, sort_keys=False)
    print("Saved:", os.listdir(SPLIT_DIR))

    # Chart: loans per loan-start month by partition (train / validation / test inside development, OOT after)
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
    SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "axes.titlecolor": INK, "axes.titlesize": 11,
        "axes.titleweight": "bold", "axes.titlelocation": "left", "xtick.color": INK_2, "ytick.color": INK_2,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID,
        "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
        "legend.fontsize": 8, "font.size": 9})

    ct = manifest.groupby([SCORING_COL, "partition"]).size().unstack(fill_value=0).reindex(columns=ORDER, fill_value=0)
    fig, ax = plt.subplots(figsize=(12, 4))
    bottom = np.zeros(len(ct))
    labels = {"train": "Train (70%)", "validation": "Validation (15%)", "test": "Test (15%)", "oot": "Out-of-time holdout"}
    for color, p in zip(SERIES, ORDER):
        ax.bar(ct.index, ct[p], bottom=bottom, width=22, color=color, label=labels[p], edgecolor=SURFACE, linewidth=0.8)
        bottom += ct[p].values
    ax.axvline(oot_start - pd.Timedelta(days=15), color=INK_2, linestyle="--", linewidth=1)
    ax.text(oot_start - pd.Timedelta(days=10), ct.sum(axis=1).max() * 1.02, " OOT starts 2024-07", fontsize=8, color=INK_2)
    ax.set_ylabel("loans")
    ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
    ax.legend(loc="upper left", ncol=4)
    ax.set_ylim(0, ct.sum(axis=1).max() * 1.15)
    ax.set_title("Step 17 split — loans per loan-start month by partition")
    fig.tight_layout()
    fig.savefig(os.path.join(CHART_DIR, "17_split_by_cohort.png"), dpi=130, bbox_inches="tight")
    _show()


STEPS = ['step14']


def run(spark_session=None, steps=None, show_charts=False):
    """Run step 14 (the split). Needs the Gold feature store and label store."""
    global spark, SHOW_CHARTS
    SHOW_CHARTS = show_charts
    if not show_charts:
        matplotlib.use("Agg")
    spark = spark_session or bronze.get_spark("ml_split")
    for name in (steps or STEPS):
        t0 = datetime.now()
        print(f"--- {name} ---")
        globals()[name]()
        print(f"--- {name} done in {(datetime.now() - t0).total_seconds():.0f}s")
    return sys.modules[__name__]


if __name__ == "__main__":
    run()
