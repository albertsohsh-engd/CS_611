# Loan default model: governance, refresh SOP and deployment

Model: `loan_default_30dpd_6mob` (probability that a new loan is 30+ days past due at month-on-book 6, scored at loan start).
Production version: `@champion` = XGBoost model A, registry version 1, 30 features, threshold 0.2867.
Evidence: `datamart/ml/deployment/v1/` (policy, trigger decisions, rollback drill, governance log) and `datamart/ml/model_development/v1/final/` (model cards, sign-off, approval).

## 1. Governance

| Topic | Decision |
|---|---|
| Risk tier | High: the score informs credit decisions about customers. Full documentation, independent validation and sign-off at every gate. |
| Model owner | Builds the model, runs monitoring, proposes changes (albertsoh). |
| Independent validator | Re-runs tests and challenges results before any promotion. |
| Approver | Risk / credit committee. Signs every promotion and every policy change. |
| Platform | Runs the scheduled pipeline and executes alias moves. |
| Evidence pack per version | Data contract version, split manifest, model card, metric contract, sign-off and approval, final evaluation (test + out-of-time), robustness analysis, registry version with checksum and lineage tags. |
| Change log | Every alias move and policy decision is appended to `governance_log.csv` (who, what, why, when). |
| Retention | Every registry version and its predictions are kept for at least 7 years; the previous champion stays deployable for rollback. |
| Fairness | Sensitive attributes (age) are not model inputs; segment calibration is re-checked at every promotion. |

## 2. When to refresh

| Type | Trigger | Action |
|---|---|---|
| Scheduled | Every 6 months (labels mature 6 months after loan start) | Refit: same features and settings, newest mature labels, rolling 18-month window |
| Triggered refit | Input PSI > 0.25 on any model input, or score PSI > 0.25, for 2 consecutive months | Refit early |
| | PR-AUC more than 0.05 below validation for 2 consecutive labelled months | Refit early |
| | Calibration gap (mean predicted − actual) above 5 points for 2 consecutive months | Refit early |
| Triggered rebuild | Source or schema change, broken feed, new product or credit policy, regulatory request | Redo feature engineering and model development (checklist steps 21–30) |
| Watch only | One month outside a rule, or PSI between 0.10 and 0.25 | Investigate and record; no model change |

Two consecutive months are required because single months hold only about 500 loans, and their PR-AUC moves by several points from sampling alone.

## 3. How to refresh and promote

1. Run the same pipeline (Docker image, `datapipeline.ipynb` then `mlpipeline.ipynb`) on the new data window.
2. The candidate must pass all of the following:
   - pipeline and stress tests;
   - PR-AUC not lower than the current champion's by more than 0.01 on the newest held-out months;
   - calibration gap within ±5 points;
   - no new weakness in the segment checks.
3. Shadow scoring: the candidate scores live batches for at least one month without driving decisions; flag rates are compared with the champion's.
4. Model card and approver sign-off.
5. Platform moves the `@champion` alias to the new version and logs the change. No code change or redeployment is needed.

## 4. Rollback

If a newly promoted version fails a post-deployment check (inference checks, flag rate outside the expected band, errors), the platform moves `@champion` back to the previous version immediately, logs it, and the owner investigates.

This was rehearsed on the real registry: `@champion` was moved to another version, scoring switched to it straight away, and moving the alias back restored the original predictions exactly (`rollback_drill.csv`).

## 5. Current decision (from the monitoring replay, 2024-07 to 2025-01)

| Model | Finding | Decision |
|---|---|---|
| Champion | Inputs and scores stable (input PSI ≤ 0.06, score PSI ≤ 0.08), flag rate within band, calibration within ±4.5 points. One alert: PR-AUC 0.542 for loans starting 2025-01, known on 2025-07-01. | No early refit (single month). Investigate the 2025-01 cohort. Scheduled refit once the newest labels are complete. |
| Challenger | Clickstream inputs 100% missing in every production month. | Rebuild trigger: not deployable while the feed is missing; stays challenger and is re-assessed if the feed returns. |

## 6. Deployment options

| Option | Fits when | Assessment |
|---|---|---|
| Batch scoring (current) | Decision can wait minutes to a day; features come from snapshots | Recommended. Matches how features are built, is simple and reproducible, and is fully audited. Move from monthly to daily batches so each loan is scored on the day it starts. |
| Real-time API (`mlflow models serve` or FastAPI in the same image) | Instant decision at application | Later option. Needs the features available online at application time, plus latency and uptime guarantees. |
| Streaming | Continuous events, a decision per event | Not needed: one decision per loan, inputs change monthly. |
| Embedded / in-database | Scoring must live inside another system | Not needed: adds conversion and validation work for every version. |

The model is always loaded from the registry by alias. Promotion and rollback therefore never require a new build of the pipeline image.

## 7. Orchestration (Airflow)

Three DAGs (`dags/`) run the pipeline in Docker (Airflow 2.10; the pipeline runs in its own Python environment inside the Airflow image):

| DAG | Schedule | Tasks |
|---|---|---|
| `data_pipeline` | Monthly, back-filled 2023-01 to 2025-11 | Bronze → Silver → Gold feature store + Gold label store for that month → `data_ready` |
| `model_training` | 1 January and 1 July (first run 2025-07), or triggered | Wait for the month's data (scheduled runs only) → full-history data validation → split → Gold EDA → feature engineering → model development → register in MLflow → monitoring reference |
| `inference` | Monthly, back-filled 2024-07 to 2025-07 | Wait for the month's data and for `@champion` → score new loans → monitor → governance check → trigger `model_training` when the refresh rules of section 2 fire |

How this maps onto the SOP:
- The scheduled refit of section 2 is the `model_training` schedule.
- A triggered refit is the `inference` DAG's governance check starting `model_training`.
- A refit or rebuild can also be started by hand from the Airflow UI.

Running the pipeline month by month gives exactly the same Bronze, Silver and Gold data as the full build, and the same predictions and monitoring results (verified in `notebooks/pipeline_monthly.ipynb`).
