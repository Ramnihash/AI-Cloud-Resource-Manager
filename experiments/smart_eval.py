"""CP7.3 SMART study: disk failure-risk classifier on real Backblaze telemetry.

Usage (from the repository root, after building the dataset with
``python -m scripts.build_smart_dataset``):

    python -m experiments.smart_eval

Task: from one day's SMART raw values, predict whether the drive fails within
the dataset's horizon (30 days). Rows of one drive are strongly correlated, so
the 5 folds are split by drive serial number (StratifiedGroupKFold): no drive
appears in both training and test data.

Compared on the same folds:
  random_forest        permitted model, used by the backend
  logistic_regression  permitted model (log-scaled, standardised features)
  rule_baseline        flag a drive when any of SMART 5, 187, 188, 197, 198 is above 0
                       (Backblaze's published rule of thumb), no training

The sample over-represents failing days (about 1:1 instead of about 1 in 1000),
so precision on the sample is optimistic. ``precision_at_prevalence`` converts
recall and false-positive rate to the precision expected at the real share of
failing drive-days in the quarter (from the dataset's .meta.json).

Writes to ``results/``:
  smart_folds.csv               one row per model and fold
  smart_metrics.csv             mean and standard deviation over folds
  smart_recall_by_lead.csv      recall by days left before failure
  smart_feature_importance.csv  Random Forest importances (model fitted on all rows)
  smart_roc_pr.png              ROC and precision-recall curves (pooled out-of-fold scores)
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedGroupKFold

from data.device.smart.build_failure_windows import FEATURE_COLUMNS, meta_path_for
from ml.device.smart.service import DATASET_PATH, RISK_THRESHOLD, SEED, build_pipeline, load_dataset

RESULTS_DIR = "results"
N_FOLDS = 5
MODELS = ["random_forest", "logistic_regression", "rule_baseline"]
RULE_COLUMNS = ["smart_5_raw", "smart_187_raw", "smart_188_raw", "smart_197_raw", "smart_198_raw"]
FPR_TARGETS = [0.01, 0.001]
LEAD_BUCKETS = [(0, 6, "0-6 days"), (7, 13, "7-13 days"), (14, 29, "14-29 days")]


def rule_scores(features: pd.DataFrame) -> np.ndarray:
    """Number of the five warning attributes that are above zero (0 to 5)."""
    present = [c for c in RULE_COLUMNS if c in features]
    return (features[present].fillna(0) > 0).sum(axis=1).to_numpy(dtype=float)


def tpr_at_fpr(y: np.ndarray, scores: np.ndarray, target: float) -> tuple[float, float]:
    """Highest recall whose false-positive rate is at most ``target``, and that FPR."""
    fpr, tpr, _ = roc_curve(y, scores)
    ok = fpr <= target
    if not ok.any():
        return 0.0, 0.0
    i = np.argmax(np.where(ok, tpr, -1))
    return float(tpr[i]), float(fpr[i])


def precision_at_prevalence(tpr: float, fpr: float, prevalence: float) -> float:
    denom = prevalence * tpr + (1 - prevalence) * fpr
    return float(prevalence * tpr / denom) if denom > 0 else float("nan")


def drive_level(test: pd.DataFrame, flagged: np.ndarray) -> tuple[float, float]:
    """Share of failed drives flagged at least once in the horizon, and the median lead time."""
    failing = test["failure"].to_numpy() == 1
    frame = pd.DataFrame({
        "serial": test["serial_number"].to_numpy()[failing],
        "days": test["days_to_failure"].to_numpy()[failing],
        "flag": flagged[failing],
    })
    if frame.empty:
        return float("nan"), float("nan")
    caught = frame.groupby("serial")["flag"].any()
    leads = frame[frame["flag"]].groupby("serial")["days"].max()
    return float(caught.mean()), float(leads.median()) if len(leads) else float("nan")


def run(dataset_path: str | None = None, out: str = RESULTS_DIR, n_folds: int = N_FOLDS) -> dict[str, pd.DataFrame]:
    dataset_path = dataset_path or DATASET_PATH
    os.makedirs(out, exist_ok=True)
    data = load_dataset(dataset_path).reset_index(drop=True)
    meta_path = meta_path_for(dataset_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    prevalence = float(meta.get("prevalence") or 0.0)

    features = data[list(FEATURE_COLUMNS)].apply(pd.to_numeric, errors="coerce")
    y = data["failure"].astype(int).to_numpy()
    groups = data["serial_number"].astype(str).to_numpy()
    splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=SEED)

    fold_rows, lead_rows = [], []
    oof = {m: np.full(len(data), np.nan) for m in MODELS}
    for fold, (train_idx, test_idx) in enumerate(splitter.split(features, y, groups)):
        x_train, x_test = features.iloc[train_idx], features.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        test = data.iloc[test_idx]
        for model_name in MODELS:
            if model_name == "rule_baseline":
                scores = rule_scores(x_test)
                flagged = scores > 0
            else:
                pipeline = build_pipeline(model_name)
                pipeline.fit(x_train, y_train)
                scores = pipeline.predict_proba(x_test)[:, 1]
                flagged = scores >= RISK_THRESHOLD
            oof[model_name][test_idx] = scores
            negatives = y_test == 0
            fpr = float(flagged[negatives].mean()) if negatives.any() else float("nan")
            recall = float(recall_score(y_test, flagged, zero_division=0))
            drive_recall, median_lead = drive_level(test, flagged)
            row = {
                "model": model_name, "fold": fold,
                "test_rows": int(len(test_idx)), "test_positive_rows": int(y_test.sum()),
                "test_failed_drives": int(test.loc[test["failure"] == 1, "serial_number"].nunique()),
                "precision": float(precision_score(y_test, flagged, zero_division=0)),
                "recall": recall,
                "f1": float(f1_score(y_test, flagged, zero_division=0)),
                "fpr": fpr,
                "roc_auc": float(roc_auc_score(y_test, scores)),
                "pr_auc": float(average_precision_score(y_test, scores)),
                "precision_at_prevalence": precision_at_prevalence(recall, fpr, prevalence) if prevalence else float("nan"),
                "drive_recall": drive_recall,
                "median_lead_days": median_lead,
            }
            for target in FPR_TARGETS:
                tpr, achieved = tpr_at_fpr(y_test, scores, target)
                tag = f"{target * 100:g}pct".replace(".", "_")
                row[f"recall_at_fpr_{tag}"] = tpr
                row[f"precision_at_prevalence_fpr_{tag}"] = (
                    precision_at_prevalence(tpr, achieved, prevalence) if prevalence and tpr > 0 else float("nan")
                )
            fold_rows.append(row)
            days = test["days_to_failure"].to_numpy()
            for lo, hi, label in LEAD_BUCKETS:
                mask = (y_test == 1) & (days >= lo) & (days <= hi)
                if mask.any():
                    lead_rows.append({"model": model_name, "fold": fold, "days_left": label,
                                      "rows": int(mask.sum()), "recall": float(flagged[mask].mean())})

    folds = pd.DataFrame(fold_rows)
    numeric = [c for c in folds.columns if c not in ("model", "fold")]
    summary = folds.groupby("model", sort=False)[numeric].agg(["mean", "std"])
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    summary = summary.reset_index()
    leads = pd.DataFrame(lead_rows).groupby(["model", "days_left"], sort=False).agg(
        rows=("rows", "sum"), recall_mean=("recall", "mean"), recall_std=("recall", "std")).reset_index()

    forest = build_pipeline("random_forest").fit(features, y)
    names = list(FEATURE_COLUMNS) + [f"{FEATURE_COLUMNS[i]}_missing" for i in forest.named_steps["impute"].indicator_.features_]
    importance = pd.DataFrame({"feature": names, "importance": forest.named_steps["model"].feature_importances_})
    importance = importance.sort_values("importance", ascending=False).reset_index(drop=True)

    folds.to_csv(os.path.join(out, "smart_folds.csv"), index=False)
    summary.to_csv(os.path.join(out, "smart_metrics.csv"), index=False)
    leads.to_csv(os.path.join(out, "smart_recall_by_lead.csv"), index=False)
    importance.to_csv(os.path.join(out, "smart_feature_importance.csv"), index=False)
    make_plot(y, oof, out)
    return {"folds": folds, "summary": summary, "leads": leads, "importance": importance, "meta": meta}


def make_plot(y: np.ndarray, oof: dict[str, np.ndarray], out: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(10, 4.2))
    for name, scores in oof.items():
        fpr, tpr, _ = roc_curve(y, scores)
        ax_roc.plot(fpr, tpr, label=f"{name} (AUC {roc_auc_score(y, scores):.3f})")
        precision, recall, _ = precision_recall_curve(y, scores)
        ax_pr.plot(recall, precision, label=f"{name} (AP {average_precision_score(y, scores):.3f})")
    ax_roc.plot([0, 1], [0, 1], color="grey", lw=0.8, ls="--")
    ax_roc.set(xlabel="False-positive rate", ylabel="Recall", title="ROC (out-of-fold, split by drive)")
    ax_pr.set(xlabel="Recall", ylabel="Precision (on the balanced sample)", title="Precision-recall")
    for ax in (ax_roc, ax_pr):
        ax.legend(fontsize=8, loc="lower right" if ax is ax_roc else "lower left")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "smart_roc_pr.png"), dpi=150)
    plt.close(fig)


def main() -> None:
    r = run()
    meta = r["meta"]
    if meta:
        print(f"Data: {meta.get('first_day')} to {meta.get('last_day')}, {meta.get('rows')} rows, "
              f"{meta.get('failed_drives_in_dataset')} failed drives, real prevalence {meta.get('prevalence', 0):.5f}")
    cols = ["model", "precision_mean", "recall_mean", "fpr_mean", "roc_auc_mean", "pr_auc_mean",
            "recall_at_fpr_1pct_mean", "precision_at_prevalence_fpr_1pct_mean", "drive_recall_mean", "median_lead_days_mean"]
    print(r["summary"][cols].round(4).to_string(index=False))
    print("\nRecall by days left:")
    print(r["leads"].round(3).to_string(index=False))
    print("\nTop features:")
    print(r["importance"].head(8).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
