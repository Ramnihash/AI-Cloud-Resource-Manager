"""CP7.3 Device study: anomaly detector on labelled simulated telemetry.

Usage (from the repository root, after training the detector):

    python -m experiments.device_eval

Telemetry from the simulator is deterministic, so "seeded" here means fixed
window start indices (all beyond the training range) and a fixed noise seed.
Two conditions are run: clean telemetry and telemetry with Gaussian noise
(5% of each feature's normal standard deviation) added to every sample.

Writes to ``results/``:
  device_anomaly_scores.csv    one row per window (label, pattern, score, flagged)
  device_anomaly_metrics.csv   precision, recall, F1, FPR, ROC-AUC per condition/pattern
  device_detection_delay.csv   samples from fault onset to first flagged window, plus false alarms before onset
  device_actions.csv           backup success/latency and reroute success per condition
  device_severity.csv          recall and risk when the faults are scaled down (severity 1.0 to 0.05)
  device_*.png                 ROC curves, score distribution, recall per pattern
"""

from __future__ import annotations

import os
import time
from typing import Callable

import numpy as np
import pandas as pd

from ml.device.sequence import TELEMETRY_FEATURES
from ml.device.anomaly_service import (
    FAULT_PATTERNS,
    WINDOW_SIZE,
    ModelUnavailableError,
    build_window,
    load_bundle,
    risk_from_score,
    score_window,
)
from modules.device.abnormal import generate_abnormal_telemetry
from modules.device.fleet import create_default_fleet
from modules.device.predictive_action import handle_predictive_action

SEED = 42
RESULTS_DIR = "results"
EVAL_START = 400  # the detector trains on sample indices 0-299
N_STARTS = 30
START_STEP = 7
NOISE_LEVELS = {"clean": 0.0, "noisy": 0.05}
STREAM_LENGTH = 60
ONSET = 30
DELAY_STARTS = 10
SEVERITIES = [1.0, 0.5, 0.25, 0.1, 0.05]


def _noisy(window: np.ndarray, std: np.ndarray, level: float, rng: np.random.Generator) -> np.ndarray:
    if level <= 0:
        return window
    return window + rng.normal(0.0, 1.0, window.shape) * std * level


def _metrics(y_true: np.ndarray, scores: np.ndarray, flagged: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

    tn = int(((y_true == 0) & ~flagged).sum())
    fp = int(((y_true == 0) & flagged).sum())
    auc = float(roc_auc_score(y_true, scores)) if len(set(y_true)) == 2 else float("nan")
    return {
        "windows": int(len(y_true)),
        "precision": float(precision_score(y_true, flagged, zero_division=0)),
        "recall": float(recall_score(y_true, flagged, zero_division=0)),
        "f1": float(f1_score(y_true, flagged, zero_division=0)),
        "false_positive_rate": fp / (fp + tn) if fp + tn else float("nan"),
        "roc_auc": auc,
    }


def evaluate_windows(bundle, scorer: Callable, fleet) -> pd.DataFrame:
    threshold = bundle["threshold"]
    std = bundle["std"]
    rows = []
    for cond_index, (condition, level) in enumerate(NOISE_LEVELS.items()):
        rng = np.random.default_rng(SEED + cond_index)
        for device in fleet:
            for k in range(N_STARTS):
                start = EVAL_START + k * START_STEP
                for pattern in [None, *FAULT_PATTERNS]:
                    window, _ = build_window(device, pattern, WINDOW_SIZE, start)
                    score = scorer(bundle, _noisy(window, std, level, rng))
                    rows.append({
                        "condition": condition,
                        "device_id": device.device_id,
                        "start_index": start,
                        "pattern": pattern or "normal",
                        "label": int(pattern is not None),
                        "score": score,
                        "flagged": bool(score > threshold),
                    })
    return pd.DataFrame(rows)


def evaluate_severity(bundle, scorer: Callable, fleet) -> pd.DataFrame:
    """Clean telemetry with faults scaled down: how mild a fault is still detected?"""
    threshold = bundle["threshold"]
    rows = []
    for device in fleet:
        for k in range(N_STARTS):
            start = EVAL_START + k * START_STEP
            for severity in SEVERITIES:
                for pattern in FAULT_PATTERNS:
                    window, _ = build_window(device, pattern, WINDOW_SIZE, start, severity=severity)
                    score = scorer(bundle, window)
                    rows.append({
                        "severity": severity,
                        "pattern": pattern,
                        "score": score,
                        "score_over_threshold": score / threshold,
                        "risk": risk_from_score(score, threshold),
                        "flagged": bool(score > threshold),
                    })
    return pd.DataFrame(rows)


def summarise_severity(severity: pd.DataFrame) -> pd.DataFrame:
    """Recall, median risk and share of windows with risk capped at 1.0, per severity and pattern."""
    grouped = severity.groupby(["severity", "pattern"])
    out = grouped.agg(
        windows=("flagged", "size"),
        recall=("flagged", "mean"),
        median_score_over_threshold=("score_over_threshold", "median"),
        median_risk=("risk", "median"),
        risk_capped_share=("risk", lambda r: float((r >= 1.0).mean())),
    ).reset_index()
    overall = severity.groupby("severity").agg(
        windows=("flagged", "size"),
        recall=("flagged", "mean"),
        median_score_over_threshold=("score_over_threshold", "median"),
        median_risk=("risk", "median"),
        risk_capped_share=("risk", lambda r: float((r >= 1.0).mean())),
    ).reset_index()
    overall["pattern"] = "all"
    return pd.concat([out, overall], ignore_index=True)


def evaluate_delay(bundle, scorer: Callable, fleet) -> pd.DataFrame:
    """Slide the detector over a stream where a fault starts at ``ONSET``."""
    threshold = bundle["threshold"]
    std = bundle["std"]
    rows = []
    for cond_index, (condition, level) in enumerate(NOISE_LEVELS.items()):
        rng = np.random.default_rng(SEED + 100 + cond_index)
        for device in fleet:
            for k in range(DELAY_STARTS):
                start = EVAL_START + k * 53
                for pattern in FAULT_PATTERNS:
                    # one long window source: normal before ONSET, fault from ONSET on
                    normal, _ = build_window(device, None, STREAM_LENGTH, start)
                    stream = normal.copy()
                    for i in range(ONSET, STREAM_LENGTH):
                        s = generate_abnormal_telemetry(device, float(start + i), pattern, sample_index=start + i).to_dict()
                        stream[i] = [s[f] for f in TELEMETRY_FEATURES]
                    stream = _noisy(stream, std, level, rng)
                    delay = None
                    false_alarm = False
                    for end in range(WINDOW_SIZE, STREAM_LENGTH + 1):
                        flagged = scorer(bundle, stream[end - WINDOW_SIZE:end]) > threshold
                        if end <= ONSET:
                            false_alarm = false_alarm or flagged  # window holds only normal samples
                        elif flagged and delay is None:
                            delay = end - ONSET  # samples after onset (1 = first fault sample)
                    rows.append({
                        "condition": condition,
                        "device_id": device.device_id,
                        "pattern": pattern,
                        "start_index": start,
                        "detected": delay is not None,
                        "delay_samples": delay,
                        "false_alarm_before_onset": false_alarm,
                    })
    return pd.DataFrame(rows)


def evaluate_actions(scores: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Run the prediction-to-action pipeline for every flagged window."""
    os.environ.pop("BACKUP_S3_BUCKET", None)  # simulated backups only: no cloud calls
    rows = []
    for condition, group in scores.groupby("condition"):
        flagged = group[group["flagged"]]
        latencies, backup_ok, reroute_ok = [], 0, 0
        for _, row in flagged.iterrows():
            risk = risk_from_score(row["score"], threshold)
            t0 = time.perf_counter()
            result = handle_predictive_action(
                row["device_id"], risk, "unhealthy", backup_device_id="spare_0"
            )
            latencies.append((time.perf_counter() - t0) * 1000.0)
            backup_ok += bool(result.backup_result.success)
            reroute_ok += bool(result.recovery_result.rerouted)
        n = len(flagged)
        rows.append({
            "condition": condition,
            "actions": n,
            "backup_success_rate": backup_ok / n if n else float("nan"),
            "reroute_success_rate": reroute_ok / n if n else float("nan"),
            "backup_latency_ms_mean": float(np.mean(latencies)) if n else float("nan"),
            "backup_latency_ms_p95": float(np.percentile(latencies, 95)) if n else float("nan"),
        })
    return pd.DataFrame(rows)


def summarise(scores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for condition, group in scores.groupby("condition"):
        overall = _metrics(group["label"].to_numpy(), group["score"].to_numpy(), group["flagged"].to_numpy())
        rows.append({"condition": condition, "pattern": "all", **overall})
        normal = group[group["label"] == 0]
        for pattern in FAULT_PATTERNS:
            sub = pd.concat([normal, group[group["pattern"] == pattern]])
            rows.append({
                "condition": condition,
                "pattern": pattern,
                **_metrics(sub["label"].to_numpy(), sub["score"].to_numpy(), sub["flagged"].to_numpy()),
            })
    return pd.DataFrame(rows)


def make_plots(scores: pd.DataFrame, metrics: pd.DataFrame, threshold: float, out: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve

    fig, ax = plt.subplots(figsize=(5, 4.5))
    for condition, group in scores.groupby("condition"):
        fpr, tpr, _ = roc_curve(group["label"], group["score"])
        auc = metrics[(metrics.condition == condition) & (metrics.pattern == "all")]["roc_auc"].iloc[0]
        ax.plot(fpr, tpr, label=f"{condition} (AUC {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Device anomaly detector: ROC")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out, "device_roc.png"), dpi=150)
    plt.close(fig)

    clean = scores[scores.condition == "clean"]
    fig, ax = plt.subplots(figsize=(6, 4))
    floor = max(float(clean["score"].min()), 1e-9)
    bins = np.logspace(np.log10(floor), np.log10(float(clean["score"].max())), 40)
    ax.hist(clean[clean.label == 0]["score"], bins=bins, alpha=0.7, label="normal")
    ax.hist(clean[clean.label == 1]["score"], bins=bins, alpha=0.7, label="fault")
    ax.axvline(threshold, color="k", linestyle="--", label="threshold")
    ax.set_xscale("log")
    ax.set_xlabel("Reconstruction error (anomaly score)")
    ax.set_ylabel("Windows")
    ax.set_title("Score distribution (clean telemetry)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out, "device_scores.png"), dpi=150)
    plt.close(fig)

    per = metrics[metrics.pattern != "all"]
    fig, ax = plt.subplots(figsize=(7, 4))
    patterns = list(per.pattern.unique())
    width = 0.38
    for i, condition in enumerate(per.condition.unique()):
        vals = [per[(per.condition == condition) & (per.pattern == p)]["recall"].iloc[0] for p in patterns]
        ax.bar(np.arange(len(patterns)) + i * width, vals, width, label=condition)
    ax.set_xticks(np.arange(len(patterns)) + width / 2)
    ax.set_xticklabels(patterns, rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Recall")
    ax.set_title("Detection recall per fault pattern")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out, "device_recall.png"), dpi=150)
    plt.close(fig)


def run(bundle=None, scorer: Callable | None = None, out: str = RESULTS_DIR) -> dict[str, pd.DataFrame]:
    bundle = bundle if bundle is not None else load_bundle()
    scorer = scorer or score_window
    os.makedirs(out, exist_ok=True)
    fleet = create_default_fleet()

    scores = evaluate_windows(bundle, scorer, fleet)
    metrics = summarise(scores)
    delay = evaluate_delay(bundle, scorer, fleet)
    actions = evaluate_actions(scores, bundle["threshold"])
    severity = summarise_severity(evaluate_severity(bundle, scorer, fleet))

    scores.to_csv(os.path.join(out, "device_anomaly_scores.csv"), index=False)
    metrics.to_csv(os.path.join(out, "device_anomaly_metrics.csv"), index=False)
    delay.to_csv(os.path.join(out, "device_detection_delay.csv"), index=False)
    actions.to_csv(os.path.join(out, "device_actions.csv"), index=False)
    severity.to_csv(os.path.join(out, "device_severity.csv"), index=False)
    make_plots(scores, metrics, bundle["threshold"], out)
    return {"scores": scores, "metrics": metrics, "delay": delay, "actions": actions, "severity": severity}


def main() -> None:
    try:
        results = run()
    except ModelUnavailableError as exc:
        raise SystemExit(str(exc))
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 20)
    print("\nMetrics (threshold fixed from held-out normal data):")
    print(results["metrics"].round(4).to_string(index=False))
    delay = results["delay"]
    print("\nDetection delay (samples after fault onset; 1 = first fault sample):")
    print(
        delay.groupby(["condition", "pattern"])
        .agg(
            detected=("detected", "mean"),
            mean_delay=("delay_samples", "mean"),
            std_delay=("delay_samples", "std"),
            false_alarm_before_onset=("false_alarm_before_onset", "mean"),
        )
        .round(3)
        .to_string()
    )
    print("\nActions on flagged windows (simulated cloud backup):")
    print(results["actions"].round(4).to_string(index=False))
    print("\nMilder faults (clean telemetry; severity 1.0 is the full fault used elsewhere):")
    print(results["severity"].query("pattern == 'all'").drop(columns="pattern").round(3).to_string(index=False))
    print("\nBy pattern (recall):")
    print(results["severity"].query("pattern != 'all'").pivot(index="pattern", columns="severity", values="recall").round(3).to_string())
    print(f"\nCSVs and plots written to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
