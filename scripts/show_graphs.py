"""Rerun the studies and show the four headline graphs on screen.

Usage (from the repository root):

    python -m scripts.show_graphs            # rerun the studies, then show the graphs
    python -m scripts.show_graphs --no-rerun # only redraw from the CSVs already in results/

Graphs:
    1. Average waiting time per workload type, AI-selected scheduler vs FCFS/SJF/Priority/RR
    2. Demand page faults per policy, 4 and 8 frames
    3. Device anomaly detectors: recall as faults get milder
    4. Model selection for Module 1: burst-time MAE and workload-class macro-F1

Each window stays open until it is closed; copies are saved to results/live/.
"""

import argparse
import os
import subprocess
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS = "results"
LIVE = os.path.join(RESULTS, "live")

STUDIES = [
    ("CPU scheduling study (7.1)", ["experiments.cpu_eval"]),
    ("Page-replacement study (7.2)", ["experiments.memory_eval"]),
    ("Model selection: burst, workload class, device detectors (7.6)",
     ["experiments.model_selection_eval", "--only", "cpu_burst,cpu_workload,device"]),
]


def rerun() -> None:
    for title, args in STUDIES:
        print(f"\n=== Running {title} ...", flush=True)
        start = time.perf_counter()
        subprocess.run([sys.executable, "-m", *args], check=True)
        print(f"=== Done in {time.perf_counter() - start:.0f} s", flush=True)


def read(name: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(RESULTS, name))


def save(fig, name: str) -> None:
    os.makedirs(LIVE, exist_ok=True)
    fig.savefig(os.path.join(LIVE, name), dpi=150, facecolor="white")


def cpu_waiting() -> None:
    summary = read("cpu_summary.csv")
    schedulers = ["AI-selected", "FCFS", "SJF", "Priority", "Round Robin"]
    types = ["CPU_BOUND", "IO_BOUND", "PRIORITY_CRITICAL", "BALANCED"]
    n = int(summary["n"].iloc[0])
    fig, ax = plt.subplots(figsize=(9, 4.5))
    width = 0.16
    for i, scheduler in enumerate(schedulers):
        sub = summary[summary.scheduler == scheduler].set_index("workload_type").loc[types]
        ax.bar(np.arange(len(types)) + i * width, sub["avg_waiting_time_mean"], width,
               yerr=sub["avg_waiting_time_ci95"], capsize=2, label=scheduler)
    ax.set_xticks(np.arange(len(types)) + 2 * width)
    ax.set_xticklabels(types)
    ax.set_ylabel("Average waiting time")
    ax.set_title(f"Average waiting time by workload type (mean, 95% CI, n={n})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    save(fig, "1_cpu_waiting.png")


def memory_faults() -> None:
    summary = read("memory_replacement_summary.csv")
    policies = list(dict.fromkeys(summary.policy))
    patterns = list(dict.fromkeys(summary.pattern))
    frame_counts = sorted(summary.frames.unique())
    fig, axes = plt.subplots(1, len(frame_counts), figsize=(11, 4), sharey=True)
    for ax, frames in zip(np.atleast_1d(axes), frame_counts):
        width = 0.16
        for i, policy in enumerate(policies):
            sub = summary[(summary.frames == frames) & (summary.policy == policy)].set_index("pattern").loc[patterns]
            ax.bar(np.arange(len(patterns)) + i * width, sub["page_faults_mean"], width,
                   yerr=sub["page_faults_ci95"], capsize=2, label=policy)
        ax.set_xticks(np.arange(len(patterns)) + 2 * width)
        ax.set_xticklabels(patterns)
        ax.set_title(f"{frames} frames")
        ax.set_ylabel("Demand page faults (of 500 references)")
    np.atleast_1d(axes)[0].legend(fontsize=7)
    fig.tight_layout()
    save(fig, "2_memory_page_faults.png")


def device_recall() -> None:
    d = read("model_selection_device.csv")
    d = d[~d.model.str.contains(r"\[")]
    severities = sorted(float(c.split()[-1]) for c in d.columns if c.startswith("recall_severity"))
    detectors = [
        ("transformer (in use)", "Transformer (in use)", "#2a78d6", "o"),
        ("one_class_svm", "One-Class SVM", "#eb6834", "s"),
        ("z_score_max", "z-score (max)", "#1baf7a", "^"),
        ("local_outlier_factor", "Local Outlier Factor", "#eda100", "D"),
        ("pca_reconstruction", "PCA reconstruction", "#e87ba4", "v"),
        ("isolation_forest", "Isolation Forest", "#6250d6", "x"),
    ]
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for model, label, color, marker in detectors:
        row = d[d.model == model].iloc[0]
        ax.plot(severities, [row[f"recall_severity {s:g}"] for s in severities], color=color,
                lw=2.5 if "in use" in model else 1.5, marker=marker, label=label)
    ax.set_xscale("log")
    ax.set_xticks(severities, [f"{s:g}" for s in severities])
    ax.minorticks_off()
    ax.set_ylim(-0.03, 1.05)
    ax.set_xlabel("Fault severity (1.0 = full simulated fault)")
    ax.set_ylabel("Recall")
    ax.set_title("Device anomaly detectors on milder faults")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    save(fig, "3_device_recall.png")


def _hbar(ax, df, col, err, title, xlabel, xlim, higher) -> None:
    df = df.sort_values(col, ascending=not higher).iloc[::-1]
    colors = ["#2a78d6" if "in use" in m else "#b9b8b1" for m in df.model]
    ax.barh(range(len(df)), df[col], xerr=df[err], color=colors, height=0.62, capsize=2)
    ax.set_yticks(range(len(df)), [m.replace("_", " ") for m in df.model])
    ax.set_xlim(*xlim)
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.grid(axis="x", alpha=0.3)
    for i, (v, e) in enumerate(zip(df[col], df[err])):
        ax.text(v + e + (xlim[1] - xlim[0]) * 0.01, i, f"{v:.3f}", va="center", fontsize=8)


def model_selection_cpu() -> None:
    burst = read("model_selection_cpu_burst.csv")
    workload = read("model_selection_cpu_workload.csv")
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7, 8))
    _hbar(a1, burst, "mae_mean", "mae_std", "(a) Burst time", "MAE (lower is better)", (0, 9), higher=False)
    _hbar(a2, workload, "macro_f1_mean", "macro_f1_std", "(b) Workload class", "Macro-F1 (higher is better)",
          (0.94, 1.0), higher=True)
    fig.tight_layout()
    save(fig, "4_model_selection_cpu.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Rerun the studies and show the headline graphs.")
    parser.add_argument("--no-rerun", action="store_true", help="redraw from the existing CSVs only")
    args = parser.parse_args()
    if not args.no_rerun:
        rerun()
    cpu_waiting()
    memory_faults()
    device_recall()
    model_selection_cpu()
    print(f"\nSaved copies in {LIVE}/. Close the graph windows to finish.")
    plt.show()


if __name__ == "__main__":
    main()
