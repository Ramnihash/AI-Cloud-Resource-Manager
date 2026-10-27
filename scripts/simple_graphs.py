"""Simple one-metric bar charts for the project review.

Usage (from the repository root):

    python -m scripts.simple_graphs            # rerun the studies, then show the graphs
    python -m scripts.simple_graphs --no-rerun # only redraw from the CSVs already in results/
    python -m scripts.simple_graphs --with-smart # also rerun the SMART comparison (about 3 more minutes)

Each chart is a plain bar chart with the value printed on every bar.
The model or method we use is drawn in blue, the alternatives in grey.

    1. CPU: average waiting time per scheduler (all 120 workloads)
    2. CPU: burst-time predictor error (MAE) per model
    3. CPU: workload classifier accuracy per model
    4. Memory: page faults per policy (locality pattern, 8 frames)
    5. Device: anomaly detectors, share of mild faults caught
    6. Device: SMART disk-failure model ROC-AUC per model

Copies are saved to results/simple/.
"""

import argparse
import os
import subprocess
import sys

import matplotlib.pyplot as plt
import pandas as pd

RESULTS = "results"
OUT = os.path.join(RESULTS, "simple")
CHOSEN = "#2a78d6"
OTHER = "#b9b8b1"

STUDIES = [
    ["experiments.cpu_eval"],
    ["experiments.memory_eval"],
    ["experiments.model_selection_eval", "--only", "cpu_burst,cpu_workload,device"],
]

NAMES = {
    "polynomial_ridge_deg3": "Polynomial ridge",
    "svr_rbf": "SVR",
    "mlp_64x64": "Neural net (MLP)",
    "hist_gradient_boosting": "Gradient boosting",
    "random_forest": "Random Forest",
    "random_forest_unlimited": "Random Forest (no depth limit)",
    "extra_trees": "Extra Trees",
    "knn_k5": "k-NN",
    "knn_k25": "k-NN",
    "gradient_boosting": "Gradient boosting (classic)",
    "decision_tree": "Decision tree",
    "linear_regression": "Linear regression",
    "svm_rbf": "SVM",
    "gaussian_naive_bayes": "Naive Bayes",
    "logistic_regression": "Logistic regression",
    "transformer": "Transformer",
    "one_class_svm": "One-Class SVM",
    "z_score_max": "z-score",
    "local_outlier_factor": "LOF",
    "pca_reconstruction": "PCA",
    "isolation_forest": "Isolation Forest",
}


def read(name: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(RESULTS, name))


def label(model: str) -> str:
    return NAMES.get(model.replace(" (in use)", ""), model)


def bar_chart(names, values, chosen, title, ylabel, filename, fmt="{:.2f}", ylim=None) -> None:
    """One plain bar chart: value on top of each bar, chosen bar in blue."""
    fig, ax = plt.subplots(figsize=(9, 5))
    chosen = {chosen} if isinstance(chosen, str) else set(chosen)
    colors = [CHOSEN if name in chosen else OTHER for name in names]
    bars = ax.bar(names, values, color=colors)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), fmt.format(value),
                ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_title(title, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=12)
    if ylim:
        ax.set_ylim(*ylim)
    ax.spines[["top", "right"]].set_visible(False)
    plt.xticks(rotation=20, ha="right", fontsize=11)
    fig.tight_layout()
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, filename), dpi=150)


def model_chart(csv, metric, title, ylabel, filename, lower_is_better, keep=None, fmt="{:.3f}", ylim=None) -> None:
    """Bar chart of a model-selection table, best model first."""
    df = read(csv)
    df = df[~df.model.str.contains(r"\[")]
    if keep:
        df = df[df.model.str.replace(" (in use)", "", regex=False).isin(keep)]
    df = df.sort_values(metric, ascending=lower_is_better)
    chosen = label(df[df.model.str.contains("in use")].model.iloc[0])
    bar_chart([label(m) for m in df.model], df[metric].tolist(), chosen, title, ylabel, filename, fmt, ylim)


def cpu_waiting() -> None:
    runs = read("cpu_runs.csv")
    order = ["AI-selected", "SJF", "FCFS", "Priority", "Round Robin"]
    means = runs.groupby("scheduler").avg_waiting_time.mean().loc[order]
    bar_chart(order, means.tolist(), "AI-selected",
              "CPU: average waiting time (120 workloads, lower is better)",
              "Average waiting time", "1_cpu_waiting.png", fmt="{:.1f}")


def memory_faults() -> None:
    summary = read("memory_replacement_summary.csv")
    sub = summary[(summary.pattern == "locality") & (summary.frames == 8)]
    order = ["FIFO", "LRU", "Predictive-protect", "Predictive-prefetch", "Optimal"]
    sub = sub.set_index("policy").loc[order]
    names = ["FIFO", "LRU", "Ours: protect", "Ours: prefetch", "Optimal (best possible)"]
    bar_chart(names, sub.page_faults_mean.tolist(), ["Ours: protect", "Ours: prefetch"],
              "Memory: page faults, locality pattern, 8 frames (lower is better)",
              "Page faults per 500 references", "4_memory_faults.png", fmt="{:.0f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Simple one-metric bar charts.")
    parser.add_argument("--no-rerun", action="store_true", help="redraw from the existing CSVs only")
    parser.add_argument("--with-smart", action="store_true", help="also rerun the SMART comparison (slow)")
    args = parser.parse_args()
    if not args.no_rerun:
        studies = STUDIES + ([["experiments.model_selection_eval", "--only", "smart"]] if args.with_smart else [])
        for study in studies:
            print("Running", " ".join(study), "...", flush=True)
            subprocess.run([sys.executable, "-m", *study], check=True)

    cpu_waiting()
    model_chart("model_selection_cpu_burst.csv", "mae_mean",
                "CPU: burst-time prediction error (lower is better)", "Mean absolute error",
                "2_burst_models.png", lower_is_better=True,
                keep=["polynomial_ridge_deg3", "svr_rbf", "mlp_64x64", "hist_gradient_boosting",
                      "random_forest", "decision_tree", "linear_regression"], fmt="{:.2f}")
    model_chart("model_selection_cpu_workload.csv", "accuracy_mean",
                "CPU: workload classifier accuracy (higher is better)", "Accuracy",
                "3_workload_models.png", lower_is_better=False,
                keep=["random_forest", "svm_rbf", "knn_k5", "hist_gradient_boosting",
                      "logistic_regression", "decision_tree"], fmt="{:.1%}", ylim=(0.95, 0.99))
    memory_faults()
    model_chart("model_selection_device.csv", "recall_severity 0.05",
                "Device: mild faults caught at 5% strength (higher is better)", "Recall (share of faults caught)",
                "5_device_detectors.png", lower_is_better=False, fmt="{:.2f}", ylim=(0, 1))
    model_chart("model_selection_smart.csv", "roc_auc_mean",
                "Device: SMART disk-failure models, real Backblaze data (higher is better)", "ROC-AUC",
                "6_smart_models.png", lower_is_better=False,
                keep=["random_forest", "logistic_regression", "hist_gradient_boosting",
                      "decision_tree", "extra_trees", "knn_k25", "gaussian_naive_bayes"], ylim=(0.6, 0.93))

    print(f"Saved copies in {OUT}/. Close the windows to finish.")
    plt.show()


if __name__ == "__main__":
    main()
