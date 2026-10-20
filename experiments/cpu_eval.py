"""CP7.1 CPU study: AI-selected scheduler vs FCFS / SJF / Priority / Round Robin.

Usage (from the repository root):

    python -m experiments.cpu_eval

Trains the burst predictor and workload classifier once (seed 42), then runs every
scheduler on N seeded 20-process workloads of each workload type. The evaluation
seeds start at 1000, so no evaluation workload shares a seed with the training data.

Writes to ``results/``:
  cpu_runs.csv        one row per (workload type, seed, scheduler)
  cpu_summary.csv     mean, std and 95% CI per (workload type, scheduler)
  cpu_paired.csv      AI-selected vs each fixed scheduler, paired (waiting, response, high-priority waiting)
  cpu_selection.csv   how often the classifier picked the right class / scheduler
  cpu_*.png           bar charts of waiting time and utilization
"""

from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd

from data.cpu.workload_classifier_data import WORKLOAD_CLASSES, generate_class_workload, label_batch
from modules.cpu.evaluation import evaluate_burst_predictor, evaluate_workload_classifier
from modules.cpu.fcfs import FCFSScheduler
from modules.cpu.meta_scheduler import CPUMetaScheduler
from modules.cpu.priority import PriorityScheduler
from modules.cpu.round_robin import RoundRobinScheduler
from modules.cpu.sjf import SJFScheduler

SEED = 42
RESULTS_DIR = "results"
N_WORKLOADS = 30
NUM_PROCESSES = 20
EVAL_SEED_START = 1000
QUANTUM = 2.0
METRICS = [
    "avg_waiting_time",
    "avg_turnaround_time",
    "avg_response_time",
    "cpu_utilization",
    "high_priority_waiting",
    "scheduling_overhead_ms",
    "wall_ms",
]
AI = "AI-selected"


def _fixed_schedulers() -> dict:
    return {
        "FCFS": FCFSScheduler(),
        "SJF": SJFScheduler(use_predicted_burst=True),
        "Priority": PriorityScheduler(),
        "Round Robin": RoundRobinScheduler(time_quantum=QUANTUM),
    }


def _high_priority_waiting(processes) -> float:
    """Mean waiting time of the most urgent processes (lowest priority number)."""
    top = min(p.priority for p in processes)
    return float(np.mean([p.waiting_time for p in processes if p.priority == top]))


def _row(workload_type, seed, scheduler, result, wall_ms, **extra):
    metrics = result.metrics
    return {
        "workload_type": workload_type,
        "seed": seed,
        "scheduler": scheduler,
        "avg_waiting_time": metrics.avg_waiting_time,
        "avg_turnaround_time": metrics.avg_turnaround_time,
        "avg_response_time": metrics.avg_response_time,
        "cpu_utilization": metrics.cpu_utilization,
        "high_priority_waiting": _high_priority_waiting(result.processes),
        "scheduling_overhead_ms": metrics.scheduling_overhead_ms,
        "wall_ms": wall_ms,
        **extra,
    }


def run_workloads(meta: CPUMetaScheduler, n: int = N_WORKLOADS) -> tuple[pd.DataFrame, pd.DataFrame]:
    fixed = _fixed_schedulers()
    runs, selections = [], []
    for workload_type in WORKLOAD_CLASSES:
        for k in range(n):
            seed = EVAL_SEED_START + k
            workload, label = generate_class_workload(workload_type, NUM_PROCESSES, seed=seed)

            t0 = time.perf_counter()
            result = meta.schedule([p.copy() for p in workload])
            wall = (time.perf_counter() - t0) * 1000.0
            runs.append(_row(workload_type, seed, AI, result, wall, effective=result.selected_scheduler))
            selections.append({
                "workload_type": workload_type,
                "seed": seed,
                "ground_truth_label": label,
                "predicted_class": result.predicted_class,
                "is_fallback": bool(result.is_fallback),
                "selected_scheduler": result.selected_scheduler,
                "class_correct": result.predicted_class == label,
            })

            for name, scheduler in fixed.items():
                batch = [p.copy() for p in workload]
                if any(p.predicted_burst_time is None for p in batch):
                    meta.burst_predictor.predict_workload(batch, inplace=True)
                t0 = time.perf_counter()
                res = scheduler.schedule(batch)
                wall = (time.perf_counter() - t0) * 1000.0
                runs.append(_row(workload_type, seed, name, res, wall, effective=name))
    return pd.DataFrame(runs), pd.DataFrame(selections)


def summarise(runs: pd.DataFrame) -> pd.DataFrame:
    from scipy import stats

    rows = []
    for (workload_type, scheduler), group in runs.groupby(["workload_type", "scheduler"], sort=False):
        row = {"workload_type": workload_type, "scheduler": scheduler, "n": len(group)}
        for metric in METRICS:
            values = group[metric].to_numpy(dtype=float)
            sd = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            half = float(stats.t.ppf(0.975, len(values) - 1) * sd / np.sqrt(len(values))) if len(values) > 1 else 0.0
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = sd
            row[f"{metric}_ci95"] = half
        rows.append(row)
    return pd.DataFrame(rows)


def paired_comparison(runs: pd.DataFrame, metric: str = "avg_waiting_time") -> pd.DataFrame:
    """AI-selected vs each fixed scheduler on the same workloads (negative diff = AI better)."""
    from scipy import stats

    rows = []
    ai = runs[runs.scheduler == AI].set_index(["workload_type", "seed"])
    for scheduler in [s for s in runs.scheduler.unique() if s != AI]:
        other = runs[runs.scheduler == scheduler].set_index(["workload_type", "seed"])
        for workload_type in list(WORKLOAD_CLASSES) + ["ALL"]:
            idx = ai.index if workload_type == "ALL" else ai.index[ai.index.get_level_values(0) == workload_type]
            a = ai.loc[idx, metric].to_numpy()
            b = other.loc[idx, metric].to_numpy()
            diff = a - b
            if np.allclose(diff, 0):
                p_value = 1.0
            else:
                p_value = float(stats.ttest_rel(a, b).pvalue)
            rows.append({
                "metric": metric,
                "baseline": scheduler,
                "workload_type": workload_type,
                "n": len(diff),
                "mean_ai": float(a.mean()),
                "mean_baseline": float(b.mean()),
                "mean_diff": float(diff.mean()),
                "pct_change": float(100.0 * diff.mean() / b.mean()) if b.mean() else float("nan"),
                "ai_better_or_equal_pct": float(100.0 * (diff <= 1e-9).mean()),
                "p_value_paired_t": p_value,
            })
    return pd.DataFrame(rows)


def selection_summary(selections: pd.DataFrame) -> pd.DataFrame:
    grouped = selections.groupby("workload_type")
    out = grouped.agg(
        workloads=("seed", "count"),
        class_accuracy=("class_correct", "mean"),
        fallback_rate=("is_fallback", "mean"),
    ).reset_index()
    counts = selections.groupby(["workload_type", "selected_scheduler"]).size().unstack(fill_value=0)
    return out.merge(counts.reset_index(), on="workload_type")


def make_plots(summary: pd.DataFrame, out: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    schedulers = [AI, "FCFS", "SJF", "Priority", "Round Robin"]
    types = list(WORKLOAD_CLASSES)
    for metric, label, name in [
        ("avg_waiting_time", "Average waiting time", "cpu_waiting.png"),
        ("cpu_utilization", "CPU utilization (%)", "cpu_utilization.png"),
    ]:
        fig, ax = plt.subplots(figsize=(9, 4.5))
        width = 0.16
        for i, scheduler in enumerate(schedulers):
            sub = summary[summary.scheduler == scheduler].set_index("workload_type").loc[types]
            ax.bar(
                np.arange(len(types)) + i * width,
                sub[f"{metric}_mean"],
                width,
                yerr=sub[f"{metric}_ci95"],
                capsize=2,
                label=scheduler,
            )
        ax.set_xticks(np.arange(len(types)) + 2 * width)
        ax.set_xticklabels(types)
        ax.set_ylabel(label)
        ax.set_title(f"{label} by workload type (mean, 95% CI, n={N_WORKLOADS})")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(out, name), dpi=150)
        plt.close(fig)


def build_meta(seed: int = SEED) -> CPUMetaScheduler:
    predictor = evaluate_burst_predictor(seed=seed)["predictor_instance"]
    classifier = evaluate_workload_classifier(predictor=predictor, seed=seed)["classifier_instance"]
    return CPUMetaScheduler(classifier=classifier, burst_predictor=predictor)


def run(meta: CPUMetaScheduler | None = None, n: int = N_WORKLOADS, out: str = RESULTS_DIR) -> dict[str, pd.DataFrame]:
    meta = meta or build_meta()
    os.makedirs(out, exist_ok=True)
    runs, selections = run_workloads(meta, n)
    summary = summarise(runs)
    paired = pd.concat(
        [paired_comparison(runs, m) for m in ("avg_waiting_time", "avg_response_time", "high_priority_waiting")],
        ignore_index=True,
    )
    selection = selection_summary(selections)
    runs.to_csv(os.path.join(out, "cpu_runs.csv"), index=False)
    summary.to_csv(os.path.join(out, "cpu_summary.csv"), index=False)
    paired.to_csv(os.path.join(out, "cpu_paired.csv"), index=False)
    selection.to_csv(os.path.join(out, "cpu_selection.csv"), index=False)
    make_plots(summary, out)
    return {"runs": runs, "summary": summary, "paired": paired, "selection": selection}


def main() -> None:
    results = run()
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    s = results["summary"]
    cols = ["workload_type", "scheduler", "avg_waiting_time_mean", "avg_waiting_time_std",
            "avg_turnaround_time_mean", "avg_response_time_mean", "high_priority_waiting_mean", "cpu_utilization_mean",
            "scheduling_overhead_ms_mean", "wall_ms_mean"]
    print("\nPer workload type (mean over workloads):")
    print(s[cols].round(3).to_string(index=False))
    print("\nAI-selected vs each fixed scheduler (paired on the same workloads; negative = AI better):")
    print(results["paired"].round(4).to_string(index=False))
    print("\nClassifier choices:")
    print(results["selection"].round(3).to_string(index=False))
    print(f"\nCSVs and plots written to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
