"""CP7.2 Memory study: page replacement, page prediction, leak and pressure detection.

Usage (from the repository root):

    python -m experiments.memory_eval

Part A  FIFO / LRU / Optimal / predictive page replacement on fixed reference strings.
Part B  Memory-leak detector against injected leaks (labelled ground truth).
Part C  Memory-pressure forecaster against injected pressure ramps.

The predictive policies are defined in this file (the project's action layer only emits
PREFETCH / PROTECT directives and leaves replacement to the classical algorithms), so
they are an evaluation harness, not a replacement for the memory module:

  Predictive-protect   LRU, plus: when the Markov top-1 next page is already resident,
                       refresh its recency so it is not evicted (the PROTECT action).
  Predictive-prefetch  Predictive-protect, plus: when the predicted page is not resident,
                       load it ahead of time (the PREFETCH action), evicting the LRU page.

A prefetch is not a demand fault, so Part A reports demand faults, prefetch loads and
their sum (total page loads). The Markov model is trained on the first 1000 references of
each string and every policy is scored on the next 500, starting from empty frames.

Writes to ``results/``: memory_replacement_runs.csv, memory_replacement_summary.csv,
memory_replacement_paired.csv, memory_prediction_accuracy.csv, memory_leak_metrics.csv,
memory_leak_delay.csv, memory_pressure_metrics.csv and memory_*.png.
"""

from __future__ import annotations

import os
import time
from collections import OrderedDict

import numpy as np
import pandas as pd

from ml.memory.markov_predictor import MarkovPagePredictor
from ml.memory.memory_leak_detector import MemoryLeakDetector
from ml.memory.memory_pressure_forecaster import MemoryPressureForecaster
from ml.memory.page_reference_generator import PageReferenceGenerator
from modules.memory.memory_health_action import MemoryHealthActionLayer
from modules.memory.page_replacement import FIFOPageReplacement, LRUPageReplacement, OptimalPageReplacement

SEED = 42
RESULTS_DIR = "results"
PATTERNS = ["uniform", "locality", "sequential"]
NUM_PAGES = 20
TRAIN_LEN = 1000
EVAL_LEN = 500
FRAME_COUNTS = [4, 8]
N_STRINGS = 30
EVAL_SEED_START = 2000
POLICIES = ["FIFO", "LRU", "Optimal", "Predictive-protect", "Predictive-prefetch"]


# ---------------------------------------------------------------- Part A

def simulate_predictive(reference: list, frames: int, predictor: MarkovPagePredictor, prefetch: bool) -> dict:
    """LRU with Markov-driven PROTECT (and optionally PREFETCH) actions."""
    resident: OrderedDict[int, bool] = OrderedDict()  # page -> brought in by prefetch and not yet used
    faults = hits = prefetches = useful = 0

    def evict_for(page: int) -> None:
        if len(resident) >= frames:
            resident.popitem(last=False)

    t0 = time.perf_counter()
    for page in reference:
        if page in resident:
            hits += 1
            if resident[page]:
                useful += 1
            resident[page] = False
            resident.move_to_end(page)
        else:
            faults += 1
            evict_for(page)
            resident[page] = False
        candidates = predictor.predict(page, top_k=1) if page in predictor.transition_table else []
        if not candidates:
            continue
        nxt = candidates[0]["page"]
        if nxt in resident:
            resident.move_to_end(nxt)  # PROTECT
        elif prefetch:
            evict_for(nxt)  # PREFETCH
            resident[nxt] = True
            prefetches += 1
    wall = (time.perf_counter() - t0) * 1000.0
    n = len(reference)
    return {
        "page_faults": faults,
        "page_hits": hits,
        "hit_ratio": hits / n,
        "prefetches": prefetches,
        "useful_prefetches": useful,
        "execution_time_ms": wall,
    }


def run_replacement(n_strings: int = N_STRINGS) -> tuple[pd.DataFrame, pd.DataFrame]:
    gen = PageReferenceGenerator()
    rows, accuracy = [], []
    for pattern in PATTERNS:
        for k in range(n_strings):
            seed = EVAL_SEED_START + k
            full = gen.generate(NUM_PAGES, TRAIN_LEN + EVAL_LEN, pattern, seed=seed)["sequence"]
            history, reference = full[:TRAIN_LEN], full[TRAIN_LEN:]
            predictor = MarkovPagePredictor()
            predictor.train(history)
            acc = predictor.evaluate(reference, top_k=3)
            accuracy.append({
                "pattern": pattern, "seed": seed,
                "top1_accuracy": acc["top1_accuracy"], "top3_accuracy": acc["topk_accuracy"],
            })
            for frames in FRAME_COUNTS:
                results = {
                    "FIFO": FIFOPageReplacement(frames).run(reference),
                    "LRU": LRUPageReplacement(frames).run(reference),
                    "Optimal": OptimalPageReplacement(frames).run(reference),
                    "Predictive-protect": simulate_predictive(reference, frames, predictor, prefetch=False),
                    "Predictive-prefetch": simulate_predictive(reference, frames, predictor, prefetch=True),
                }
                for policy, r in results.items():
                    prefetches = r.get("prefetches", 0)
                    rows.append({
                        "pattern": pattern,
                        "frames": frames,
                        "seed": seed,
                        "policy": policy,
                        "page_faults": r["page_faults"],
                        "hit_ratio": r["hit_ratio"],
                        "prefetches": prefetches,
                        "total_loads": r["page_faults"] + prefetches,
                        "prefetch_useful_rate": (r.get("useful_prefetches", 0) / prefetches) if prefetches else float("nan"),
                        "execution_time_ms": r["execution_time_ms"],
                    })
    return pd.DataFrame(rows), pd.DataFrame(accuracy)


def _mean_ci(values: np.ndarray) -> tuple[float, float, float]:
    from scipy import stats

    values = values[~np.isnan(values)]
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    sd = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    half = float(stats.t.ppf(0.975, len(values) - 1) * sd / np.sqrt(len(values))) if len(values) > 1 else 0.0
    return float(values.mean()), sd, half


def summarise_replacement(runs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (pattern, frames, policy), g in runs.groupby(["pattern", "frames", "policy"], sort=False):
        row = {"pattern": pattern, "frames": frames, "policy": policy, "n": len(g)}
        for col in ["page_faults", "hit_ratio", "prefetches", "total_loads", "prefetch_useful_rate", "execution_time_ms"]:
            m, sd, ci = _mean_ci(g[col].to_numpy(dtype=float))
            row[f"{col}_mean"], row[f"{col}_std"], row[f"{col}_ci95"] = m, sd, ci
        rows.append(row)
    return pd.DataFrame(rows)


def _paired_p(a: np.ndarray, b: np.ndarray) -> float:
    import warnings

    from scipy import stats

    if np.allclose(a - b, 0):
        return 1.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return float(stats.ttest_rel(a, b).pvalue)


def paired_replacement(runs: pd.DataFrame) -> pd.DataFrame:
    """Predictive policies vs LRU on identical strings (negative = fewer faults than LRU)."""
    rows = []
    for (pattern, frames), g in runs.groupby(["pattern", "frames"], sort=False):
        lru = g[g.policy == "LRU"].set_index("seed")
        for policy in ["Predictive-protect", "Predictive-prefetch"]:
            other = g[g.policy == policy].set_index("seed")
            for metric in ["page_faults", "total_loads"]:
                a, b = other[metric].to_numpy(dtype=float), lru.loc[other.index, metric].to_numpy(dtype=float)
                diff = a - b
                p = _paired_p(a, b)
                rows.append({
                    "pattern": pattern, "frames": frames, "policy": policy, "metric": metric,
                    "mean_policy": float(a.mean()), "mean_lru": float(b.mean()),
                    "mean_diff": float(diff.mean()),
                    "pct_change": float(100 * diff.mean() / b.mean()) if b.mean() else float("nan"),
                    "better_or_equal_pct": float(100 * (diff <= 1e-9).mean()),
                    "p_value_paired_t": p,
                })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- Part B

LEAK_TYPES = {
    "leak_fast": True,
    "leak_slow": True,
    "flat": False,
    "sawtooth": False,
    "random_walk": False,
    "one_time_jump": False,
}
BASE_RSS = 200_000_000.0
LEAK_SEQ_LEN = 60
LEAK_ONSET = 30


def make_rss(kind: str, rng: np.random.Generator, length: int = LEAK_SEQ_LEN, onset: int = LEAK_ONSET) -> np.ndarray:
    steps = np.arange(length)
    noise = rng.normal(0, 4096.0, length)
    if kind == "leak_fast":
        slope = rng.uniform(20_000, 80_000)
        return BASE_RSS + noise + np.where(steps >= onset, (steps - onset) * slope, 0.0)
    if kind == "leak_slow":
        slope = rng.uniform(4_000, 12_000)
        return BASE_RSS + noise + np.where(steps >= onset, (steps - onset) * slope, 0.0)
    if kind == "flat":
        return BASE_RSS + noise
    if kind == "sawtooth":  # allocate then free in cycles
        period = int(rng.integers(8, 20))
        return BASE_RSS + noise + (steps % period) * rng.uniform(20_000, 60_000)
    if kind == "random_walk":
        return BASE_RSS + np.cumsum(rng.normal(0, 15_000, length))
    if kind == "one_time_jump":
        return BASE_RSS + noise + np.where(steps >= onset, rng.uniform(2e6, 8e6), 0.0)
    raise ValueError(kind)


def _snapshots(rss: np.ndarray, pid: int = 1234) -> list[dict]:
    return [{"proc_pid": pid, "proc_rss_bytes": float(v)} for v in rss]


def run_leak(n_per_type: int = 40) -> tuple[pd.DataFrame, pd.DataFrame]:
    detector = MemoryLeakDetector(window_size=10)
    rng = np.random.default_rng(SEED)
    rows, delays = [], []
    for kind, is_leak in LEAK_TYPES.items():
        for k in range(n_per_type):
            rss = make_rss(kind, rng)
            snaps = _snapshots(rss)
            flagged = [
                bool(detector.detect(snaps[end - 10:end])["is_leak_detected"])
                for end in range(10, LEAK_SEQ_LEN + 1)
            ]
            # sequence-level decision on the final window (what the live endpoint sees)
            final = flagged[-1]
            rows.append({"type": kind, "truth_leak": is_leak, "flagged_final_window": final, "seed_index": k})
            if is_leak:
                after = [i for i, f in enumerate(flagged, start=10) if i > LEAK_ONSET and f]
                delays.append({
                    "type": kind, "seed_index": k, "detected": bool(after),
                    "delay_steps": (after[0] - LEAK_ONSET) if after else None,
                })
            else:
                delays.append({"type": kind, "seed_index": k, "false_alarm_any_window": any(flagged)})
    return pd.DataFrame(rows), pd.DataFrame(delays)


def summarise_leak(rows: pd.DataFrame) -> pd.DataFrame:
    def metrics(frame: pd.DataFrame) -> dict:
        tp = int(((frame.truth_leak) & (frame.flagged_final_window)).sum())
        fp = int(((~frame.truth_leak) & (frame.flagged_final_window)).sum())
        fn = int(((frame.truth_leak) & (~frame.flagged_final_window)).sum())
        tn = int(((~frame.truth_leak) & (~frame.flagged_final_window)).sum())
        precision = tp / (tp + fp) if tp + fp else float("nan")
        recall = tp / (tp + fn) if tp + fn else float("nan")
        f1 = 2 * precision * recall / (precision + recall) if tp and precision + recall else (0.0 if tp + fp + fn else float("nan"))
        return {
            "sequences": len(frame), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": precision, "recall": recall, "f1": f1,
            "false_positive_rate": fp / (fp + tn) if fp + tn else float("nan"),
        }

    out = [{"type": "ALL", **metrics(rows)}]
    for kind in LEAK_TYPES:
        out.append({"type": kind, **metrics(rows[rows.type == kind])})
    return pd.DataFrame(out)


# ---------------------------------------------------------------- Part C

PRESSURE_THRESHOLD = 90.0
PRESSURE_LEN = 80


def make_ram(kind: str, rng: np.random.Generator, length: int = PRESSURE_LEN) -> tuple[np.ndarray, int | None]:
    """Return a ram_percent series and the step at which it first reaches the threshold."""
    steps = np.arange(length)
    if kind == "ramp":
        rate = rng.uniform(0.4, 1.2)
        start = rng.uniform(55, 65)
        series = start + rate * steps + rng.normal(0, 0.4, length)
    elif kind == "stable":
        series = 55 + 8 * np.sin(steps / rng.uniform(6, 12)) + rng.normal(0, 0.8, length)
    elif kind == "high_stable":  # busy but below the threshold
        series = 80 + 3 * np.sin(steps / rng.uniform(6, 12)) + rng.normal(0, 0.6, length)
    else:
        raise ValueError(kind)
    series = np.clip(series, 0.0, 100.0)
    crossing = np.flatnonzero(series >= PRESSURE_THRESHOLD)
    return series, int(crossing[0]) if len(crossing) else None


def _ram_snaps(series: np.ndarray) -> list[dict]:
    return [{"ram_percent": float(v)} for v in series]


def run_pressure(n_per_type: int = 40, window: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(SEED + 7)
    train_series = np.concatenate([make_ram(k, rng)[0] for k in ("ramp", "stable", "high_stable") for _ in range(10)])
    forecaster = MemoryPressureForecaster(window_size=window)
    forecaster.train(_ram_snaps(train_series))
    health = MemoryHealthActionLayer(pressure_threshold=PRESSURE_THRESHOLD)

    holdout = {k: [make_ram(k, rng) for _ in range(n_per_type)] for k in ("ramp", "stable", "high_stable")}
    err_rows, det_rows = [], []
    for kind, items in holdout.items():
        maes = [forecaster.evaluate(_ram_snaps(s))["MAE"] for s, _ in items]
        err_rows.append({"type": kind, "mae_mean": float(np.mean(maes)), "mae_std": float(np.std(maes, ddof=1))})
        for k, (series, crossing) in enumerate(items):
            first_alert = None
            for end in range(window, len(series)):
                pred = forecaster.predict(_ram_snaps(series[end - window:end]))
                action = health.evaluate_and_act(pressure_prediction=pred)["action"]
                if action == "ALERT_HIGH_PRESSURE":
                    first_alert = end - 1  # last observed step when the alert is raised
                    break
            row = {"type": kind, "index": k, "truth_crossing": crossing, "first_alert": first_alert}
            if crossing is not None:
                row["alerted_before_crossing"] = first_alert is not None and first_alert < crossing
                row["lead_steps"] = (crossing - first_alert) if first_alert is not None else None  # > 0 = early warning
            else:
                row["false_alert"] = first_alert is not None
            det_rows.append(row)
    return pd.DataFrame(err_rows), pd.DataFrame(det_rows)


def summarise_pressure(errors: pd.DataFrame, det: pd.DataFrame) -> pd.DataFrame:
    rows = []
    ramp = det[det.type == "ramp"]
    rows.append({
        "scenario": "ramp to threshold (positive)",
        "series": len(ramp),
        "alerted_any_pct": float(100 * ramp.first_alert.notna().mean()),
        "alerted_before_crossing_pct": float(100 * ramp.alerted_before_crossing.mean()),
        "mean_lead_steps": float(ramp.lead_steps.dropna().mean()),
        "std_lead_steps": float(ramp.lead_steps.dropna().std(ddof=1)),
        "false_alert_pct": float("nan"),
        "forecast_mae": float(errors[errors.type == "ramp"].mae_mean.iloc[0]),
    })
    for kind in ("stable", "high_stable"):
        sub = det[det.type == kind]
        rows.append({
            "scenario": f"{kind} (negative)",
            "series": len(sub),
            "alerted_any_pct": float("nan"),
            "alerted_before_crossing_pct": float("nan"),
            "mean_lead_steps": float("nan"),
            "std_lead_steps": float("nan"),
            "false_alert_pct": float(100 * sub.false_alert.mean()),
            "forecast_mae": float(errors[errors.type == kind].mae_mean.iloc[0]),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- plots and driver

def make_plots(summary: pd.DataFrame, accuracy: pd.DataFrame, leak: pd.DataFrame, out: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(FRAME_COUNTS), figsize=(11, 4), sharey=True)
    for ax, frames in zip(np.atleast_1d(axes), FRAME_COUNTS):
        width = 0.16
        for i, policy in enumerate(POLICIES):
            sub = summary[(summary.frames == frames) & (summary.policy == policy)].set_index("pattern").loc[PATTERNS]
            ax.bar(np.arange(len(PATTERNS)) + i * width, sub["page_faults_mean"], width,
                   yerr=sub["page_faults_ci95"], capsize=2, label=policy)
        ax.set_xticks(np.arange(len(PATTERNS)) + 2 * width)
        ax.set_xticklabels(PATTERNS)
        ax.set_title(f"{frames} frames")
        ax.set_ylabel("Demand page faults (of 500 references)")
    np.atleast_1d(axes)[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "memory_page_faults.png"), dpi=150)
    plt.close(fig)

    acc = accuracy.groupby("pattern")[["top1_accuracy", "top3_accuracy"]].agg(["mean", "std"])
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(PATTERNS))
    ax.bar(x - 0.18, [acc.loc[p, ("top1_accuracy", "mean")] for p in PATTERNS], 0.36,
           yerr=[acc.loc[p, ("top1_accuracy", "std")] for p in PATTERNS], capsize=2, label="top-1")
    ax.bar(x + 0.18, [acc.loc[p, ("top3_accuracy", "mean")] for p in PATTERNS], 0.36,
           yerr=[acc.loc[p, ("top3_accuracy", "std")] for p in PATTERNS], capsize=2, label="top-3")
    ax.set_xticks(x)
    ax.set_xticklabels(PATTERNS)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Next-page accuracy")
    ax.set_title("Markov next-page prediction (mean, std over strings)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out, "memory_prediction_accuracy.png"), dpi=150)
    plt.close(fig)

    per = leak[leak.type != "ALL"]
    fig, ax = plt.subplots(figsize=(7, 4))
    flagged = per.assign(rate=np.where(per.tp + per.fn > 0, per.recall, per.false_positive_rate))
    ax.bar(flagged.type, flagged.rate)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Flag rate (recall for leaks, false-positive rate otherwise)")
    ax.set_title("Leak detector on injected scenarios")
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "memory_leak.png"), dpi=150)
    plt.close(fig)


def run(out: str = RESULTS_DIR, n_strings: int = N_STRINGS, n_per_type: int = 40) -> dict[str, pd.DataFrame]:
    os.makedirs(out, exist_ok=True)
    runs, accuracy = run_replacement(n_strings)
    summary = summarise_replacement(runs)
    paired = paired_replacement(runs)
    leak_rows, leak_delay = run_leak(n_per_type)
    leak = summarise_leak(leak_rows)
    errors, det = run_pressure(n_per_type)
    pressure = summarise_pressure(errors, det)

    runs.to_csv(os.path.join(out, "memory_replacement_runs.csv"), index=False)
    summary.to_csv(os.path.join(out, "memory_replacement_summary.csv"), index=False)
    paired.to_csv(os.path.join(out, "memory_replacement_paired.csv"), index=False)
    accuracy.to_csv(os.path.join(out, "memory_prediction_accuracy.csv"), index=False)
    leak.to_csv(os.path.join(out, "memory_leak_metrics.csv"), index=False)
    leak_delay.to_csv(os.path.join(out, "memory_leak_delay.csv"), index=False)
    pressure.to_csv(os.path.join(out, "memory_pressure_metrics.csv"), index=False)
    make_plots(summary, accuracy, leak, out)
    return {
        "runs": runs, "summary": summary, "paired": paired, "accuracy": accuracy,
        "leak": leak, "leak_delay": leak_delay, "pressure": pressure,
    }


def main() -> None:
    r = run()
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    s = r["summary"]
    print("\nA. Page replacement (mean over strings): demand faults, hit ratio, prefetch loads, runtime")
    print(s[["pattern", "frames", "policy", "page_faults_mean", "page_faults_std", "hit_ratio_mean",
             "prefetches_mean", "total_loads_mean", "prefetch_useful_rate_mean", "execution_time_ms_mean"]]
          .round(3).to_string(index=False))
    print("\nA. Predictive vs LRU, paired on the same strings (negative = fewer than LRU):")
    print(r["paired"].round(4).to_string(index=False))
    print("\nA. Markov next-page accuracy:")
    print(r["accuracy"].groupby("pattern")[["top1_accuracy", "top3_accuracy"]].agg(["mean", "std"]).round(3).to_string())
    print("\nB. Leak detector vs injected ground truth (final 10-sample window):")
    print(r["leak"].round(3).to_string(index=False))
    ld = r["leak_delay"]
    pos = ld[ld.type.isin(["leak_fast", "leak_slow"])]
    print("\nB. Leak detection delay (steps after the leak starts):")
    print(pos.groupby("type").agg(detected=("detected", "mean"), mean_delay=("delay_steps", "mean"),
                                 std_delay=("delay_steps", "std")).round(2).to_string())
    neg = ld[~ld.type.isin(["leak_fast", "leak_slow"])]
    print("   False alarm in any window of the stream (negatives):")
    print(neg.groupby("type")["false_alarm_any_window"].mean().round(3).to_string())
    print("\nC. Pressure forecaster vs injected ramps (threshold 90%):")
    print(r["pressure"].round(3).to_string(index=False))
    print(f"\nCSVs and plots written to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
