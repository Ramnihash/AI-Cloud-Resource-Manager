"""CP7.2 Memory study, part D: the Transformer page predictor evaluated on its own.

Usage (from the repository root, with PyTorch installed):

    python -m experiments.transformer_page_eval

Uses the same reference strings as part A of ``experiments/memory_eval.py``
(3 patterns x 30 seeds, 20 pages; first 1000 references train, next 500 are
scored) and compares, on identical strings:

  Markov                      first-order transition table (part A's predictor)
  Transformer (API config)    context 2, d_model 16, 20 epochs: what /api/v1/memory/analysis runs
  Transformer (trained)       context 4, d_model 32, 300 epochs at lr 0.01

Part D1 next-page accuracy (top-1 / top-3), training time and prediction latency.
Part D2 page replacement: LRU with PROTECT / PREFETCH driven by each predictor
        (same harness as part A), demand faults and total page loads.
Part D3 request-sized strings: the Memory page sends a short reference string and
        the backend trains on it per request. 300 seeded strings of 30 references
        per pattern; each model trains on the first 29 and predicts the 30th.

Writes to ``results/``: memory_transformer_accuracy.csv, memory_transformer_replacement.csv,
memory_transformer_paired.csv, memory_transformer_request.csv, memory_transformer_accuracy.png.
"""

from __future__ import annotations

import os
import time
from collections import OrderedDict
from typing import Callable

import numpy as np
import pandas as pd
import torch

from experiments.memory_eval import (
    EVAL_LEN,
    EVAL_SEED_START,
    FRAME_COUNTS,
    N_STRINGS,
    NUM_PAGES,
    PATTERNS,
    RESULTS_DIR,
    TRAIN_LEN,
    _paired_p,
)
from ml.memory.markov_predictor import MarkovPagePredictor
from ml.memory.page_reference_generator import PageReferenceGenerator
from ml.memory.transformer_predictor import TransformerPagePredictor
from modules.memory.page_replacement import LRUPageReplacement

TRANSFORMERS = {
    "Transformer (API config)": {"context_len": 2, "d_model": 16, "nhead": 2, "epochs": 20, "lr": 1e-3},
    "Transformer (trained)": {"context_len": 4, "d_model": 32, "nhead": 2, "epochs": 300, "lr": 1e-2},
}
PREDICTORS = ["Markov", *TRANSFORMERS]
REQUEST_LEN = 30
REQUEST_STRINGS = 300
REQUEST_SEED_START = 5000


def markov_top(train: list[int], sequence: list[int], k: int = 3) -> tuple[list[list[int]], float]:
    """Top-k next-page guesses after each position of ``sequence`` (index i predicts i+1)."""
    t0 = time.perf_counter()
    predictor = MarkovPagePredictor()
    predictor.train(train)
    train_ms = (time.perf_counter() - t0) * 1000.0
    out = []
    for page in sequence:
        cands = predictor.predict(page, top_k=k) if page in predictor.transition_table else []
        out.append([c["page"] for c in cands])
    return out, train_ms


def transformer_top(train: list[int], sequence: list[int], cfg: dict, k: int = 3,
                    seed: int = 42) -> tuple[list[list[int]], float, float]:
    """Same as ``markov_top`` for a Transformer; positions without a full context get no guess."""
    torch.set_num_threads(1)
    predictor = TransformerPagePredictor(num_pages=NUM_PAGES, context_len=cfg["context_len"],
                                         d_model=cfg["d_model"], nhead=cfg["nhead"], seed=seed)
    t0 = time.perf_counter()
    predictor.train(train, epochs=cfg["epochs"], lr=cfg["lr"])
    train_ms = (time.perf_counter() - t0) * 1000.0
    c = cfg["context_len"]
    # The context for position i ends at i, so it may start in the training string.
    full = list(train[-(c - 1):]) + list(sequence) if c > 1 else list(sequence)
    contexts = [full[i:i + c] for i in range(len(sequence))]
    predictor.model.eval()
    t1 = time.perf_counter()
    with torch.no_grad():
        logits = predictor.model(torch.tensor(contexts, dtype=torch.long))
        top = torch.topk(logits, min(k, NUM_PAGES), dim=-1).indices.tolist()
    latency_ms = (time.perf_counter() - t1) * 1000.0 / max(len(contexts), 1)
    return top, train_ms, latency_ms


def accuracy(top: list[list[int]], sequence: list[int]) -> tuple[float, float]:
    hits1 = hits3 = 0
    n = len(sequence) - 1
    for i in range(n):
        guess, actual = top[i], sequence[i + 1]
        hits1 += bool(guess) and guess[0] == actual
        hits3 += actual in guess
    return hits1 / n, hits3 / n


def simulate(reference: list[int], frames: int, next_page: Callable[[int], int | None], prefetch: bool) -> dict:
    """LRU with PROTECT (and optionally PREFETCH) of the predicted next page (part A's harness)."""
    resident: OrderedDict[int, bool] = OrderedDict()
    faults = prefetches = useful = 0
    for i, page in enumerate(reference):
        if page in resident:
            useful += resident[page]
            resident[page] = False
            resident.move_to_end(page)
        else:
            faults += 1
            if len(resident) >= frames:
                resident.popitem(last=False)
            resident[page] = False
        nxt = next_page(i)
        if nxt is None:
            continue
        if nxt in resident:
            resident.move_to_end(nxt)
        elif prefetch:
            if len(resident) >= frames:
                resident.popitem(last=False)
            resident[nxt] = True
            prefetches += 1
    return {"page_faults": faults, "prefetches": prefetches, "useful_prefetches": useful}


def run_long(n_strings: int = N_STRINGS) -> tuple[pd.DataFrame, pd.DataFrame]:
    gen = PageReferenceGenerator()
    acc_rows, rep_rows = [], []
    for pattern in PATTERNS:
        for s in range(n_strings):
            seed = EVAL_SEED_START + s
            full = gen.generate(NUM_PAGES, TRAIN_LEN + EVAL_LEN, pattern, seed=seed)["sequence"]
            train, ref = full[:TRAIN_LEN], full[TRAIN_LEN:]
            tops = {}
            top, train_ms = markov_top(train, ref)
            tops["Markov"] = top
            a1, a3 = accuracy(top, ref)
            acc_rows.append({"pattern": pattern, "seed": seed, "predictor": "Markov", "top1_accuracy": a1,
                             "top3_accuracy": a3, "train_ms": train_ms, "predict_ms": float("nan")})
            for name, cfg in TRANSFORMERS.items():
                top, train_ms, latency = transformer_top(train, ref, cfg)
                tops[name] = top
                a1, a3 = accuracy(top, ref)
                acc_rows.append({"pattern": pattern, "seed": seed, "predictor": name, "top1_accuracy": a1,
                                 "top3_accuracy": a3, "train_ms": train_ms, "predict_ms": latency})
            for frames in FRAME_COUNTS:
                lru = LRUPageReplacement(frames).run(ref)
                rep_rows.append({"pattern": pattern, "frames": frames, "seed": seed, "predictor": "none",
                                 "policy": "LRU", "page_faults": lru["page_faults"], "prefetches": 0,
                                 "total_loads": lru["page_faults"], "prefetch_useful_rate": float("nan")})
                for name, top in tops.items():
                    def next_page(i: int, top=top) -> int | None:
                        return top[i][0] if top[i] else None
                    for policy, prefetch in (("Predictive-protect", False), ("Predictive-prefetch", True)):
                        r = simulate(ref, frames, next_page, prefetch)
                        rep_rows.append({
                            "pattern": pattern, "frames": frames, "seed": seed, "predictor": name, "policy": policy,
                            "page_faults": r["page_faults"], "prefetches": r["prefetches"],
                            "total_loads": r["page_faults"] + r["prefetches"],
                            "prefetch_useful_rate": r["useful_prefetches"] / r["prefetches"] if r["prefetches"] else float("nan"),
                        })
    return pd.DataFrame(acc_rows), pd.DataFrame(rep_rows)


def paired(rep: pd.DataFrame) -> pd.DataFrame:
    """Each predictor-driven policy vs LRU, and Transformer vs Markov, on the same strings."""
    rows = []
    for (pattern, frames), g in rep.groupby(["pattern", "frames"], sort=False):
        lru = g[g.policy == "LRU"].set_index("seed")
        for policy in ("Predictive-protect", "Predictive-prefetch"):
            markov = g[(g.policy == policy) & (g.predictor == "Markov")].set_index("seed")
            for name in PREDICTORS:
                other = g[(g.policy == policy) & (g.predictor == name)].set_index("seed")
                for base_name, base in (("LRU", lru), ("Markov", markov)):
                    if name == base_name:
                        continue
                    for metric in ("page_faults", "total_loads"):
                        a = other[metric].to_numpy(dtype=float)
                        b = base.loc[other.index, metric].to_numpy(dtype=float)
                        rows.append({
                            "pattern": pattern, "frames": frames, "policy": policy, "predictor": name,
                            "baseline": base_name, "metric": metric, "mean": float(a.mean()),
                            "mean_baseline": float(b.mean()),
                            "pct_change": float(100 * (a - b).mean() / b.mean()) if b.mean() else float("nan"),
                            "p_value_paired_t": _paired_p(a, b),
                        })
    return pd.DataFrame(rows)


def run_request(n_strings: int = REQUEST_STRINGS) -> pd.DataFrame:
    torch.set_num_threads(1)
    gen = PageReferenceGenerator()
    rows = []
    for pattern in PATTERNS:
        for s in range(n_strings):
            seed = REQUEST_SEED_START + s
            seq = gen.generate(NUM_PAGES, REQUEST_LEN, pattern, seed=seed)["sequence"]
            train, actual = seq[:-1], seq[-1]
            markov = MarkovPagePredictor()
            markov.train(train)
            guess = [c["page"] for c in markov.predict(train[-1], top_k=3)] if train[-1] in markov.transition_table else []
            rows.append({"pattern": pattern, "seed": seed, "predictor": "Markov",
                         "top1": bool(guess) and guess[0] == actual, "top3": actual in guess, "train_ms": float("nan")})
            for name, cfg in TRANSFORMERS.items():
                predictor = TransformerPagePredictor(num_pages=NUM_PAGES, context_len=cfg["context_len"],
                                                     d_model=cfg["d_model"], nhead=cfg["nhead"], seed=seed)
                t0 = time.perf_counter()
                predictor.train(train, epochs=cfg["epochs"], lr=cfg["lr"])
                train_ms = (time.perf_counter() - t0) * 1000.0
                guess = [c["page"] for c in predictor.predict(train[-cfg["context_len"]:], top_k=3)]
                rows.append({"pattern": pattern, "seed": seed, "predictor": name,
                             "top1": guess[0] == actual, "top3": actual in guess, "train_ms": train_ms})
    return pd.DataFrame(rows)


def make_plot(acc: pd.DataFrame, out: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    means = acc.groupby(["pattern", "predictor"])["top1_accuracy"].mean().unstack()[PREDICTORS].loc[PATTERNS]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    x = np.arange(len(PATTERNS))
    width = 0.8 / len(PREDICTORS)
    for i, name in enumerate(PREDICTORS):
        ax.bar(x + (i - (len(PREDICTORS) - 1) / 2) * width, means[name], width, label=name)
    ax.set_xticks(x, [p.capitalize() for p in PATTERNS])
    ax.set_ylabel("Top-1 next-page accuracy")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "memory_transformer_accuracy.png"), dpi=150)
    plt.close(fig)


def run(out: str = RESULTS_DIR, n_strings: int = N_STRINGS, n_request: int = REQUEST_STRINGS) -> dict[str, pd.DataFrame]:
    os.makedirs(out, exist_ok=True)
    acc, rep = run_long(n_strings)
    pair = paired(rep)
    req = run_request(n_request)
    acc.to_csv(os.path.join(out, "memory_transformer_accuracy.csv"), index=False)
    rep.to_csv(os.path.join(out, "memory_transformer_replacement.csv"), index=False)
    pair.to_csv(os.path.join(out, "memory_transformer_paired.csv"), index=False)
    req.to_csv(os.path.join(out, "memory_transformer_request.csv"), index=False)
    make_plot(acc, out)
    return {"accuracy": acc, "replacement": rep, "paired": pair, "request": req}


def main() -> None:
    r = run()
    print("D1. Next-page accuracy (mean over strings):")
    print(r["accuracy"].groupby(["pattern", "predictor"], sort=False)[
        ["top1_accuracy", "top3_accuracy", "train_ms", "predict_ms"]].mean().round(3).to_string())
    print("\nD2. Demand faults and total loads (mean):")
    print(r["replacement"].groupby(["pattern", "frames", "policy", "predictor"], sort=False)[
        ["page_faults", "total_loads"]].mean().round(1).to_string())
    print("\nD3. Request-sized strings, accuracy on the 30th reference:")
    print(r["request"].groupby(["pattern", "predictor"], sort=False)[["top1", "top3"]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
