"""CP3.4: fair next-page evaluation on independent memory traces.

Run from repository root:
    python -m scripts.evaluate_memory_predictors

Requires torch and the CP2.2 dataset at data/memory/page_sequences_v1.
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ml.memory.markov_predictor import MarkovPagePredictor
from ml.memory.transformer_predictor import TransformerPagePredictor


def load_split(path: Path, window_size: int, num_pages: int):
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            context = ast.literal_eval(row["input_pages"])
            target = int(row["target_page"])
            if not isinstance(context, list) or len(context) != window_size:
                raise ValueError(f"Invalid window in {path}: {row['trace_id']}")
            if any(type(x) is not int or not 0 <= x < num_pages for x in context):
                raise ValueError("Invalid page ID in context")
            if not 0 <= target < num_pages:
                raise ValueError("Invalid target page")
            rows.append((row["trace_id"], row["pattern"], context, target))
    if not rows:
        raise ValueError(f"Empty split: {path}")
    return rows


def score_predictions(rows, predict_fn, batch_size=512):
    counts = defaultdict(lambda: [0, 0, 0])
    start = time.perf_counter()
    for start_idx in range(0, len(rows), batch_size):
        chunk = rows[start_idx:start_idx + batch_size]
        guesses = predict_fn(chunk)
        for (_, pattern, _, target), preds in zip(chunk, guesses):
            for key in ("overall", pattern):
                c = counts[key]
                c[0] += 1
                c[1] += int(bool(preds) and preds[0] == target)
                c[2] += int(target in preds[:3])
    elapsed = time.perf_counter() - start
    result = {}
    for pattern, (n, top1, top3) in sorted(counts.items()):
        result[pattern] = {"samples": n, "top1_accuracy": top1 / n,
                           "top3_accuracy": top3 / n}
    return result, elapsed


def markov_predictor(train_rows):
    # Reconstruct transitions without introducing artificial trace boundaries.
    # Each unique position is represented once by the (last context page, target) pair.
    transitions = Counter((context[-1], target) for _, _, context, target in train_rows)
    model = MarkovPagePredictor()
    # The existing API trains on one sequence; populate the same learned table
    # directly from pooled transition counts, without altering existing code.
    totals = Counter()
    for (source, _), count in transitions.items():
        totals[source] += count
    model.transition_table = defaultdict(dict)
    for (source, target), count in transitions.items():
        model.transition_table[source][target] = count / totals[source]
    model.transition_table = dict(model.transition_table)
    model._training_length = len(train_rows) + 1
    model._trained = True
    return model


def transformer_predictions(model, rows, batch_size=512):
    model.eval()
    guesses = []
    with torch.inference_mode():
        for i in range(0, len(rows), batch_size):
            contexts = torch.tensor([r[2] for r in rows[i:i + batch_size]], dtype=torch.long)
            logits = model(contexts)
            guesses.extend(torch.topk(logits, k=min(3, logits.shape[-1]), dim=-1).indices.tolist())
    return guesses


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("data/memory/page_sequences_v1"))
    parser.add_argument("--output", type=Path, default=Path("data/memory/evaluation_cp34"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.001)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0:
        parser.error("epochs, batch-size, and lr must be positive")

    metadata = json.loads((args.dataset / "metadata.json").read_text(encoding="utf-8"))
    num_pages = int(metadata["num_pages"])
    window_size = int(metadata["window_size"])
    splits = {name: load_split(args.dataset / f"{name}.csv", window_size, num_pages)
              for name in ("train", "validation", "test")}
    ids = {name: {r[0] for r in rows} for name, rows in splits.items()}
    if any(ids[a] & ids[b] for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))):
        raise ValueError("Trace leakage between dataset splits")

    seed = int(metadata.get("base_seed", 2026))
    random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    args.output.mkdir(parents=True, exist_ok=True)

    markov = markov_predictor(splits["train"])
    def markov_batch(rows):
        return [[p["page"] for p in markov.predict(context[-1], top_k=3)]
                for _, _, context, _ in rows]

    predictor = TransformerPagePredictor(num_pages=num_pages, context_len=window_size,
                                          d_model=32, nhead=2, d_ff=64, seed=seed)
    predictor._init_model()
    network = predictor.model
    optimizer = torch.optim.Adam(network.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    train_rows = splits["train"]
    X = torch.tensor([r[2] for r in train_rows], dtype=torch.long)
    y = torch.tensor([r[3] for r in train_rows], dtype=torch.long)
    loader = DataLoader(TensorDataset(X, y), batch_size=args.batch_size, shuffle=True)

    best_accuracy = -1.0
    history = []
    weights_path = args.output / "transformer_best.pt"
    for epoch in range(1, args.epochs + 1):
        network.train()
        total_loss = 0.0
        for inputs, labels in loader:
            optimizer.zero_grad()
            loss = criterion(network(inputs), labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(labels)
        val_metrics, _ = score_predictions(
            splits["validation"],
            lambda rows: transformer_predictions(network, rows, args.batch_size),
            args.batch_size,
        )
        val_acc = val_metrics["overall"]["top1_accuracy"]
        history.append({"epoch": epoch, "train_loss": total_loss / len(train_rows),
                        "validation_top1": val_acc})
        print(f"Epoch {epoch}/{args.epochs}: loss={history[-1]['train_loss']:.4f}, val top1={val_acc:.4f}", flush=True)
        if val_acc > best_accuracy:
            best_accuracy = val_acc
            torch.save(network.state_dict(), weights_path)

    network.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
    markov_test, markov_seconds = score_predictions(splits["test"], markov_batch)
    transformer_test, transformer_seconds = score_predictions(
        splits["test"], lambda rows: transformer_predictions(network, rows, args.batch_size), args.batch_size)

    report = {
        "dataset": str(args.dataset), "num_pages": num_pages, "window_size": window_size,
        "split_rows": {name: len(rows) for name, rows in splits.items()},
        "seed": seed, "epochs": args.epochs, "batch_size": args.batch_size,
        "learning_rate": args.lr, "training_history": history,
        "markov": {"metrics": markov_test, "inference_seconds": markov_seconds},
        "transformer": {"metrics": transformer_test, "inference_seconds": transformer_seconds,
                        "weights": str(weights_path)},
        "notes": ["Both models evaluated on identical held-out test windows.",
                  "Markov uses first-order transitions only; Transformer uses 8-page context.",
                  "Sequential traces repeat by design; inspect per-pattern scores.",
                  "Inference timing includes Python/CSV-row batch overhead and is not a hardware-neutral benchmark."],
    }
    report_path = args.output / "comparison.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    csv_path = args.output / "comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model", "pattern", "samples", "top1_accuracy", "top3_accuracy"])
        for model_name, result in (("markov", markov_test), ("transformer", transformer_test)):
            for pattern, metrics in result.items():
                writer.writerow([model_name, pattern, metrics["samples"],
                                 metrics["top1_accuracy"], metrics["top3_accuracy"]])
    for model_name, result in (("Markov", markov_test), ("Transformer", transformer_test)):
        print(f"{model_name}: top1={result['overall']['top1_accuracy']:.2%}, "
              f"top3={result['overall']['top3_accuracy']:.2%}")
    print("Saved:", report_path, csv_path, weights_path, sep="\n  ")


if __name__ == "__main__":
    main()
