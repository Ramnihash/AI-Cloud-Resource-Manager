"""Build a labelled SMART dataset from raw Backblaze daily CSVs (CP2.5 -> CP3.8).

A raw Backblaze quarter is about 90 daily files with one row per drive per day,
where ``failure`` is 1 only on the day a drive failed. Predicting that single
day is not useful, so this builder relabels the data as a forecasting task:

    target ``failure`` = 1  when the drive fails within ``horizon_days`` (default 30)
    target ``failure`` = 0  for a sampled day of a drive that did not fail in the quarter

Pass 1 reads only ``serial_number`` and ``failure`` from every file to find each
failed drive and its failure date, and counts drive-days so the real prevalence
of positive days is known. Pass 2 keeps every row of a failed drive inside its
horizon and a seeded random sample of rows from drives that never failed. Rows
of failed drives from before their horizon are dropped, because they are neither
clearly healthy nor clearly failing.

The output keeps ``serial_number`` so evaluation can split by drive (rows of one
drive are strongly correlated and must not appear in both train and test).

Usage (from the repository root):

    python -m data.device.smart.build_failure_windows --source data_Q1_2025.zip

``--source`` is a Backblaze quarterly zip or a folder of its daily CSVs. Nothing
is downloaded here; see ``scripts/build_smart_dataset.py`` for that.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import zipfile
from collections.abc import Iterator
from datetime import date, datetime
from pathlib import Path
from typing import IO, Any

import numpy as np
import pandas as pd

# SMART attributes commonly linked to drive failure (Backblaze's five: 5, 187, 188,
# 197, 198) plus wear and environment attributes. Raw values only; vendors do not
# report every attribute, so missing values are expected and handled at training.
SMART_ATTRIBUTES = (1, 3, 4, 5, 7, 9, 10, 12, 187, 188, 192, 193, 194, 197, 198, 199, 240, 241, 242)
FEATURE_COLUMNS = tuple(f"smart_{i}_raw" for i in SMART_ATTRIBUTES)
META_COLUMNS = ("date", "serial_number", "model")
OUTPUT_COLUMNS = (*META_COLUMNS, "days_to_failure", *FEATURE_COLUMNS, "failure")

DEFAULT_HORIZON_DAYS = 30
DEFAULT_NEGATIVES_PER_DAY = 400
DEFAULT_SEED = 42
DEFAULT_OUTPUT = os.path.join("data", "device", "smart", "backblaze_failure_windows.csv.gz")


def iter_daily_csvs(source: str | Path) -> Iterator[tuple[str, IO[bytes]]]:
    """Yield ``(file name, open binary file)`` for each daily CSV, in date order."""
    path = Path(source)
    if not path.exists():
        raise ValueError(f"Source does not exist: {path}")

    if path.is_dir():
        for csv_path in sorted(path.rglob("*.csv")):
            if csv_path.name.startswith("._"):
                continue
            with open(csv_path, "rb") as handle:
                yield csv_path.name, handle
        return

    if path.suffix.lower() != ".zip":
        raise ValueError(f"Source must be a .zip file or a folder of CSVs, got: {path.name}")

    with zipfile.ZipFile(path) as archive:
        names = sorted(
            n for n in archive.namelist()
            if n.lower().endswith(".csv") and "__MACOSX" not in n and not os.path.basename(n).startswith("._")
        )
        for name in names:
            with archive.open(name) as handle:
                yield os.path.basename(name), io.BufferedReader(handle)


def _read_columns(handle: IO[bytes], wanted: set[str]) -> pd.DataFrame:
    header = pd.read_csv(handle, nrows=0).columns
    handle.seek(0)
    present = [c for c in header if c in wanted]
    return pd.read_csv(handle, usecols=present, low_memory=False)


def _seekable(handle: IO[bytes]) -> IO[bytes]:
    if handle.seekable():
        return handle
    return io.BytesIO(handle.read())


def find_failures(source: str | Path) -> tuple[dict[str, date], dict[str, Any]]:
    """Pass 1: failure date per failed drive, plus drive-day counts."""
    failures: dict[str, date] = {}
    days: list[date] = []
    rows_per_day: dict[date, int] = {}
    for name, raw in iter_daily_csvs(source):
        handle = _seekable(raw)
        frame = _read_columns(handle, {"date", "serial_number", "failure"})
        if frame.empty or "failure" not in frame or "serial_number" not in frame:
            continue
        day = _day_of(frame, name)
        days.append(day)
        rows_per_day[day] = len(frame)
        failed = frame.loc[pd.to_numeric(frame["failure"], errors="coerce") == 1, "serial_number"]
        for serial in failed.astype(str):
            failures.setdefault(serial, day)
    if not days:
        raise ValueError("No daily CSV files with 'serial_number' and 'failure' columns were found.")
    stats = {
        "first_day": min(days).isoformat(),
        "last_day": max(days).isoformat(),
        "files": len(days),
        "drive_days": int(sum(rows_per_day.values())),
        "failed_drives": len(failures),
    }
    return failures, stats


def _day_of(frame: pd.DataFrame, name: str) -> date:
    if "date" in frame and len(frame):
        try:
            return datetime.strptime(str(frame["date"].iloc[0])[:10], "%Y-%m-%d").date()
        except ValueError:
            pass
    return datetime.strptime(os.path.basename(name)[:10], "%Y-%m-%d").date()


def build_failure_windows(
    source: str | Path,
    horizon_days: int = DEFAULT_HORIZON_DAYS,
    negatives_per_day: int = DEFAULT_NEGATIVES_PER_DAY,
    seed: int = DEFAULT_SEED,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build the labelled dataset and a summary of how it was built."""
    if horizon_days < 1:
        raise ValueError("horizon_days must be at least 1")
    if negatives_per_day < 1:
        raise ValueError("negatives_per_day must be at least 1")

    failures, stats = find_failures(source)
    rng = np.random.default_rng(seed)
    wanted = {*META_COLUMNS, "failure", *FEATURE_COLUMNS}
    parts: list[pd.DataFrame] = []
    positive_drive_days = 0

    for name, raw in iter_daily_csvs(source):
        handle = _seekable(raw)
        frame = _read_columns(handle, wanted)
        if frame.empty or "serial_number" not in frame:
            continue
        day = _day_of(frame, name)
        frame["serial_number"] = frame["serial_number"].astype(str)
        fail_day = frame["serial_number"].map(failures)
        has_failure = fail_day.notna()
        days_left = pd.Series(np.nan, index=frame.index)
        days_left[has_failure] = [(d - day).days for d in fail_day[has_failure]]

        positive = has_failure & (days_left >= 0) & (days_left < horizon_days)
        positive_drive_days += int(positive.sum())
        healthy_idx = frame.index[~has_failure]
        take = min(negatives_per_day, len(healthy_idx))
        negative_idx = rng.choice(healthy_idx, size=take, replace=False) if take else []

        chosen = frame.loc[positive].copy()
        chosen["failure"] = 1
        chosen["days_to_failure"] = days_left[positive].astype(int)
        negatives = frame.loc[negative_idx].copy()
        negatives["failure"] = 0
        negatives["days_to_failure"] = -1
        part = pd.concat([chosen, negatives])
        part["date"] = day.isoformat()
        for col in (*FEATURE_COLUMNS, "model"):
            if col not in part:
                part[col] = np.nan
        parts.append(part[list(OUTPUT_COLUMNS)])

    dataset = pd.concat(parts, ignore_index=True)
    for col in FEATURE_COLUMNS:
        dataset[col] = pd.to_numeric(dataset[col], errors="coerce")
    dataset["failure"] = dataset["failure"].astype(int)

    stats.update({
        "horizon_days": horizon_days,
        "negatives_per_day": negatives_per_day,
        "seed": seed,
        "positive_drive_days": positive_drive_days,
        "prevalence": positive_drive_days / stats["drive_days"] if stats["drive_days"] else 0.0,
        "rows": int(len(dataset)),
        "positive_rows": int(dataset["failure"].sum()),
        "negative_rows": int((dataset["failure"] == 0).sum()),
        "drives": int(dataset["serial_number"].nunique()),
        "failed_drives_in_dataset": int(dataset.loc[dataset["failure"] == 1, "serial_number"].nunique()),
        "features": list(FEATURE_COLUMNS),
        "feature_missing_rate": {c: round(float(dataset[c].isna().mean()), 4) for c in FEATURE_COLUMNS},
    })
    return dataset, stats


def save(dataset: pd.DataFrame, stats: dict[str, Any], output: str | Path = DEFAULT_OUTPUT) -> tuple[Path, Path]:
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(out, index=False)
    meta = meta_path_for(out)
    meta.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    return out, meta


def meta_path_for(dataset_path: str | Path) -> Path:
    p = Path(dataset_path)
    stem = p.name.split(".csv")[0]
    return p.with_name(stem + ".meta.json")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", required=True, help="Backblaze quarterly zip or folder of daily CSVs")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--horizon-days", type=int, default=DEFAULT_HORIZON_DAYS)
    parser.add_argument("--negatives-per-day", type=int, default=DEFAULT_NEGATIVES_PER_DAY)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args(argv)

    print(f"Reading {args.source} (two passes; a full quarter takes a few minutes)...")
    dataset, stats = build_failure_windows(args.source, args.horizon_days, args.negatives_per_day, args.seed)
    out, meta = save(dataset, stats, args.output)
    print(f"Wrote {out} ({stats['rows']} rows: {stats['positive_rows']} failing within "
          f"{stats['horizon_days']} days from {stats['failed_drives_in_dataset']} drives, "
          f"{stats['negative_rows']} healthy) and {meta}")


if __name__ == "__main__":
    main()
