"""Download one Backblaze Drive Stats quarter and build the SMART training set.

Usage (from the repository root, on a machine with internet access and about
2 GB of free disk):

    python -m scripts.build_smart_dataset                  # default quarter
    python -m scripts.build_smart_dataset --quarter Q2_2025
    python -m scripts.build_smart_dataset --zip D:\\Downloads\\data_Q1_2025.zip

The zip is saved under ``data/device/smart/raw/`` (git-ignored) and read in
place, so the daily CSVs are never extracted. The output is a compact labelled
sample, ``data/device/smart/backblaze_failure_windows.csv.gz`` plus a
``.meta.json`` summary, small enough to commit so the backend and the
evaluation can be rerun without the raw quarter.

Data: Backblaze Drive Stats, https://www.backblaze.com/cloud-storage/resources/hard-drive-test-data
(free to use with attribution to Backblaze; the raw files are not redistributed here).
"""

from __future__ import annotations

import argparse
import os
import sys
import urllib.request
from pathlib import Path

from data.device.smart import build_failure_windows as builder

BASE_URL = "https://f001.backblazeb2.com/file/Backblaze-Hard-Drive-Data/data_{quarter}.zip"
DEFAULT_QUARTER = "Q1_2025"
RAW_DIR = Path("data") / "device" / "smart" / "raw"


def download(quarter: str, dest_dir: Path = RAW_DIR) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    target = dest_dir / f"data_{quarter}.zip"
    if target.exists() and target.stat().st_size > 0:
        print(f"Using the already downloaded {target}")
        return target

    url = BASE_URL.format(quarter=quarter)
    partial = target.with_suffix(".zip.part")
    print(f"Downloading {url}\n(about 1 GB; this can take several minutes)")
    with urllib.request.urlopen(url, timeout=60) as response, open(partial, "wb") as out:
        total = int(response.headers.get("Content-Length") or 0)
        done = 0
        while True:
            chunk = response.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)
            done += len(chunk)
            if total:
                sys.stdout.write(f"\r  {done / 1e6:,.0f} / {total / 1e6:,.0f} MB")
                sys.stdout.flush()
    print()
    os.replace(partial, target)
    return target


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Download a Backblaze quarter and build the SMART dataset.")
    parser.add_argument("--quarter", default=DEFAULT_QUARTER, help="for example Q1_2025")
    parser.add_argument("--zip", help="use an already downloaded quarterly zip instead of downloading")
    parser.add_argument("--output", default=builder.DEFAULT_OUTPUT)
    parser.add_argument("--horizon-days", type=int, default=builder.DEFAULT_HORIZON_DAYS)
    parser.add_argument("--negatives-per-day", type=int, default=builder.DEFAULT_NEGATIVES_PER_DAY)
    args = parser.parse_args(argv)

    source = Path(args.zip) if args.zip else download(args.quarter)
    builder.main([
        "--source", str(source),
        "--output", args.output,
        "--horizon-days", str(args.horizon_days),
        "--negatives-per-day", str(args.negatives_per_day),
    ])


if __name__ == "__main__":
    main()
