"""Synthetic CPU workload and process data generator for CPU scheduling."""

from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from modules.cpu.process import Process


@dataclass
class WorkloadConfig:
    """Configuration settings for generating synthetic CPU process workloads."""

    num_processes: int = 10
    arrival_range: tuple[float, float] = (0.0, 10.0)
    burst_range: tuple[float, float] = (1.0, 20.0)
    priority_range: tuple[int, int] = (1, 5)
    arrival_distribution: str = "uniform"  # 'uniform', 'exponential', 'zero'
    burst_distribution: str = "uniform"  # 'uniform', 'exponential'
    pid_prefix: str = "P"
    seed: Optional[int] = None
    sort_by_arrival: bool = True
    round_digits: int = 2

    def validate(self) -> None:
        """Validate configuration bounds and distribution options."""
        if self.num_processes < 1:
            raise ValueError(f"num_processes must be at least 1, got {self.num_processes}")

        min_arr, max_arr = self.arrival_range
        if min_arr < 0 or max_arr < min_arr:
            raise ValueError(
                f"Invalid arrival_range: ({min_arr}, {max_arr}). Must be non-negative with min <= max."
            )

        min_burst, max_burst = self.burst_range
        if min_burst <= 0 or max_burst < min_burst:
            raise ValueError(
                f"Invalid burst_range: ({min_burst}, {max_burst}). Burst times must be positive (> 0) with min <= max."
            )

        min_pri, max_pri = self.priority_range
        if min_pri < 0 or max_pri < min_pri:
            raise ValueError(
                f"Invalid priority_range: ({min_pri}, {max_pri}). Priority must be non-negative with min <= max."
            )

        valid_arrival_dists = {"uniform", "exponential", "zero"}
        if self.arrival_distribution not in valid_arrival_dists:
            raise ValueError(
                f"Unknown arrival_distribution: '{self.arrival_distribution}'. Expected one of {valid_arrival_dists}"
            )

        valid_burst_dists = {"uniform", "exponential"}
        if self.burst_distribution not in valid_burst_dists:
            raise ValueError(
                f"Unknown burst_distribution: '{self.burst_distribution}'. Expected one of {valid_burst_dists}"
            )


class CPUWorkloadGenerator:
    """Generator for synthetic CPU process workloads.

    Produces Process-compatible workloads ready for CPU scheduling simulation
    and future burst-time regression datasets.
    """

    def __init__(self, config: Optional[WorkloadConfig] = None, seed: Optional[int] = None) -> None:
        self.config = config or WorkloadConfig()
        if seed is not None:
            self.config.seed = seed
        self.config.validate()

    def generate(
        self,
        num_processes: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> list[Process]:
        """Generate a deterministic or pseudo-random list of Process objects.

        Args:
            num_processes: Override the configured number of processes.
            seed: Override the configured random seed.

        Returns:
            A list of Process instances populated with pid, arrival_time,
            burst_time, and priority.
        """
        cfg = self.config
        n = num_processes if num_processes is not None else cfg.num_processes
        if n < 1:
            raise ValueError(f"num_processes must be at least 1, got {n}")

        effective_seed = seed if seed is not None else cfg.seed
        rng = random.Random(effective_seed)

        processes: list[Process] = []
        min_arr, max_arr = cfg.arrival_range
        min_burst, max_burst = cfg.burst_range
        min_pri, max_pri = cfg.priority_range

        # Generate arrival times
        current_arrival = 0.0
        for i in range(1, n + 1):
            pid = f"{cfg.pid_prefix}{i}"

            # Arrival time generation
            if cfg.arrival_distribution == "zero":
                arrival_time = 0.0
            elif cfg.arrival_distribution == "uniform":
                arrival_time = round(rng.uniform(min_arr, max_arr), cfg.round_digits)
            elif cfg.arrival_distribution == "exponential":
                mean_inter = (max_arr - min_arr) / float(n) if n > 0 else 1.0
                if mean_inter <= 0:
                    mean_inter = 1.0
                if i == 1:
                    arrival_time = min_arr
                    current_arrival = min_arr
                else:
                    delta = rng.expovariate(1.0 / mean_inter)
                    current_arrival += delta
                    arrival_time = round(current_arrival, cfg.round_digits)
            else:
                arrival_time = 0.0

            # Burst time generation
            if cfg.burst_distribution == "uniform":
                burst_time = round(rng.uniform(min_burst, max_burst), cfg.round_digits)
            elif cfg.burst_distribution == "exponential":
                mean_burst = (min_burst + max_burst) / 2.0
                sampled = rng.expovariate(1.0 / mean_burst)
                # Ensure burst time stays strictly positive
                burst_time = max(min_burst, round(sampled, cfg.round_digits))
            else:
                burst_time = min_burst

            # Priority generation (integer range)
            priority = rng.randint(min_pri, max_pri)

            processes.append(
                Process(
                    pid=pid,
                    arrival_time=arrival_time,
                    burst_time=burst_time,
                    priority=priority,
                )
            )

        if cfg.sort_by_arrival:
            processes.sort(key=lambda p: (p.arrival_time, str(p.pid)))

        return processes

    def to_dataframe(self, processes: list[Process]) -> pd.DataFrame:
        """Convert a list of processes into a pandas DataFrame."""
        return pd.DataFrame(
            [
                {
                    "pid": p.pid,
                    "arrival_time": p.arrival_time,
                    "burst_time": p.burst_time,
                    "priority": p.priority,
                }
                for p in processes
            ]
        )

    @staticmethod
    def export_to_csv(processes: list[Process], filepath: str | Path) -> Path:
        """Export a list of Process objects to a CSV file.

        Args:
            processes: Workload to export.
            filepath: Destination CSV path.

        Returns:
            Resolved Path of the written file.
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["pid", "arrival_time", "burst_time", "priority"])
            writer.writeheader()
            for p in processes:
                writer.writerow(
                    {
                        "pid": p.pid,
                        "arrival_time": p.arrival_time,
                        "burst_time": p.burst_time,
                        "priority": p.priority,
                    }
                )
        return path

    @staticmethod
    def load_from_csv(filepath: str | Path) -> list[Process]:
        """Load processes from a CSV file into Process instances.

        Args:
            filepath: Source CSV path.

        Returns:
            List of validated Process objects.
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Workload CSV file not found: {path}")

        processes: list[Process] = []
        with open(path, mode="r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            required = {"pid", "arrival_time", "burst_time", "priority"}
            if not required.issubset(set(reader.fieldnames or [])):
                raise ValueError(
                    f"CSV file missing required columns. Required: {required}, found: {reader.fieldnames}"
                )

            for row in reader:
                processes.append(
                    Process(
                        pid=row["pid"],
                        arrival_time=float(row["arrival_time"]),
                        burst_time=float(row["burst_time"]),
                        priority=int(row["priority"]),
                    )
                )
        return processes
