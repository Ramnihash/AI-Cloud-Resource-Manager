"""Synthetic CPU workload and process data generator for CPU scheduling and ML regression."""

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

    # ML Feature ranges (CP3.1 Burst-Time Regression Features)
    instruction_count_range: tuple[float, float] = (1000.0, 100000.0)
    cpu_intensity_range: tuple[float, float] = (0.1, 1.0)
    io_wait_ratio_range: tuple[float, float] = (0.0, 0.8)
    process_complexity_range: tuple[float, float] = (1.0, 5.0)

    # Optional mode to derive burst_time directly from ML features via synthetic relationship
    correlate_burst_to_features: bool = False
    noise_level: float = 0.2
    k_scale: float = 2500.0

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

        # Validate ML Feature Ranges
        min_instr, max_instr = self.instruction_count_range
        if min_instr <= 0 or max_instr < min_instr:
            raise ValueError(
                f"Invalid instruction_count_range: ({min_instr}, {max_instr}). Must be > 0 with min <= max."
            )

        min_cpu, max_cpu = self.cpu_intensity_range
        if min_cpu <= 0 or max_cpu > 1.0 or max_cpu < min_cpu:
            raise ValueError(
                f"Invalid cpu_intensity_range: ({min_cpu}, {max_cpu}). Must be in (0, 1] with min <= max."
            )

        min_io, max_io = self.io_wait_ratio_range
        if min_io < 0 or max_io >= 1.0 or max_io < min_io:
            raise ValueError(
                f"Invalid io_wait_ratio_range: ({min_io}, {max_io}). Must be in [0, 1) with min <= max."
            )

        min_comp, max_comp = self.process_complexity_range
        if min_comp <= 0 or max_comp < min_comp:
            raise ValueError(
                f"Invalid process_complexity_range: ({min_comp}, {max_comp}). Must be > 0 with min <= max."
            )

        if self.noise_level < 0:
            raise ValueError(f"noise_level cannot be negative, got {self.noise_level}")


class CPUWorkloadGenerator:
    """Generator for synthetic CPU process workloads.

    Produces Process-compatible workloads ready for CPU scheduling simulation
    and future burst-time regression datasets.
    """

    ML_FEATURE_NAMES = [
        "instruction_count",
        "cpu_intensity",
        "io_wait_ratio",
        "process_complexity",
    ]

    CORE_COLUMNS = ["pid", "arrival_time", "burst_time", "priority"]
    ALL_COLUMNS = CORE_COLUMNS + ML_FEATURE_NAMES

    def __init__(self, config: Optional[WorkloadConfig] = None, seed: Optional[int] = None) -> None:
        self.config = config or WorkloadConfig()
        if seed is not None:
            self.config.seed = seed
        self.config.validate()

    @staticmethod
    def calculate_synthetic_burst_time(
        instruction_count: float,
        cpu_intensity: float,
        io_wait_ratio: float,
        process_complexity: float,
        k_scale: float = 2500.0,
        noise: float = 0.0,
        min_burst: float = 0.5,
        round_digits: int = 2,
    ) -> float:
        """Calculates synthetic CPU burst time from process workload features.

        Physical formulation (causal ground-truth relationship):
            effective_instructions = instruction_count * process_complexity * cpu_intensity
            io_scaling = 1.0 + io_wait_ratio
            raw_burst = effective_instructions / (k_scale * io_scaling)
            burst_time = max(min_burst, round(raw_burst + noise, round_digits))

        This relationship is deterministic, avoids target leakage, and maps
        features into a realistic CPU execution duration.
        """
        effective_instructions = instruction_count * process_complexity * cpu_intensity
        io_scaling = 1.0 + max(0.0, io_wait_ratio)
        raw_burst = effective_instructions / (k_scale * io_scaling)
        return max(min_burst, round(raw_burst + noise, round_digits))

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
            A list of Process instances populated with core scheduler fields
            (pid, arrival_time, burst_time, priority) and ML features in metadata.
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

        min_instr, max_instr = cfg.instruction_count_range
        min_cpu, max_cpu = cfg.cpu_intensity_range
        min_io, max_io = cfg.io_wait_ratio_range
        min_comp, max_comp = cfg.process_complexity_range

        current_arrival = 0.0
        for i in range(1, n + 1):
            pid = f"{cfg.pid_prefix}{i}"

            # 1. Arrival time generation
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

            # 2. Priority generation (integer range)
            priority = rng.randint(min_pri, max_pri)

            # 3. ML features generation
            instruction_count = round(rng.uniform(min_instr, max_instr), 1)
            cpu_intensity = round(rng.uniform(min_cpu, max_cpu), 3)
            io_wait_ratio = round(rng.uniform(min_io, max_io), 3)
            process_complexity = round(rng.uniform(min_comp, max_comp), 2)

            # 4. Burst time generation
            if cfg.correlate_burst_to_features:
                noise = rng.gauss(0.0, cfg.noise_level) if cfg.noise_level > 0 else 0.0
                burst_time = self.calculate_synthetic_burst_time(
                    instruction_count=instruction_count,
                    cpu_intensity=cpu_intensity,
                    io_wait_ratio=io_wait_ratio,
                    process_complexity=process_complexity,
                    k_scale=cfg.k_scale,
                    noise=noise,
                    min_burst=min_burst,
                    round_digits=cfg.round_digits,
                )
            elif cfg.burst_distribution == "uniform":
                burst_time = round(rng.uniform(min_burst, max_burst), cfg.round_digits)
            elif cfg.burst_distribution == "exponential":
                mean_burst = (min_burst + max_burst) / 2.0
                sampled = rng.expovariate(1.0 / mean_burst)
                burst_time = max(min_burst, round(sampled, cfg.round_digits))
            else:
                burst_time = min_burst

            proc = Process(
                pid=pid,
                arrival_time=arrival_time,
                burst_time=burst_time,
                priority=priority,
            )
            # Attach ML workload features to process metadata
            proc.metadata = {
                "instruction_count": instruction_count,
                "cpu_intensity": cpu_intensity,
                "io_wait_ratio": io_wait_ratio,
                "process_complexity": process_complexity,
            }
            processes.append(proc)

        if cfg.sort_by_arrival:
            processes.sort(key=lambda p: (p.arrival_time, str(p.pid)))

        return processes

    def to_dataframe(self, processes: list[Process]) -> pd.DataFrame:
        """Convert a list of processes into a pandas DataFrame including ML features."""
        rows = []
        for p in processes:
            row = {
                "pid": p.pid,
                "arrival_time": p.arrival_time,
                "burst_time": p.burst_time,
                "priority": p.priority,
                "instruction_count": p.metadata.get("instruction_count", 0.0),
                "cpu_intensity": p.metadata.get("cpu_intensity", 0.0),
                "io_wait_ratio": p.metadata.get("io_wait_ratio", 0.0),
                "process_complexity": p.metadata.get("process_complexity", 0.0),
            }
            rows.append(row)
        return pd.DataFrame(rows, columns=self.ALL_COLUMNS)

    @staticmethod
    def export_to_csv(processes: list[Process], filepath: str | Path) -> Path:
        """Export a list of Process objects to a CSV file with ML feature columns."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CPUWorkloadGenerator.ALL_COLUMNS)
            writer.writeheader()
            for p in processes:
                writer.writerow(
                    {
                        "pid": p.pid,
                        "arrival_time": p.arrival_time,
                        "burst_time": p.burst_time,
                        "priority": p.priority,
                        "instruction_count": p.metadata.get("instruction_count", 0.0),
                        "cpu_intensity": p.metadata.get("cpu_intensity", 0.0),
                        "io_wait_ratio": p.metadata.get("io_wait_ratio", 0.0),
                        "process_complexity": p.metadata.get("process_complexity", 0.0),
                    }
                )
        return path

    @staticmethod
    def load_from_csv(filepath: str | Path) -> list[Process]:
        """Load processes from a CSV file into Process instances, restoring ML features."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Workload CSV file not found: {path}")

        processes: list[Process] = []
        with open(path, mode="r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            required_core = {"pid", "arrival_time", "burst_time", "priority"}
            fieldnames = set(reader.fieldnames or [])
            if not required_core.issubset(fieldnames):
                raise ValueError(
                    f"CSV file missing required columns. Required: {required_core}, found: {reader.fieldnames}"
                )

            for row in reader:
                proc = Process(
                    pid=row["pid"],
                    arrival_time=float(row["arrival_time"]),
                    burst_time=float(row["burst_time"]),
                    priority=int(row["priority"]),
                )
                # Load ML features if available in CSV
                for feat in CPUWorkloadGenerator.ML_FEATURE_NAMES:
                    if feat in row and row[feat] != "":
                        proc.metadata[feat] = float(row[feat])
                processes.append(proc)

        return processes
