"""CPU Workload Classification Dataset Generation and Feature Extraction for CP3.2.

Architecture & Pipeline:
    CPU Workload Batch (10-30 processes generated via CPUWorkloadGenerator)
        ↓
    CPUBurstPredictor (CP3.1 fitted regression model)
        ↓
    Populates process.predicted_burst_time on every process
        ↓
    Batch Feature Extraction (5 batch-level aggregated features)
        ↓
    Deterministic Workload Labeling (4-class first-match-wins hierarchy)

Locked CP3.2 Classes:
    1. PRIORITY_CRITICAL: priority_variance >= 2.0 AND at least one process has priority == 1
    2. IO_BOUND:          mean_io_wait_ratio >= 0.60
    3. CPU_BOUND:         mean_cpu_intensity >= 0.70 AND mean_burst_time >= 15.0
    4. BALANCED:          fallback for all other workloads

Classifier Features (exactly 5):
    1. mean_predicted_burst_time: arithmetic mean of Process.predicted_burst_time
    2. std_predicted_burst_time:  population standard deviation of Process.predicted_burst_time (ddof=0)
    3. mean_cpu_intensity:        arithmetic mean of process.metadata["cpu_intensity"]
    4. mean_io_wait_ratio:        arithmetic mean of process.metadata["io_wait_ratio"]
    5. priority_variance:         population variance of process.priority (ddof=0)

Statistical Convention:
    All variance and standard deviation calculations consistently use the population statistic (ddof=0),
    matching NumPy standard defaults for observed finite populations and avoiding undefined behavior.
"""

from __future__ import annotations

import csv
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from data.cpu.generator import CPUWorkloadGenerator, WorkloadConfig
from ml.cpu.burst_predictor import CPUBurstPredictor
from modules.cpu.process import Process

# Locked CP3.2 Class Definitions
CLASS_CPU_BOUND = "CPU_BOUND"
CLASS_IO_BOUND = "IO_BOUND"
CLASS_PRIORITY_CRITICAL = "PRIORITY_CRITICAL"
CLASS_BALANCED = "BALANCED"

WORKLOAD_CLASSES = [
    CLASS_CPU_BOUND,
    CLASS_IO_BOUND,
    CLASS_PRIORITY_CRITICAL,
    CLASS_BALANCED,
]

# Locked CP3.2 Batch Classifier Features (Strictly 5 features)
CLASSIFIER_FEATURE_NAMES = [
    "mean_predicted_burst_time",
    "std_predicted_burst_time",
    "mean_cpu_intensity",
    "mean_io_wait_ratio",
    "priority_variance",
]

# Metadata & Target Columns
METADATA_COLUMNS = ["batch_id", "batch_size"]
TARGET_COLUMN = "label"
ALL_DATASET_COLUMNS = METADATA_COLUMNS + CLASSIFIER_FEATURE_NAMES + [TARGET_COLUMN]

# Locked Thresholds
PRIORITY_CRITICAL_VARIANCE_THRESHOLD = 2.0
PRIORITY_CRITICAL_TARGET_PRIORITY = 1
IO_BOUND_IO_WAIT_THRESHOLD = 0.60
CPU_BOUND_CPU_INTENSITY_THRESHOLD = 0.70
CPU_BOUND_BURST_TIME_THRESHOLD = 15.0


def label_batch(processes: list[Process]) -> str:
    """Deterministically assign a ground-truth workload class to an entire process batch.

    Evaluation Rules (Strict first-match-wins order):
        1. PRIORITY_CRITICAL:
           priority_variance >= 2.0 AND at least one process has priority == 1
        2. IO_BOUND:
           mean_io_wait_ratio >= 0.60
        3. CPU_BOUND:
           mean_cpu_intensity >= 0.70 AND mean_burst_time >= 15.0
        4. BALANCED:
           all other workloads

    Args:
        processes: A non-empty list of Process objects.

    Returns:
        One of the four locked class labels: 'PRIORITY_CRITICAL', 'IO_BOUND',
        'CPU_BOUND', or 'BALANCED'.

    Raises:
        ValueError: If processes list is empty.
    """
    if not processes:
        raise ValueError("Cannot label an empty workload batch.")

    priorities = [p.priority for p in processes]
    # Population variance (ddof=0) rounded to 8 digits to safeguard against IEEE-754 precision artifacts
    priority_var = round(float(np.var(priorities, ddof=0)), 8)
    has_p1 = any(p == PRIORITY_CRITICAL_TARGET_PRIORITY for p in priorities)

    io_waits = [
        float(p.metadata.get("io_wait_ratio", 0.0))
        if isinstance(p.metadata, dict)
        else 0.0
        for p in processes
    ]
    mean_io = round(float(np.mean(io_waits)), 8)

    cpu_intensities = [
        float(p.metadata.get("cpu_intensity", 0.0))
        if isinstance(p.metadata, dict)
        else 0.0
        for p in processes
    ]
    mean_cpu = round(float(np.mean(cpu_intensities)), 8)

    # True burst_time is used ONLY for ground-truth labeling, never as a classifier feature
    mean_burst = round(float(np.mean([p.burst_time for p in processes])), 8)

    # 1. PRIORITY_CRITICAL rule
    if priority_var >= PRIORITY_CRITICAL_VARIANCE_THRESHOLD and has_p1:
        return CLASS_PRIORITY_CRITICAL

    # 2. IO_BOUND rule
    if mean_io >= IO_BOUND_IO_WAIT_THRESHOLD:
        return CLASS_IO_BOUND

    # 3. CPU_BOUND rule
    if (
        mean_cpu >= CPU_BOUND_CPU_INTENSITY_THRESHOLD
        and mean_burst >= CPU_BOUND_BURST_TIME_THRESHOLD
    ):
        return CLASS_CPU_BOUND

    # 4. Fallback rule
    return CLASS_BALANCED


def extract_batch_features(processes: list[Process]) -> dict[str, float]:
    """Extract the exactly five locked classifier features from a predicted process batch.

    Requires that CPUBurstPredictor has already populated predicted_burst_time on every process.

    Args:
        processes: A non-empty list of Process objects with predicted_burst_time populated.

    Returns:
        Dictionary mapping the five feature names to their calculated float values.

    Raises:
        ValueError: If processes is empty or any process lacks predicted_burst_time.
    """
    if not processes:
        raise ValueError("Cannot extract features from an empty workload batch.")

    for p in processes:
        if p.predicted_burst_time is None:
            raise ValueError(
                f"Process {p.pid} lacks predicted_burst_time. "
                f"Workloads must be predicted by CPUBurstPredictor before feature extraction."
            )

    pred_bursts = [float(p.predicted_burst_time) for p in processes]
    priorities = [p.priority for p in processes]

    io_waits = [
        float(p.metadata.get("io_wait_ratio", 0.0))
        if isinstance(p.metadata, dict)
        else 0.0
        for p in processes
    ]
    cpu_intensities = [
        float(p.metadata.get("cpu_intensity", 0.0))
        if isinstance(p.metadata, dict)
        else 0.0
        for p in processes
    ]

    return {
        "mean_predicted_burst_time": round(float(np.mean(pred_bursts)), 4),
        "std_predicted_burst_time": round(float(np.std(pred_bursts, ddof=0)), 4),
        "mean_cpu_intensity": round(float(np.mean(cpu_intensities)), 4),
        "mean_io_wait_ratio": round(float(np.mean(io_waits)), 4),
        "priority_variance": round(float(np.var(priorities, ddof=0)), 4),
    }


@dataclass
class WorkloadClassificationSample:
    """Represents a single classified CPU workload batch for CP3.2."""

    batch_id: str
    batch_size: int
    mean_predicted_burst_time: float
    std_predicted_burst_time: float
    mean_cpu_intensity: float
    mean_io_wait_ratio: float
    priority_variance: float
    label: str

    def to_dict(self) -> dict[str, Any]:
        """Convert sample to dictionary."""
        return asdict(self)


def samples_to_dataframe(samples: list[WorkloadClassificationSample]) -> pd.DataFrame:
    """Convert a list of WorkloadClassificationSample instances to a pandas DataFrame.

    Guarantees column layout matching ALL_DATASET_COLUMNS.
    """
    if not samples:
        return pd.DataFrame(columns=ALL_DATASET_COLUMNS)
    df = pd.DataFrame([s.to_dict() for s in samples])
    return df[ALL_DATASET_COLUMNS]


def candidate_config_for_class(target_class: str, rng: random.Random) -> WorkloadConfig:
    """Create a candidate WorkloadConfig biased to efficiently produce a specific class."""
    if target_class == CLASS_PRIORITY_CRITICAL:
        return WorkloadConfig(
            priority_range=(1, rng.randint(6, 10)),
            io_wait_ratio_range=(0.05, round(rng.uniform(0.3, 0.5), 2)),
            cpu_intensity_range=(0.1, round(rng.uniform(0.4, 0.65), 2)),
            instruction_count_range=(5000.0, round(rng.uniform(30000.0, 60000.0), 1)),
            process_complexity_range=(1.0, round(rng.uniform(2.5, 4.0), 1)),
            correlate_burst_to_features=True,
            noise_level=0.1,
        )
    elif target_class == CLASS_IO_BOUND:
        return WorkloadConfig(
            priority_range=(rng.randint(2, 3), rng.randint(4, 5)),
            io_wait_ratio_range=(
                round(rng.uniform(0.60, 0.70), 2),
                round(rng.uniform(0.75, 0.90), 2),
            ),
            cpu_intensity_range=(0.1, round(rng.uniform(0.4, 0.65), 2)),
            instruction_count_range=(5000.0, round(rng.uniform(30000.0, 60000.0), 1)),
            process_complexity_range=(1.0, round(rng.uniform(2.0, 4.0), 1)),
            correlate_burst_to_features=True,
            noise_level=0.1,
        )
    elif target_class == CLASS_CPU_BOUND:
        return WorkloadConfig(
            priority_range=(rng.randint(2, 3), rng.randint(4, 5)),
            io_wait_ratio_range=(0.0, round(rng.uniform(0.2, 0.45), 2)),
            cpu_intensity_range=(
                round(rng.uniform(0.72, 0.80), 2),
                round(rng.uniform(0.85, 1.0), 2),
            ),
            instruction_count_range=(round(rng.uniform(15000.0, 65000.0), 1), 100000.0),
            process_complexity_range=(round(rng.uniform(1.5, 3.5), 1), 5.0),
            correlate_burst_to_features=True,
            noise_level=0.1,
        )
    else:  # CLASS_BALANCED
        return WorkloadConfig(
            priority_range=(rng.randint(2, 3), rng.randint(4, 5)),
            io_wait_ratio_range=(
                round(rng.uniform(0.1, 0.25), 2),
                round(rng.uniform(0.35, 0.55), 2),
            ),
            cpu_intensity_range=(
                round(rng.uniform(0.2, 0.35), 2),
                round(rng.uniform(0.45, 0.65), 2),
            ),
            instruction_count_range=(
                round(rng.uniform(5000.0, 15000.0), 1),
                round(rng.uniform(25000.0, 45000.0), 1),
            ),
            process_complexity_range=(
                round(rng.uniform(1.2, 2.0), 1),
                round(rng.uniform(2.2, 3.5), 1),
            ),
            correlate_burst_to_features=True,
            noise_level=0.1,
        )


def generate_class_workload(
    target_class: str,
    num_processes: int,
    seed: Optional[int] = None,
    max_attempts: int = 50,
) -> tuple[list[Process], str]:
    """Generate one workload batch biased towards `target_class`.

    Uses the same candidate configurations as the training dataset, and retries
    until `label_batch` assigns the requested class or `max_attempts` is reached.

    Returns:
        The generated processes and the label `label_batch` assigned to them,
        which equals `target_class` unless every attempt missed.
    """
    if target_class not in WORKLOAD_CLASSES:
        raise ValueError(f"Unknown workload class: {target_class}")
    if num_processes < 1:
        raise ValueError(f"num_processes must be at least 1, got {num_processes}")

    rng = random.Random(seed)
    processes: list[Process] = []
    label = CLASS_BALANCED
    for _ in range(max(1, max_attempts)):
        cfg = candidate_config_for_class(target_class, rng)
        cfg.seed = rng.randint(0, 10_000_000)
        processes = CPUWorkloadGenerator(config=cfg).generate(num_processes=num_processes)
        label = label_batch(processes)
        if label == target_class:
            break
    return processes, label


class WorkloadClassificationDatasetGenerator:
    """Generator for balanced synthetic CPU workload classification datasets.

    Collects exactly `target_per_class` (default 250) batches of each of the four locked
    classes (total 1,000 batches), ensuring each batch contains between 10 and 30 processes.
    Uses CPUBurstPredictor to populate predicted_burst_time, ensuring authentic CP3.1 → CP3.2 flow.
    """

    def __init__(
        self,
        predictor: Optional[CPUBurstPredictor] = None,
        seed: int = 42,
        min_batch_size: int = 10,
        max_batch_size: int = 30,
        target_per_class: int = 250,
    ) -> None:
        self.seed = seed
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.target_per_class = target_per_class

        if self.min_batch_size < 1 or self.max_batch_size < self.min_batch_size:
            raise ValueError(
                f"Invalid batch size range: [{self.min_batch_size}, {self.max_batch_size}]"
            )
        if self.target_per_class < 1:
            raise ValueError(
                f"target_per_class must be >= 1, got {self.target_per_class}"
            )

        if predictor is not None:
            self.predictor = predictor
            if not self.predictor.is_fitted:
                self.predictor.train_on_synthetic_workload(seed=self.seed)
        else:
            self.predictor = CPUBurstPredictor(random_state=self.seed)
            self.predictor.train_on_synthetic_workload(seed=self.seed)

    def _generate_candidate_config(self, target_class: str, rng: random.Random) -> WorkloadConfig:
        """Create a candidate WorkloadConfig biased to efficiently produce a specific class."""
        return candidate_config_for_class(target_class, rng)

    def generate_dataset(
        self,
        seed: Optional[int] = None,
        target_per_class: Optional[int] = None,
    ) -> list[WorkloadClassificationSample]:
        """Generate candidate batches and collect exactly `target_per_class` for every class.

        Each candidate batch is evaluated by `label_batch` against the deterministic rules.
        Accepted batches are processed through `CPUBurstPredictor` to populate `predicted_burst_time`,
        and their 5 classifier features are recorded.

        Returns:
            A list of WorkloadClassificationSample instances with exactly
            len(WORKLOAD_CLASSES) * target_per_class samples.
        """
        effective_seed = seed if seed is not None else self.seed
        target_count = target_per_class if target_per_class is not None else self.target_per_class
        rng = random.Random(effective_seed)

        collected: dict[str, list[WorkloadClassificationSample]] = {
            c: [] for c in WORKLOAD_CLASSES
        }
        all_samples: list[WorkloadClassificationSample] = []
        attempts = 0

        while any(len(collected[c]) < target_count for c in WORKLOAD_CLASSES):
            attempts += 1
            needed_classes = [c for c in WORKLOAD_CLASSES if len(collected[c]) < target_count]
            target_class = needed_classes[attempts % len(needed_classes)]

            cfg = self._generate_candidate_config(target_class, rng)
            cfg.seed = rng.randint(0, 10_000_000)
            batch_size = rng.randint(self.min_batch_size, self.max_batch_size)

            generator = CPUWorkloadGenerator(config=cfg)
            processes = generator.generate(num_processes=batch_size)

            # Assign label using strict deterministic rules
            label = label_batch(processes)

            if len(collected[label]) < target_count:
                # Step 1 Pipeline: CPUBurstPredictor populates predicted_burst_time
                self.predictor.predict_workload(processes, inplace=True)

                # Extract the 5 batch features
                features = extract_batch_features(processes)

                batch_id = f"batch_{len(all_samples) + 1:04d}"
                sample = WorkloadClassificationSample(
                    batch_id=batch_id,
                    batch_size=len(processes),
                    mean_predicted_burst_time=features["mean_predicted_burst_time"],
                    std_predicted_burst_time=features["std_predicted_burst_time"],
                    mean_cpu_intensity=features["mean_cpu_intensity"],
                    mean_io_wait_ratio=features["mean_io_wait_ratio"],
                    priority_variance=features["priority_variance"],
                    label=label,
                )

                collected[label].append(sample)
                all_samples.append(sample)

        return all_samples

    def to_dataframe(
        self,
        samples: list[WorkloadClassificationSample],
    ) -> pd.DataFrame:
        """Convert samples to DataFrame."""
        return samples_to_dataframe(samples)

    @staticmethod
    def get_feature_matrix_and_labels(
        samples_or_df: list[WorkloadClassificationSample] | pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extract the (N, 5) float feature matrix X and (N,) string label vector y.

        Guarantees that pid, arrival_time, and true burst_time are never included in X.
        """
        if isinstance(samples_or_df, list):
            df = samples_to_dataframe(samples_or_df)
        else:
            df = samples_or_df

        X = df[CLASSIFIER_FEATURE_NAMES].to_numpy(dtype=np.float64)
        y = df[TARGET_COLUMN].to_numpy(dtype=str)
        return X, y

    @staticmethod
    def get_train_test_split(
        samples_or_df: list[WorkloadClassificationSample] | pd.DataFrame,
        test_size: float = 0.20,
        random_state: int = 42,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Produce a stratified train/test split (default 80% train, 20% test).

        Returns:
            (X_train, X_test, y_train, y_test)
        """
        X, y = WorkloadClassificationDatasetGenerator.get_feature_matrix_and_labels(samples_or_df)
        return train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y,
        )

    @staticmethod
    def export_csv(
        samples_or_df: list[WorkloadClassificationSample] | pd.DataFrame,
        filepath: str | Path,
    ) -> Path:
        """Export dataset samples to CSV file."""
        if isinstance(samples_or_df, list):
            df = samples_to_dataframe(samples_or_df)
        else:
            df = samples_or_df

        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        return path
