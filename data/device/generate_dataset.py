"""Dataset generator for simulated device telemetry.

This module combines the existing deterministic normal and abnormal telemetry
generators into a reproducible tabular dataset suitable for later machine
learning and anomaly-detection experiments.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from modules.device.abnormal import (
    AbnormalPattern,
    generate_abnormal_telemetry,
)
from modules.device.fleet import create_default_fleet
from modules.device.telemetry import generate_normal_telemetry


DATASET_COLUMNS = [
    "device_id",
    "timestamp",
    "temperature",
    "error_rate",
    "response_latency",
    "queue_length",
    "io_activity",
    "label",
    "pattern",
]


def generate_device_telemetry_dataset(
    normal_samples: int = 1000,
    abnormal_samples_per_pattern: int = 100,
    timestamp_start: float = 0.0,
    timestamp_step: float = 1.0,
) -> pd.DataFrame:
    """Generate a deterministic device telemetry dataset.

    Args:
        normal_samples: Number of normal telemetry records to generate.
        abnormal_samples_per_pattern: Number of records for each supported
            abnormal pattern.
        timestamp_start: Timestamp assigned to the first generated record.
        timestamp_step: Difference between consecutive timestamps.

    Returns:
        A pandas DataFrame containing normal and abnormal telemetry.

    Raises:
        ValueError: If sample counts are negative or timestamp_step is
            non-positive.
    """
    if normal_samples < 0:
        raise ValueError("normal_samples must be non-negative.")

    if abnormal_samples_per_pattern < 0:
        raise ValueError("abnormal_samples_per_pattern must be non-negative.")

    if timestamp_step <= 0:
        raise ValueError("timestamp_step must be positive.")

    fleet = create_default_fleet()
    records = []

    devices = fleet
    patterns = list(AbnormalPattern)

    for sample_index in range(normal_samples):
        device = devices[sample_index % len(devices)]
        timestamp = timestamp_start + sample_index * timestamp_step

        telemetry = generate_normal_telemetry(
            device=device,
            timestamp=timestamp,
            sample_index=sample_index,
        )

        record = telemetry.to_dict()
        record["label"] = "normal"
        record["pattern"] = "normal"
        records.append(record)

    abnormal_index = normal_samples

    for pattern in patterns:
        for pattern_index in range(abnormal_samples_per_pattern):
            device = devices[pattern_index % len(devices)]
            timestamp = (
                timestamp_start
                + abnormal_index * timestamp_step
            )

            telemetry = generate_abnormal_telemetry(
                device=device,
                timestamp=timestamp,
                pattern=pattern,
                sample_index=pattern_index,
            )

            record = telemetry.to_dict()
            record["label"] = "abnormal"
            record["pattern"] = pattern.value
            records.append(record)

            abnormal_index += 1

    return pd.DataFrame(records, columns=DATASET_COLUMNS)


def save_device_telemetry_dataset(
    dataframe: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    """Save a generated telemetry DataFrame to CSV.

    Args:
        dataframe: Telemetry dataset to save.
        output_path: Destination CSV path.

    Returns:
        Path to the saved CSV file.

    Raises:
        ValueError: If required dataset columns are missing.
    """
    missing_columns = [
        column
        for column in DATASET_COLUMNS
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Dataset is missing required columns: {missing_columns}"
        )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output, index=False)

    return output


if __name__ == "__main__":
    dataset = generate_device_telemetry_dataset()

    output_file = Path("data/device/device_telemetry.csv")
    save_device_telemetry_dataset(dataset, output_file)

    print(f"Generated {len(dataset)} telemetry records.")
    print(f"Saved dataset to: {output_file}")

