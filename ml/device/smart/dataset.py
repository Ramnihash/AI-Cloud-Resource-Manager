"""Training-data utilities for the SMART disk failure-risk classifier."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


TARGET_COLUMN = "failure"


def load_prepared_dataset(input_path: str | Path) -> pd.DataFrame:
    """Load a prepared SMART dataset produced by CP2.5."""
    path = Path(input_path)

    if not path.exists():
        raise ValueError(f"Prepared dataset does not exist: {path}")

    if path.suffix.lower() != ".csv":
        raise ValueError(f"Prepared dataset must be a CSV file: {path.name}")

    try:
        dataframe = pd.read_csv(path)
    except Exception as exc:
        raise ValueError(f"Failed to read prepared dataset: {exc}") from exc

    if dataframe.empty:
        raise ValueError("Prepared dataset contains no rows.")

    if TARGET_COLUMN not in dataframe.columns:
        raise ValueError(
            f"Prepared dataset must contain target column '{TARGET_COLUMN}'."
        )

    smart_columns = sorted(
        column for column in dataframe.columns
        if column.startswith("smart_")
    )

    if not smart_columns:
        raise ValueError("Prepared dataset contains no SMART feature columns.")

    return dataframe


def split_features_target(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Split a prepared SMART dataset into features and failure target."""
    if dataframe.empty:
        raise ValueError("Cannot split an empty dataset.")

    if TARGET_COLUMN not in dataframe.columns:
        raise ValueError(
            f"Dataset must contain target column '{TARGET_COLUMN}'."
        )

    feature_columns = sorted(
        column
        for column in dataframe.columns
        if column.startswith("smart_")
    )

    if not feature_columns:
        raise ValueError("Dataset contains no SMART feature columns.")

    features = dataframe[feature_columns].copy()
    target = dataframe[TARGET_COLUMN].copy()

    return features, target