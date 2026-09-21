"""SMART and Backblaze hard-drive telemetry data preparation pipeline (CP2.5).

This module prepares real-world hard-drive telemetry containing S.M.A.R.T.
attributes for downstream disk failure-risk classification (CP3.8).

Key design properties:
- Strict separation from CP2.4 synthetic device telemetry.
- Dynamic discovery of SMART features (`smart_*`).
- Deterministic numeric coercion and median imputation on input data.
- Strict validation of binary target labels and tabular structures.
"""

from __future__ import annotations

from pathlib import Path
from typing import Set

import pandas as pd


# Explicit metadata columns to exclude from feature representation
EXCLUDED_METADATA_COLUMNS: Set[str] = {
    "serial_number",
    "date",
    "model",
    "capacity_bytes",
}


def prepare_smart_dataset(input_path: str | Path) -> pd.DataFrame:
    """Ingest, sanitize, and prepare a Backblaze hard drive telemetry CSV.

    Performs:
    1. Validation of CSV existence and structure.
    2. Verification of the binary 'failure' target column.
    3. Dynamic identification of all 'smart_*' feature columns.
    4. Exclusion of drive identifiers and metadata columns.
    5. Numeric coercion (coercing non-numeric items to NaN).
    6. Deterministic median imputation on input features.
    7. Construction of a deterministically ordered output DataFrame.

    Note on Imputation:
        Median imputation is computed directly on the ingested dataset for offline
        cleaning. Leak-free cross-validation transformers belong to CP3.8.

    Args:
        input_path: Path to the raw Backblaze CSV file.

    Returns:
        pd.DataFrame: Sanitized tabular dataset containing deterministically
            sorted SMART features followed by the 'failure' column.

    Raises:
        ValueError: If file is missing, not a CSV, lacks required columns,
            contains non-binary failure labels, or contains no valid rows.
    """
    path = Path(input_path)

    if not path.exists():
        raise ValueError(f"Input file does not exist: {path}")

    if path.suffix.lower() != ".csv":
        raise ValueError(f"Input file must be a CSV file, got: {path.name}")

    try:
        df = pd.read_csv(path)
    except Exception as exc:
        raise ValueError(f"Failed to read CSV from {path}: {exc}") from exc

    if df.empty:
        raise ValueError("Input CSV dataset contains no rows.")

    # Validate target column
    if "failure" not in df.columns:
        raise ValueError("Required target column 'failure' is missing from dataset.")

    failure_numeric = pd.to_numeric(df["failure"], errors="coerce")
    if failure_numeric.isna().any():
        raise ValueError("Target column 'failure' contains missing or non-numeric values.")

    unique_failures = set(failure_numeric.unique())
    if not unique_failures.issubset({0, 1}):
        raise ValueError(
            f"Target column 'failure' must be strictly binary (0 or 1), got: {unique_failures}"
        )

    # Identify SMART features dynamically (excluding known metadata)
    smart_columns = sorted([
        col for col in df.columns
        if col.startswith("smart_") and col not in EXCLUDED_METADATA_COLUMNS
    ])

    if not smart_columns:
        raise ValueError(
            "No valid SMART feature columns (starting with 'smart_') found in dataset."
        )

    # Coerce SMART columns to numeric and apply median imputation
    prepared_df = pd.DataFrame(index=df.index)

    for col in smart_columns:
        numeric_series = pd.to_numeric(df[col], errors="coerce")
        median_val = numeric_series.median()

        # Fallback to 0.0 if the column is entirely NaN
        fill_value = 0.0 if pd.isna(median_val) else float(median_val)
        prepared_df[col] = numeric_series.fillna(fill_value).astype(float)

    # Append target column as integer (0 or 1)
    prepared_df["failure"] = failure_numeric.astype(int)

    if prepared_df.empty:
        raise ValueError("No valid rows remain after processing.")

    return prepared_df


def save_prepared_dataset(
    dataframe: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    """Save the prepared SMART telemetry DataFrame to a destination CSV.

    Args:
        dataframe: Prepared SMART telemetry DataFrame.
        output_path: Target CSV file path.

    Returns:
        Path: Absolute or resolved Path to the saved file.

    Raises:
        ValueError: If the DataFrame is empty or missing 'failure' column.
    """
    if dataframe.empty:
        raise ValueError("Cannot save an empty DataFrame.")

    if "failure" not in dataframe.columns:
        raise ValueError("Prepared DataFrame must contain the target column 'failure'.")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output, index=False)
    return output
