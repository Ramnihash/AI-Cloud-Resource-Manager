"""Telemetry sequence preparation for CP3.7 Transformer anomaly detection.

Converts tabular device telemetry into fixed-length temporal sliding windows
suitable for training a Transformer-based sequence anomaly detector.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


TELEMETRY_FEATURES: list[str] = [
    "temperature",
    "error_rate",
    "response_latency",
    "queue_length",
    "io_activity",
]


_REQUIRED_COLUMNS: frozenset[str] = frozenset(TELEMETRY_FEATURES) | {
    "device_id",
    "timestamp",
}


def _validate_inputs(dataframe: pd.DataFrame, window_size: int) -> None:
    """Raise ValueError for any invalid input."""

    if not isinstance(window_size, int) or isinstance(window_size, bool):
        raise ValueError(
            f"window_size must be an integer, got {type(window_size).__name__!r}"
        )
    if window_size < 2:
        raise ValueError(
            f"window_size must be at least 2, got {window_size}"
        )


    missing = _REQUIRED_COLUMNS - set(dataframe.columns)
    if missing:
        raise ValueError(
            f"DataFrame is missing required columns: {sorted(missing)}"
        )


def create_telemetry_sequences(
    dataframe: pd.DataFrame,
    window_size: int,
) -> np.ndarray:
    """Convert tabular telemetry into fixed-length temporal sequences.

    Records are grouped by ``device_id``, sorted by ``timestamp`` in ascending
    order within each device, and then converted into overlapping sliding
    windows of length ``window_size``. Windows never span two different devices.

    Parameters
    ----------
    dataframe: pd.DataFrame
        Tabular telemetry containing at minimum the columns:
        ``device_id``, ``timestamp``, ``temperature``, ``error_rate``,
        ``response_latency``, ``queue_length``, ``io_activity``.
        Extra columns (e.g. ``label``, ``pattern``) are ignored.
    window_size: int
        Number of consecutive time-steps in each window. Must be >= 2.

    Returns
    -------
    np.ndarray
        Array of shape ``(n_windows, window_size, 5)`` where the last
        dimension corresponds to the five features in the order defined by
        :data:`TELEMETRY_FEATURES`.

    Raises
    ------
    ValueError
        If required columns are missing, ``window_size`` is invalid, or there
        are not enough records to form at least one window.
    """
    _validate_inputs(dataframe, window_size)

    all_windows: list[np.ndarray] = []

    for device_id, group in dataframe.groupby("device_id", sort=True):

        sorted_group = group.sort_values("timestamp", kind="stable")


        feature_array: np.ndarray = sorted_group[TELEMETRY_FEATURES].to_numpy(
            dtype=np.float64
        )

        n_rows = len(feature_array)
        n_windows_for_device = n_rows - window_size + 1

        if n_windows_for_device < 1:

            continue


        for start in range(n_windows_for_device):
            all_windows.append(feature_array[start : start + window_size])

    if not all_windows:
        raise ValueError(
            f"No device has enough records to create a window of size {window_size}. "
            "Provide more records or reduce window_size."
        )

    return np.stack(all_windows, axis=0)
