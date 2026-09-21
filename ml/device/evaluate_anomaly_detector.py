from __future__ import annotations

import numpy as np
import torch

from ml.device.anomaly_detector import TelemetryTransformerReconstructor
from ml.device.score_anomaly_detector import score_anomaly_sequences


def calculate_anomaly_threshold(
    normal_scores: np.ndarray,
    percentile: float = 95.0,
) -> float:
    """
    Calculate an anomaly threshold from normal validation scores.

    Parameters
    ----------
    normal_scores:
        Reconstruction-error scores obtained from normal validation data.

    percentile:
        Percentile used as the anomaly threshold.

    Returns
    -------
    float
        Reconstruction-error threshold.
    """
    scores = np.asarray(normal_scores, dtype=np.float64)

    if scores.ndim != 1:
        raise ValueError("normal_scores must be one-dimensional")

    if scores.size == 0:
        raise ValueError("normal_scores must not be empty")

    if not np.isfinite(scores).all():
        raise ValueError("normal_scores must contain only finite values")

    if np.any(scores < 0):
        raise ValueError("normal_scores must be non-negative")

    if not 0.0 < percentile < 100.0:
        raise ValueError("percentile must be between 0 and 100")

    return float(np.percentile(scores, percentile))


def classify_anomaly_scores(
    scores: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """
    Classify reconstruction-error scores using an anomaly threshold.

    Returns True for anomalous scores and False otherwise.
    """
    values = np.asarray(scores, dtype=np.float64)

    if values.ndim != 1:
        raise ValueError("scores must be one-dimensional")

    if values.size == 0:
        raise ValueError("scores must not be empty")

    if not np.isfinite(values).all():
        raise ValueError("scores must contain only finite values")

    if np.any(values < 0):
        raise ValueError("scores must be non-negative")

    if not np.isfinite(threshold):
        raise ValueError("threshold must be finite")

    if threshold < 0:
        raise ValueError("threshold must be non-negative")

    return values > threshold


def evaluate_anomaly_detector(
    model: TelemetryTransformerReconstructor,
    normal_validation_sequences: np.ndarray | torch.Tensor,
    anomalous_sequences: np.ndarray | torch.Tensor,
    percentile: float = 95.0,
) -> dict[str, object]:
    """
    Calculate a threshold from normal validation data and evaluate
    injected anomalous telemetry sequences.

    This function performs evaluation only. It does not perform
    recovery, rerouting, backup, or other device actions.
    """
    normal_scores = score_anomaly_sequences(
        model,
        normal_validation_sequences,
    )

    anomaly_scores = score_anomaly_sequences(
        model,
        anomalous_sequences,
    )

    threshold = calculate_anomaly_threshold(
        normal_scores,
        percentile=percentile,
    )

    predictions = classify_anomaly_scores(
        anomaly_scores,
        threshold,
    )

    return {
        "threshold": threshold,
        "normal_scores": normal_scores,
        "anomaly_scores": anomaly_scores,
        "anomaly_predictions": predictions,
        "normal_count": int(len(normal_scores)),
        "anomaly_count": int(len(anomaly_scores)),
        "detected_anomaly_count": int(np.sum(predictions)),
    }