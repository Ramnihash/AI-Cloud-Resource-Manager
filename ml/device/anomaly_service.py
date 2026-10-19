"""Serving helpers for the telemetry anomaly detector (CP3.7 -> API).

Builds telemetry windows from the device simulator, loads the trained
Transformer reconstructor saved by ``scripts/train_device_anomaly_detector.py``
and turns its reconstruction error into an anomaly decision and a failure-risk
value for the predictive-action pipeline.

PyTorch is imported lazily so the rest of the backend works without it.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from ml.device.sequence import TELEMETRY_FEATURES
from modules.device.abnormal import AbnormalPattern, generate_abnormal_telemetry
from modules.device.device import Device
from modules.device.telemetry import generate_normal_telemetry

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "anomaly_detector.pt")
WINDOW_SIZE = 10
BACKUP_THRESHOLD = 0.7
FAULT_PATTERNS = [p.value for p in AbnormalPattern]


_cache: dict[str, tuple[float, dict[str, Any]]] = {}


_cache: dict[str, tuple[float, dict[str, Any]]] = {}


class ModelUnavailableError(RuntimeError):
    """The trained model file or PyTorch is not available."""


def build_window(
    device: Device,
    pattern: str | None = None,
    window_size: int = WINDOW_SIZE,
    start_index: int = 0,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Return a ``(window_size, 5)`` telemetry window and its raw samples.

    With ``pattern`` set, the second half of the window shows that fault
    (an onset in the middle of the window); otherwise all samples are normal.
    """
    onset = window_size // 2
    samples = []
    for i in range(window_size):
        index = start_index + i
        if pattern and i >= onset:
            sample = generate_abnormal_telemetry(device, float(index), pattern, sample_index=index)
        else:
            sample = generate_normal_telemetry(device, float(index), sample_index=index)
        samples.append(sample.to_dict())
    window = np.array([[s[f] for f in TELEMETRY_FEATURES] for s in samples], dtype=np.float64)
    return window, samples


def load_bundle(path: str = MODEL_PATH) -> dict[str, Any]:
    """Load the saved model plus its normalisation and threshold."""
    if not os.path.exists(path):
        raise ModelUnavailableError(
            "Anomaly detector is not trained yet. Run: python -m scripts.train_device_anomaly_detector"
        )
    mtime = os.path.getmtime(path)
    cached = _cache.get(path)
    if cached and cached[0] == mtime:
        return cached[1]
    mtime = os.path.getmtime(path)
    cached = _cache.get(path)
    if cached and cached[0] == mtime:
        return cached[1]
    try:
        import torch
        from ml.device.anomaly_detector import TelemetryTransformerReconstructor
    except ImportError as exc:
        raise ModelUnavailableError("PyTorch is not installed on the backend") from exc

    saved = torch.load(path, map_location="cpu", weights_only=False)
    model = TelemetryTransformerReconstructor(**saved["model_config"])
    model.load_state_dict(saved["state_dict"])
    model.eval()
    bundle = {
        "model": model,
        "mean": np.array(saved["mean"], dtype=np.float64),
        "std": np.array(saved["std"], dtype=np.float64),
        "threshold": float(saved["threshold"]),
        "window_size": int(saved["window_size"]),
        "trained_at": saved.get("trained_at"),
    }
    _cache[path] = (mtime, bundle)
    return bundle


def score_window(bundle: dict[str, Any], window: np.ndarray) -> float:
    """Reconstruction-error anomaly score for one raw telemetry window."""
    from ml.device.score_anomaly_detector import score_anomaly_sequences

    normalised = (window - bundle["mean"]) / bundle["std"]
    return float(score_anomaly_sequences(bundle["model"], normalised)[0])


def risk_from_score(score: float, threshold: float) -> float:
    """Map an anomaly score to a 0..1 failure-risk value.

    A score exactly at the detector threshold maps to the backup threshold
    (0.7), so every window flagged anomalous triggers the backup action.
    The mapping is linear below that and capped at 1.0.
    """
    if threshold <= 0:
        return 1.0
    return float(min(1.0, BACKUP_THRESHOLD * score / threshold))
