"""Train the device telemetry anomaly detector and save it for the backend.

Usage (from the repository root, with PyTorch installed):

    python -m scripts.train_device_anomaly_detector

Trains on deterministic *normal* telemetry from the simulated fleet, sets the
anomaly threshold from held-out normal data, then reports how well each fault
pattern is detected. Writes ``ml/device/models/anomaly_detector.pt`` (git-ignored).
"""

from __future__ import annotations

import datetime
import os

import numpy as np
import pandas as pd
import torch

from ml.device.anomaly_detector import TelemetryTransformerReconstructor
from ml.device.anomaly_service import FAULT_PATTERNS, MODEL_PATH, WINDOW_SIZE, build_window
from ml.device.score_anomaly_detector import score_anomaly_sequences
from ml.device.sequence import TELEMETRY_FEATURES, create_telemetry_sequences
from ml.device.train_anomaly_detector import train_anomaly_detector
from modules.device.fleet import create_default_fleet
from modules.device.telemetry import generate_normal_telemetry

TRAIN_SAMPLES = 200
VAL_SAMPLES = 100
EPOCHS = 40
THRESHOLD_MARGIN = 1.5
SEED = 42
MODEL_CONFIG = {"input_dim": 5, "d_model": 32, "nhead": 4, "num_layers": 2, "dim_feedforward": 64, "dropout": 0.0}


def _telemetry_frame(fleet, start: int, count: int) -> pd.DataFrame:
    rows = []
    for device in fleet:
        for i in range(start, start + count):
            rows.append(generate_normal_telemetry(device, float(i), sample_index=i).to_dict())
    return pd.DataFrame(rows)


def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    fleet = create_default_fleet()

    train = create_telemetry_sequences(_telemetry_frame(fleet, 0, TRAIN_SAMPLES), WINDOW_SIZE)
    val = create_telemetry_sequences(_telemetry_frame(fleet, TRAIN_SAMPLES, VAL_SAMPLES), WINDOW_SIZE)

    flat = train.reshape(-1, len(TELEMETRY_FEATURES))
    mean = flat.mean(axis=0)
    std = np.maximum(flat.std(axis=0), 1e-6)

    model, history = train_anomaly_detector(
        (train - mean) / std,
        model=TelemetryTransformerReconstructor(**MODEL_CONFIG),
        epochs=EPOCHS,
    )
    print(f"Trained on {len(train)} windows. Loss {history[0]:.4f} -> {history[-1]:.4f}")

    val_scores = score_anomaly_sequences(model, (val - mean) / std)
    threshold = float(val_scores.max() * THRESHOLD_MARGIN)
    print(f"Held-out normal scores: max {val_scores.max():.4f}. Threshold set to {threshold:.4f}")

    saved = {
        "state_dict": model.state_dict(),
        "model_config": MODEL_CONFIG,
        "mean": mean.tolist(),
        "std": std.tolist(),
        "threshold": threshold,
        "window_size": WINDOW_SIZE,
        "trained_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    torch.save(saved, MODEL_PATH)
    print(f"Saved {MODEL_PATH}")

    print("\nDetection check (served windows, start index 500):")
    false_alarms = 0
    for device in fleet:
        window, _ = build_window(device, None, WINDOW_SIZE, 500)
        score = float(score_anomaly_sequences(model, (window - mean) / std)[0])
        false_alarms += score > threshold
    print(f"  normal windows flagged: {false_alarms} of {len(fleet)}")
    for pattern in FAULT_PATTERNS:
        caught = 0
        for device in fleet:
            window, _ = build_window(device, pattern, WINDOW_SIZE, 500)
            score = float(score_anomaly_sequences(model, (window - mean) / std)[0])
            caught += score > threshold
        print(f"  {pattern}: detected {caught} of {len(fleet)} devices")


if __name__ == "__main__":
    main()
