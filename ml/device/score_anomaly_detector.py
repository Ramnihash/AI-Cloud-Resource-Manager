from __future__ import annotations

import numpy as np
import torch

from ml.device.anomaly_detector import TelemetryTransformerReconstructor


def score_anomaly_sequences(
    model: TelemetryTransformerReconstructor,
    sequences: np.ndarray | torch.Tensor,
) -> np.ndarray:
    """
    Calculate one reconstruction-error anomaly score per telemetry sequence.

    Parameters
    ----------
    model:
        Trained TelemetryTransformerReconstructor.

    sequences:
        Telemetry sequences with shape:
        - (window_size, 5) for a single sequence, or
        - (N, window_size, 5) for a batch.

    Returns
    -------
    np.ndarray
        One non-negative reconstruction-error score per sequence.
        Shape: (N,)
    """
    if isinstance(sequences, np.ndarray):
        if not np.issubdtype(sequences.dtype, np.number):
            raise TypeError("sequences must contain numeric values")
        array = sequences
    elif isinstance(sequences, torch.Tensor):
        if not (sequences.is_floating_point() or sequences.is_complex()):
            raise TypeError("sequences must contain numeric values")
        array = sequences.detach().cpu().numpy()
    else:
        raise TypeError("sequences must be a NumPy array or PyTorch tensor")

    if array.ndim not in (2, 3):
        raise ValueError(
            "sequences must have shape (window_size, 5) "
            "or (N, window_size, 5)"
        )

    if array.size == 0:
        raise ValueError("sequences must not be empty")

    if array.shape[-1] != 5:
        raise ValueError("the final dimension must contain exactly 5 features")

    if array.ndim == 2:
        if array.shape[0] < 1:
            raise ValueError("window_size must be greater than zero")
        array = array[np.newaxis, :]
    else:
        if array.shape[1] < 1:
            raise ValueError("window_size must be greater than zero")
        if array.shape[0] < 1:
            raise ValueError("batch must contain at least one sequence")

    if not np.isfinite(array).all():
        raise ValueError("sequences must contain only finite values")

    input_tensor = torch.as_tensor(
        array,
        dtype=torch.float32,
        device="cpu",
    )

    was_training = model.training
    model.eval()

    try:
        with torch.no_grad():
            reconstructed = model(input_tensor)

            if reconstructed.shape != input_tensor.shape:
                raise ValueError(
                    "model output shape must match input shape"
                )

            scores = (
                (input_tensor - reconstructed) ** 2
            ).mean(dim=(1, 2))

    finally:
        model.train(was_training)

    result = scores.cpu().numpy()

    if not np.isfinite(result).all():
        raise ValueError("anomaly scores must be finite")

    if np.any(result < 0):
        raise ValueError("anomaly scores must be non-negative")

    return result