"""Training utility for CP3.7 Transformer-based device telemetry anomaly detection.

Trains the :class:`TelemetryTransformerReconstructor` on *normal* telemetry
sequences using MSE reconstruction loss.

Pipeline position
-----------------
This module covers the first step of the anomaly-detection pipeline::

    normal telemetry sequences
        → Transformer training          ← THIS MODULE
        → reconstruction error scoring  (next CP3.7 step)
        → anomaly threshold             (later)
        → anomaly decision              (later)
        → recovery action               (later)

Nothing in this module performs scoring, thresholding, or classification.
"""

from __future__ import annotations

import math
from typing import Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ml.device.anomaly_detector import TelemetryTransformerReconstructor


__all__ = ["train_anomaly_detector"]


_ArrayLike = Union[np.ndarray, torch.Tensor]


_EXPECTED_FEATURES: int = 5






def _validate_hyperparameters(
    epochs: int,
    learning_rate: float,
    batch_size: int,
) -> None:
    """Raise ValueError for any invalid hyperparameter value."""
    if not isinstance(epochs, int) or isinstance(epochs, bool) or epochs < 1:
        raise ValueError(
            f"epochs must be a positive integer, got {epochs!r}"
        )
    if (
        not isinstance(learning_rate, (int, float))
        or isinstance(learning_rate, bool)
        or learning_rate <= 0.0
    ):
        raise ValueError(
            f"learning_rate must be a positive number, got {learning_rate!r}"
        )
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
        raise ValueError(
            f"batch_size must be a positive integer, got {batch_size!r}"
        )


def _validate_and_convert(data: _ArrayLike) -> torch.Tensor:
    """Validate *data* and return it as a CPU ``torch.FloatTensor``.

    Parameters
    ----------
    data:
        Raw normal-sequence array with shape
        ``(num_sequences, window_size, 5)``.

    Returns
    -------
    torch.Tensor
        Validated tensor of dtype ``torch.float32`` on CPU.

    Raises
    ------
    ValueError
        For any shape, type, or value problem.
    """

    if isinstance(data, np.ndarray):
        tensor = torch.from_numpy(data).float()
    elif isinstance(data, torch.Tensor):
        tensor = data.float()
    else:
        raise ValueError(
            f"normal_sequences must be a NumPy ndarray or torch.Tensor, "
            f"got {type(data).__name__!r}"
        )


    tensor = tensor.cpu()


    if tensor.dim() != 3:
        raise ValueError(
            f"normal_sequences must be 3-dimensional "
            f"(num_sequences, window_size, {_EXPECTED_FEATURES}), "
            f"got {tensor.dim()}D tensor with shape {tuple(tensor.shape)}"
        )
    if tensor.size(-1) != _EXPECTED_FEATURES:
        raise ValueError(
            f"Last dimension of normal_sequences must be {_EXPECTED_FEATURES} "
            f"(one per telemetry feature), got {tensor.size(-1)}"
        )
    if tensor.size(0) < 1:
        raise ValueError(
            "normal_sequences must contain at least one sequence, "
            f"got shape {tuple(tensor.shape)}"
        )


    if not torch.isfinite(tensor).all():
        raise ValueError(
            "normal_sequences contains non-finite values (NaN or ±inf). "
            "Ensure all telemetry values are finite before training."
        )

    return tensor






def train_anomaly_detector(
    normal_sequences: _ArrayLike,
    model: TelemetryTransformerReconstructor | None = None,
    epochs: int = 10,
    learning_rate: float = 1e-3,
    batch_size: int = 32,
) -> tuple[TelemetryTransformerReconstructor, list[float]]:
    """Train :class:`TelemetryTransformerReconstructor` on normal sequences.

    Parameters
    ----------
    normal_sequences:
        Array of shape ``(num_sequences, window_size, 5)`` containing ONLY
        normal (non-anomalous) telemetry windows.  Accepted as either a
        ``numpy.ndarray`` or a ``torch.Tensor``.
    model:
        An existing :class:`TelemetryTransformerReconstructor` instance to
        continue training, or ``None`` to create a fresh default model.
    epochs:
        Number of full passes over *normal_sequences*. Must be >= 1.
    learning_rate:
        Adam optimiser learning rate. Must be > 0.
    batch_size:
        Number of sequences per mini-batch. Must be >= 1.

    Returns
    -------
    model : TelemetryTransformerReconstructor
        The trained model (same object that was passed in, or a freshly
        created one if *model* was ``None``).
    history : list[float]
        One mean MSE loss value per epoch (length == *epochs*).

    Raises
    ------
    ValueError
        If *normal_sequences* has the wrong shape, contains non-finite values,
        or any hyperparameter is invalid.
    """

    _validate_hyperparameters(epochs, learning_rate, batch_size)


    tensor = _validate_and_convert(normal_sequences)


    if model is None:
        model = TelemetryTransformerReconstructor()


    model = model.cpu()


    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()


    dataset = TensorDataset(tensor)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )


    history: list[float] = []

    model.train()
    for _ in range(epochs):
        epoch_loss_sum = 0.0
        num_batches = 0

        for (batch,) in loader:
            optimiser.zero_grad()
            reconstruction = model(batch)
            loss = criterion(reconstruction, batch)
            loss.backward()
            optimiser.step()

            epoch_loss_sum += loss.item()
            num_batches += 1



        epoch_mean_loss = epoch_loss_sum / num_batches if num_batches > 0 else math.nan
        history.append(epoch_mean_loss)

    return model, history
