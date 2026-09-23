"""Transformer reconstruction model for CP3.7 device telemetry anomaly detection.

The model learns to reconstruct normal device telemetry sequences.
Anomaly detection (reconstruction-error scoring and thresholding) is NOT
implemented here; it belongs to the next CP3.7 step.

Architecture overview:
    Input projection   : (input_dim=5) → d_model
    Positional encoding: sinusoidal, added to the projected sequence
    Transformer encoder: n_head × num_layers (encoder-only)
    Output projection  : d_model → (input_dim=5)

The model is intentionally small so it can run on a standard CPU.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class _SinusoidalPositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding (Vaswani et al., 2017).

    Encodes temporal position into the sequence before passing it to the
    Transformer encoder. The encoding is computed once and cached.
    """

    def __init__(self, d_model: int, max_len: int = 4096, dropout: float = 0.0) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)


        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term[: d_model // 2])



        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to *x* without modifying it in-place.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(batch, seq_len, d_model)``.
        """
        seq_len = x.size(1)
        return self.dropout(x + self.pe[:, :seq_len, :])


class TelemetryTransformerReconstructor(nn.Module):
    """Small Transformer encoder that reconstructs device telemetry sequences.

    Parameters
    ----------
    input_dim : int
        Number of telemetry features (must be 5 for this project).
    d_model : int
        Internal embedding dimension of the Transformer.
    nhead : int
        Number of attention heads; must divide ``d_model`` evenly.
    num_layers : int
        Number of stacked ``TransformerEncoderLayer`` blocks.
    dim_feedforward : int
        Width of the feed-forward sublayer inside each encoder block.
    dropout : float
        Dropout rate applied in the Transformer layers and positional encoding.
    """

    def __init__(
        self,
        input_dim: int = 5,
        d_model: int = 32,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 64,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()


        if not isinstance(input_dim, int) or input_dim < 1:
            raise ValueError(f"input_dim must be a positive integer, got {input_dim!r}")
        if not isinstance(d_model, int) or d_model < 1:
            raise ValueError(f"d_model must be a positive integer, got {d_model!r}")
        if not isinstance(nhead, int) or nhead < 1:
            raise ValueError(f"nhead must be a positive integer, got {nhead!r}")
        if d_model % nhead != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by nhead ({nhead})"
            )
        if not isinstance(num_layers, int) or num_layers < 1:
            raise ValueError(f"num_layers must be a positive integer, got {num_layers!r}")
        if not isinstance(dim_feedforward, int) or dim_feedforward < 1:
            raise ValueError(f"dim_feedforward must be a positive integer, got {dim_feedforward!r}")
        if not isinstance(dropout, (int, float)) or not (0.0 <= float(dropout) <= 1.0):
            raise ValueError(f"dropout must be a float in [0, 1], got {dropout!r}")

        self.input_dim = input_dim
        self.d_model = d_model


        self.input_projection = nn.Linear(input_dim, d_model)

        self.positional_encoding = _SinusoidalPositionalEncoding(
            d_model=d_model,
            dropout=float(dropout),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=float(dropout),
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_projection = nn.Linear(d_model, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstruct the input telemetry sequence.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(batch_size, window_size, input_dim)``.

        Returns
        -------
        torch.Tensor
            Reconstructed tensor with the same shape as *x*.

        Raises
        ------
        ValueError
            If *x* is not 3-dimensional or its last dimension does not match
            ``self.input_dim``.
        """
        if x.dim() != 3:
            raise ValueError(
                f"Input tensor must be 3-dimensional (batch, seq, features), "
                f"got {x.dim()}D tensor with shape {tuple(x.shape)}"
            )
        if x.size(-1) != self.input_dim:
            raise ValueError(
                f"Last dimension of input ({x.size(-1)}) does not match "
                f"model input_dim ({self.input_dim})"
            )


        projected = self.input_projection(x)
        encoded_pos = self.positional_encoding(projected)


        encoded = self.encoder(encoded_pos)


        reconstruction = self.output_projection(encoded)
        return reconstruction
