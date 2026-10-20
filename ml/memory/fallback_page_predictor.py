"""
Fallback page predictor for Module 2 — Predictive Memory Management (CP7.4).

Provides :class:`FallbackPagePredictor`, which combines a primary high-capacity
predictor (e.g., :class:`TransformerPagePredictor`) with a lightweight,
deterministic fallback predictor (e.g., :class:`MarkovPagePredictor`).

When the primary predictor fails—due to missing checkpoints, corrupted weights,
insufficient context length, uninitialized state, or runtime exceptions—the
fallback predictor seamlessly services the request.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ml.memory.markov_predictor import MarkovPagePredictor
from ml.memory.transformer_predictor import TransformerPagePredictor

logger = logging.getLogger(__name__)


class FallbackPagePredictor:
    """Hybrid page predictor with automatic fallback to Markov.

    Primary: TransformerPagePredictor (higher capacity, handles multi-step context).
    Fallback: MarkovPagePredictor (zero-dependency, instant CPU lookup, high reliability).

    Public API matches existing predictors:
        candidates = predictor.predict(context, top_k=3)
        # -> [{"page": int, "probability": float}, ...]

    Extended API for inspection and telemetry:
        status = predictor.predict_with_status(context, top_k=3)
        # -> {"candidates": [...], "used_fallback": True, "active_model": "markov", ...}
    """

    def __init__(
        self,
        primary_predictor: Optional[TransformerPagePredictor] = None,
        fallback_predictor: Optional[MarkovPagePredictor] = None,
    ):
        self.primary_predictor: Optional[TransformerPagePredictor] = primary_predictor
        self.fallback_predictor: MarkovPagePredictor = (
            fallback_predictor if fallback_predictor is not None else MarkovPagePredictor()
        )
        self.last_status: Dict[str, Any] = {}
        self.fallback_history: List[Dict[str, Any]] = []

    def load_transformer_checkpoint(
        self,
        weights_path: Union[str, Path],
        num_pages: int = 32,
        context_len: int = 8,
        d_model: int = 32,
        nhead: int = 2,
        d_ff: int = 64,
        seed: int = 2026,
        device: str = "cpu",
    ) -> bool:
        """Attempt to load a trained Transformer model from checkpoint weights.

        If checkpoint loading fails for any reason (missing file, corrupted weights,
        incompatible state dict, PyTorch load error), this method returns False and
        leaves the fallback predictor ready to handle requests without crashing.

        Returns:
            bool: True if checkpoint was loaded successfully, False otherwise.
        """
        try:
            import torch

            weights_file = Path(weights_path)
            if not weights_file.exists():
                raise FileNotFoundError(f"Checkpoint file not found: {weights_file}")

            predictor = TransformerPagePredictor(
                num_pages=num_pages,
                context_len=context_len,
                d_model=d_model,
                nhead=nhead,
                d_ff=d_ff,
                seed=seed,
            )
            predictor._init_model()
            state_dict = torch.load(weights_file, map_location=device, weights_only=True)
            predictor.model.load_state_dict(state_dict)
            predictor._trained = True
            self.primary_predictor = predictor
            return True
        except Exception as exc:
            logger.warning("Failed to load Transformer checkpoint from %s: %s", weights_path, exc)
            self.last_status = {
                "load_success": False,
                "weights_path": str(weights_path),
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            }
            return False

    def train_fallback(self, sequence: list) -> dict:
        """Train the fallback Markov predictor on a sequence."""
        if self.fallback_predictor is None:
            self.fallback_predictor = MarkovPagePredictor()
        return self.fallback_predictor.train(sequence)

    def train_primary(self, sequence: list, epochs: int = 5, lr: float = 1e-3) -> dict:
        """Train the primary Transformer predictor on a sequence."""
        if self.primary_predictor is None:
            raise RuntimeError("Primary predictor is not configured. Initialize it first.")
        return self.primary_predictor.train(sequence, epochs=epochs, lr=lr)

    def predict(
        self,
        context: Union[List[int], int],
        top_k: int = 1,
    ) -> List[Dict[str, Any]]:
        """Predict the most likely next page(s).

        Matches standard predictor interface: returns list of dicts with
        {"page": int, "probability": float}.
        """
        status = self.predict_with_status(context=context, top_k=top_k)
        return status["candidates"]

    def predict_with_status(
        self,
        context: Union[List[int], int],
        top_k: int = 1,
    ) -> Dict[str, Any]:
        """Predict the next page(s) and return full status metadata.

        Returns:
            dict containing:
                - candidates: list of candidate dicts [{"page": int, "probability": float}]
                - used_fallback: bool
                - active_model: "transformer", "markov", or "none"
                - fallback_reason: str or None
                - error_type: str or None
                - context: the input provided
                - top_k: int
        """
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
            raise ValueError(f"top_k must be an integer >= 1, got {top_k!r}")

        if not isinstance(context, (list, int)) or isinstance(context, bool):
            raise TypeError(
                f"context must be a list or int, got {type(context).__name__}"
            )

        if isinstance(context, list) and len(context) == 0:
            status = {
                "candidates": [],
                "used_fallback": False,
                "active_model": "none",
                "fallback_reason": "Empty context provided",
                "error_type": None,
                "context": context,
                "top_k": top_k,
            }
            self.last_status = status
            return status

        used_fallback = False
        fallback_reason: Optional[str] = None
        error_type: Optional[str] = None
        active_model = "none"
        candidates: List[Dict[str, Any]] = []

        # 1. Attempt primary prediction if configured
        if self.primary_predictor is None:
            used_fallback = True
            fallback_reason = "Primary predictor is not configured"
            error_type = "ConfigurationError"
        else:
            try:
                # Transformer expects list context
                if isinstance(context, int):
                    raise ValueError(
                        f"Transformer requires list context, got single int {context}"
                    )
                candidates = self.primary_predictor.predict(context, top_k=top_k)
                active_model = "transformer"
            except Exception as exc:
                used_fallback = True
                error_type = type(exc).__name__
                fallback_reason = f"{error_type}: {str(exc)}"

        # 2. If primary failed or unconfigured, execute fallback
        if used_fallback:
            if self.fallback_predictor is None:
                candidates = []
                active_model = "none"
            else:
                # Derive single current_page for Markov
                if isinstance(context, list):
                    current_page = context[-1]
                else:
                    current_page = context

                try:
                    candidates = self.fallback_predictor.predict(current_page, top_k=top_k)
                    active_model = "markov"
                except Exception as fb_exc:
                    logger.warning("Fallback predictor also failed: %s", fb_exc)
                    candidates = []
                    active_model = "none"
                    fallback_reason = (
                        f"{fallback_reason} | Secondary fallback failure: "
                        f"{type(fb_exc).__name__}: {str(fb_exc)}"
                    )

        status = {
            "candidates": candidates,
            "used_fallback": used_fallback,
            "active_model": active_model,
            "fallback_reason": fallback_reason,
            "error_type": error_type,
            "context": context,
            "top_k": top_k,
        }

        self.last_status = status
        if used_fallback:
            self.fallback_history.append(status)

        return status

    def clear_history(self) -> None:
        """Clear recorded fallback history."""
        self.fallback_history.clear()
        self.last_status.clear()
