"""
First-order Markov page predictor for Module 2 — Predictive Memory Management.

This module provides :class:`MarkovPagePredictor`, which learns page-to-page
transition probabilities from a training sequence and predicts the most likely
next page(s) to be accessed.

Public API::

    predictor = MarkovPagePredictor()

    # 1. Train on any page-reference sequence (e.g., from PageReferenceGenerator)
    train_result = predictor.train(sequence)

    # 2. Predict the next page(s) from the current page
    candidates = predictor.predict(current_page=7, top_k=3)
    # → [{"page": 5, "probability": 0.60}, {"page": 3, "probability": 0.25}, ...]

    # 3. Evaluate prediction accuracy on a held-out test sequence
    metrics = predictor.evaluate(test_sequence, top_k=1)

Model:
    First-order Markov chain.  The probability of the next page depends only
    on the current page:

        P(page_{i+1} | page_i)

    Transition probabilities are normalised row-wise from raw counts.
    No Laplace smoothing is applied — unseen states return an empty list so
    that CP4.2 (prediction-to-action layer) can apply its own fallback policy.

Relationship to other checkpoints:
    * Consumes PageReferenceGenerator output (CP2.2):
        result = generator.generate(num_pages=32, length=500,
                                    pattern="locality", seed=42)
        predictor.train(result["sequence"])
    * Complements CP1.2 classical replacement: predictions can later guide
      prefetch / page-protection actions in CP4.2.
    * CP3.4 (Transformer predictor) is a higher-capacity alternative.

Actions (prefetch, page protection) are NOT implemented here — they belong
to CP4.2 and are explicitly out of scope for CP3.3.
"""

from collections import defaultdict


class MarkovPagePredictor:
    """First-order Markov page predictor.

    Learns pairwise page-transition probabilities from a training sequence
    and ranks next-page candidates by probability.

    Lifecycle::

        predictor = MarkovPagePredictor()
        predictor.train(sequence)          # required before predict/evaluate
        predictor.predict(current_page)
        predictor.evaluate(test_sequence)

    Attributes (populated after :meth:`train`)
    ------------------------------------------
    transition_table : dict
        Normalised first-order transition table.
        ``transition_table[from_page][to_page]`` is the empirical probability
        that *to_page* immediately follows *from_page* in the training data.
    """

    def __init__(self):
        self.transition_table: dict = {}
        self._training_length: int = 0
        self._trained: bool = False

    # ── Public methods ────────────────────────────────────────────────────────

    def train(self, sequence: list) -> dict:
        """Learn transition probabilities from *sequence*.

        Parameters
        ----------
        sequence : list[int]
            Ordered page-reference sequence.  Must contain at least 2 elements
            so that at least one transition can be observed.

        Returns
        -------
        dict
            Training summary:

            ``order``            – Markov order used (always 1) (int)
            ``num_pages_seen``   – distinct page IDs in *sequence* (int)
            ``num_states``       – states with ≥ 1 outgoing transition (int)
            ``training_length``  – len(sequence) (int)
            ``transition_table`` – normalised table (same object as self.transition_table)

        Raises
        ------
        TypeError
            If *sequence* is not a list.
        ValueError
            If *sequence* has fewer than 2 elements.
        """
        if not isinstance(sequence, list):
            raise TypeError(
                f"sequence must be a list, got {type(sequence).__name__}"
            )
        if len(sequence) < 2:
            raise ValueError(
                f"sequence must have at least 2 elements to form a transition, "
                f"got {len(sequence)}"
            )

        # ── Count transitions ─────────────────────────────────────────────
        counts: dict = defaultdict(lambda: defaultdict(int))
        for i in range(len(sequence) - 1):
            counts[sequence[i]][sequence[i + 1]] += 1

        # ── Normalise to probabilities ────────────────────────────────────
        table: dict = {}
        for from_page, to_counts in counts.items():
            total = sum(to_counts.values())
            table[from_page] = {
                to_page: count / total
                for to_page, count in to_counts.items()
            }

        self.transition_table = table
        self._training_length = len(sequence)
        self._trained = True

        return {
            "order":            1,
            "num_pages_seen":   len(set(sequence)),
            "num_states":       len(table),
            "training_length":  self._training_length,
            "transition_table": self.transition_table,
        }

    def predict(self, current_page, top_k: int = 1) -> list:
        """Predict the most likely next page(s) given *current_page*.

        Parameters
        ----------
        current_page : int
            The page most recently accessed.
        top_k : int
            Number of candidates to return.  Must be an integer >= 1.
            The list may be shorter than *top_k* when the current page has
            fewer outgoing transitions, or empty when it was never seen during
            training.

        Returns
        -------
        list[dict]
            Candidates sorted by probability descending.  Each entry::

                {"page": int, "probability": float}

            Returns ``[]`` if *current_page* has no outgoing transitions
            (i.e., was never seen as a from-state during training).

        Raises
        ------
        ValueError
            If *top_k* is not an integer or is < 1.
        RuntimeError
            If :meth:`train` has not been called yet.
        """
        self._require_trained("predict")
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
            raise ValueError(
                f"top_k must be an integer >= 1, got {top_k!r}"
            )

        if current_page not in self.transition_table:
            return []

        sorted_candidates = sorted(
            self.transition_table[current_page].items(),
            key=lambda item: item[1],
            reverse=True,
        )

        return [
            {"page": page, "probability": prob}
            for page, prob in sorted_candidates[:top_k]
        ]

    def evaluate(self, test_sequence: list, top_k: int = 1) -> dict:
        """Evaluate prediction accuracy on *test_sequence*.

        For each consecutive pair ``(test_sequence[i], test_sequence[i+1])``,
        the predictor attempts to predict the next page from the current one.
        Top-1 and top-*k* accuracy are computed.

        Parameters
        ----------
        test_sequence : list[int]
            Ordered page-reference sequence.  Must have at least 2 elements.
        top_k : int
            A prediction is counted as top-*k* correct if the actual next page
            appears anywhere in the top-*k* candidates.  Must be >= 1.

        Returns
        -------
        dict
            Keys:

            ``total_predictions`` – len(test_sequence) - 1 (int)
            ``correct_top1``      – transitions where top-1 candidate matched (int)
            ``correct_topk``      – transitions where actual page is in top-k (int)
            ``top1_accuracy``     – correct_top1 / total_predictions (float)
            ``topk_accuracy``     – correct_topk / total_predictions (float)
            ``top_k``             – *top_k* value used (int)

        Raises
        ------
        TypeError
            If *test_sequence* is not a list.
        ValueError
            If *test_sequence* has fewer than 2 elements, or *top_k* < 1.
        RuntimeError
            If :meth:`train` has not been called yet.
        """
        self._require_trained("evaluate")
        if not isinstance(test_sequence, list):
            raise TypeError(
                f"test_sequence must be a list, got {type(test_sequence).__name__}"
            )
        if len(test_sequence) < 2:
            raise ValueError(
                f"test_sequence must have at least 2 elements, "
                f"got {len(test_sequence)}"
            )
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
            raise ValueError(
                f"top_k must be an integer >= 1, got {top_k!r}"
            )

        total = len(test_sequence) - 1
        correct_top1 = 0
        correct_topk = 0

        for i in range(total):
            current = test_sequence[i]
            actual_next = test_sequence[i + 1]

            preds = self.predict(current, top_k=top_k)
            pred_pages = [p["page"] for p in preds]

            if pred_pages and pred_pages[0] == actual_next:
                correct_top1 += 1
            if actual_next in pred_pages:
                correct_topk += 1

        top1_acc = correct_top1 / total if total > 0 else 0.0
        topk_acc = correct_topk / total if total > 0 else 0.0

        return {
            "total_predictions": total,
            "correct_top1":      correct_top1,
            "correct_topk":      correct_topk,
            "top1_accuracy":     top1_acc,
            "topk_accuracy":     topk_acc,
            "top_k":             top_k,
        }

    # ── Private helpers ───────────────────────────────────────────────────────

    def _require_trained(self, method_name: str) -> None:
        """Raise RuntimeError if :meth:`train` has not yet been called."""
        if not self._trained:
            raise RuntimeError(
                f"MarkovPagePredictor.{method_name}() called before train(). "
                f"Call train(sequence) first."
            )
