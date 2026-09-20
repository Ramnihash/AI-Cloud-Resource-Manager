"""
Page-reference dataset generator for Module 2 — Predictive Memory Management.

This module provides :class:`PageReferenceGenerator`, which generates synthetic
page-reference sequences to feed the memory ML pipeline.

Public API::

    gen = PageReferenceGenerator()
    result = gen.generate(
        num_pages=32,
        length=500,
        pattern="locality",
        seed=42,
        save_path="data/memory/refs.csv",  # optional
    )

Supported patterns (implementation choices designed to support the Markov and
Transformer next-page predictors in CP3.3 and CP3.4; not explicitly mandated
by the project specification):

    ``"uniform"``
        Each reference is drawn uniformly at random from [0, num_pages-1].
        Serves as a baseline / stress-test for predictors.

    ``"locality"``
        Working-set locality: a small hot set of pages (approx. 25% of the
        page space) is accessed with high probability (80%), while the
        remaining cold pages are accessed less frequently.  This is the most
        OS-realistic pattern and the one most useful for training the Markov
        predictor (CP3.3), which learns from repeated transition clusters.

    ``"sequential"``
        Pages accessed in cyclic order: 0, 1, 2, ..., num_pages-1, 0, 1, ...
        Fully deterministic (seed has no effect).  Useful for predictor sanity
        checks and verifying cyclic-access optimisations.

CSV output (optional):
    When ``save_path`` is supplied, the sequence is written as a CSV with a
    single column ``page_id``, one row per reference.  The built-in ``csv``
    module is used — no external dependencies.

Design principle (DEVELOPMENT.md §5):
    Data generation is a supporting component, not a standalone module.  This
    generator feeds the AI components in ml/memory/ (CP3.3, CP3.4).
"""

import csv
import random


# ── Module-level constants ────────────────────────────────────────────────────

VALID_PATTERNS = ("uniform", "locality", "sequential")

# Locality pattern tunables — kept as module-level constants so tests can
# reference them without magic numbers.
_HOT_FRACTION = 0.25    # fraction of pages that form the hot working set
_HOT_PROBABILITY = 0.80  # probability of drawing from the hot set


# ── Generator ─────────────────────────────────────────────────────────────────

class PageReferenceGenerator:
    """Synthetic page-reference sequence generator.

    Generates ordered sequences of page numbers that simulate OS memory
    access patterns.  Each :meth:`generate` call is self-contained; the
    class itself holds no state between calls.

    Sequences produced by this class are suitable for:

    * Feeding directly into FIFO/LRU/Optimal algorithms (CP1.2)
    * Training the Markov next-page predictor (CP3.3)
    * Training the Transformer next-page predictor (CP3.4)
    """

    def generate(
        self,
        num_pages: int,
        length: int,
        pattern: str,
        seed=None,
        save_path=None,
    ) -> dict:
        """Generate a synthetic page-reference sequence.

        Parameters
        ----------
        num_pages : int
            Size of the virtual page space.  Pages are numbered 0 to
            ``num_pages - 1``.  Must be >= 1.
        length : int
            Number of page references to generate.  Must be >= 1.
        pattern : str
            Access pattern.  One of ``"uniform"``, ``"locality"``,
            ``"sequential"``.  See module docstring for descriptions.
        seed : int or None, optional
            Random seed for reproducibility.  When provided, identical
            arguments always produce the same sequence.  ``None`` (default)
            uses non-deterministic random state.  Has no effect on the
            ``"sequential"`` pattern, which is fully deterministic.
        save_path : str or None, optional
            If provided, the sequence is written to this path as a CSV with
            header ``page_id``, one row per reference.  The file is created
            or overwritten.  Default is ``None`` (do not save).

        Returns
        -------
        dict
            Keys:

            * ``sequence``  – generated page references (list[int])
            * ``num_pages`` – size of the page space (int)
            * ``length``    – number of references generated (int)
            * ``pattern``   – pattern used (str)
            * ``seed``      – seed used, or None (int | None)
            * ``saved_to``  – path written to, or None (str | None)

        Raises
        ------
        ValueError
            If ``num_pages < 1``, ``length < 1``, or ``pattern`` is not a
            recognised pattern name.
        """
        self._validate(num_pages, length, pattern)

        # Isolated RNG — does not affect global random state
        rng = random.Random(seed)

        # ── Generate sequence ─────────────────────────────────────────────
        if pattern == "uniform":
            sequence = self._generate_uniform(rng, num_pages, length)
        elif pattern == "locality":
            sequence = self._generate_locality(rng, num_pages, length)
        else:  # "sequential"
            sequence = self._generate_sequential(num_pages, length)

        # ── Optional CSV save ─────────────────────────────────────────────
        saved_to = None
        if save_path is not None:
            self._save_csv(sequence, save_path)
            saved_to = save_path

        return {
            "sequence":  sequence,
            "num_pages": num_pages,
            "length":    length,
            "pattern":   pattern,
            "seed":      seed,
            "saved_to":  saved_to,
        }

    # ── Pattern generators ────────────────────────────────────────────────────

    @staticmethod
    def _generate_uniform(rng: random.Random, num_pages: int, length: int) -> list:
        """Each reference drawn uniformly at random from [0, num_pages-1]."""
        return [rng.randint(0, num_pages - 1) for _ in range(length)]

    @staticmethod
    def _generate_locality(
        rng: random.Random, num_pages: int, length: int
    ) -> list:
        """Working-set locality pattern.

        A hot set of ``ceil(_HOT_FRACTION * num_pages)`` pages is chosen once
        (using the seeded RNG).  Each reference is then drawn from the hot set
        with probability ``_HOT_PROBABILITY``, otherwise from the cold set
        (the remaining pages).  When no cold pages exist (``num_pages == 1``),
        all references fall back to the hot set.
        """
        hot_size = max(1, int(num_pages * _HOT_FRACTION))
        all_pages = list(range(num_pages))
        hot_set = rng.sample(all_pages, hot_size)
        cold_set = [p for p in all_pages if p not in hot_set]

        sequence = []
        for _ in range(length):
            # ── Hot reference ─────────────────────────────────────────────
            if not cold_set or rng.random() < _HOT_PROBABILITY:
                sequence.append(rng.choice(hot_set))
            else:
                # ── Cold reference ────────────────────────────────────────
                sequence.append(rng.choice(cold_set))
        return sequence

    @staticmethod
    def _generate_sequential(num_pages: int, length: int) -> list:
        """Cyclic sequential pattern: 0, 1, 2, ..., num_pages-1, 0, 1, ..."""
        return [i % num_pages for i in range(length)]

    # ── File I/O ──────────────────────────────────────────────────────────────

    @staticmethod
    def _save_csv(sequence: list, path: str) -> None:
        """Write *sequence* to *path* as a CSV file with header ``page_id``."""
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["page_id"])
            for page in sequence:
                writer.writerow([page])

    # ── Validation ────────────────────────────────────────────────────────────

    @staticmethod
    def _validate(num_pages: int, length: int, pattern: str) -> None:
        """Raise descriptive exceptions for invalid arguments.

        Raises
        ------
        ValueError
            ``num_pages < 1``, ``length < 1``, or unrecognised *pattern*.
        """
        if num_pages < 1:
            raise ValueError(
                f"num_pages must be >= 1, got {num_pages}"
            )
        if length < 1:
            raise ValueError(
                f"length must be >= 1, got {length}"
            )
        if pattern not in VALID_PATTERNS:
            raise ValueError(
                f"pattern must be one of {VALID_PATTERNS}, got {pattern!r}"
            )
