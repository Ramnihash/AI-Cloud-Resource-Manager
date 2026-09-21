"""
Unit tests for modules.memory.page_replacement
===============================================

Run from the repository root::

    .venv\\Scripts\\python.exe -m unittest tests.memory.test_page_replacement -v

Reference sequence used as ground truth (verified by tracing each algorithm):

    reference_string = [1, 2, 3, 4, 1, 2, 5, 1, 2, 3, 4, 5]
    num_frames       = 3

    Verified page faults (computed, not assumed from a textbook summary):
        FIFO    → 9
        LRU     → 10
        Optimal → 7

Note: The 9/8/6 values cited in some textbooks correspond to a DIFFERENT
reference string (e.g. Silberschatz uses [7,0,1,2,0,3,0,4,2,3,0,3,2,1,2,0,1,7,0,1]).
The values here are verified by step-by-step algorithm trace.
"""

import unittest
from modules.memory.page_replacement import (
    FIFOPageReplacement,
    LRUPageReplacement,
    OptimalPageReplacement,
)

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------
CLASSIC_REF = [1, 2, 3, 4, 1, 2, 5, 1, 2, 3, 4, 5]
CLASSIC_FRAMES = 3


# ---------------------------------------------------------------------------
# FIFO tests
# ---------------------------------------------------------------------------
class TestFIFOPageReplacement(unittest.TestCase):
    """Tests for FIFOPageReplacement."""

    def setUp(self):
        self.algo = FIFOPageReplacement(num_frames=CLASSIC_FRAMES)

    # ── Ground-truth fault count ─────────────────────────────────────────
    def test_fifo_known_sequence(self):
        """Classic textbook sequence must produce exactly 9 page faults."""
        result = self.algo.run(CLASSIC_REF)
        self.assertEqual(result["page_faults"], 9)

    # ── Boundary: all pages fit in frames ────────────────────────────────
    def test_fifo_no_faults_when_all_fit(self):
        """All distinct pages fit in frames → only 4 cold-start faults."""
        algo = FIFOPageReplacement(num_frames=4)
        result = algo.run([1, 2, 3, 4, 1, 2, 3, 4])
        self.assertEqual(result["page_faults"], 4)
        self.assertEqual(result["page_hits"], 4)

    # ── Boundary: single frame ───────────────────────────────────────────
    def test_fifo_single_frame(self):
        """1 frame: every unique page access is a fault."""
        algo = FIFOPageReplacement(num_frames=1)
        result = algo.run([1, 2, 3, 1, 2])
        self.assertEqual(result["page_faults"], 5)
        self.assertEqual(result["page_hits"], 0)

    # ── Repeated identical page ──────────────────────────────────────────
    def test_fifo_repeated_same_page(self):
        """Accessing the same page repeatedly → exactly 1 fault."""
        result = FIFOPageReplacement(num_frames=3).run([7, 7, 7, 7, 7])
        self.assertEqual(result["page_faults"], 1)
        self.assertEqual(result["page_hits"], 4)

    # ── Metrics consistency ──────────────────────────────────────────────
    def test_fifo_metrics_sum(self):
        """hits + faults must equal len(reference_string)."""
        result = self.algo.run(CLASSIC_REF)
        self.assertEqual(
            result["page_hits"] + result["page_faults"], len(CLASSIC_REF)
        )

    def test_fifo_hit_ratio_range(self):
        """hit_ratio must lie in [0.0, 1.0]."""
        result = self.algo.run(CLASSIC_REF)
        self.assertGreaterEqual(result["hit_ratio"], 0.0)
        self.assertLessEqual(result["hit_ratio"], 1.0)

    # ── Snapshot correctness ─────────────────────────────────────────────
    def test_fifo_frame_snapshots_length(self):
        """frame_snapshots must have exactly one entry per reference."""
        result = self.algo.run(CLASSIC_REF)
        self.assertEqual(len(result["frame_snapshots"]), len(CLASSIC_REF))

    def test_fifo_frame_snapshots_max_size(self):
        """No snapshot must contain more pages than num_frames."""
        result = self.algo.run(CLASSIC_REF)
        for snapshot in result["frame_snapshots"]:
            self.assertLessEqual(len(snapshot), CLASSIC_FRAMES)


# ---------------------------------------------------------------------------
# LRU tests
# ---------------------------------------------------------------------------
class TestLRUPageReplacement(unittest.TestCase):
    """Tests for LRUPageReplacement."""

    def setUp(self):
        self.algo = LRUPageReplacement(num_frames=CLASSIC_FRAMES)

    # ── Ground-truth fault count ─────────────────────────────────────────
    def test_lru_known_sequence(self):
        """Verified sequence [1,2,3,4,1,2,5,1,2,3,4,5] with 3 frames → 10 faults."""
        result = self.algo.run(CLASSIC_REF)
        self.assertEqual(result["page_faults"], 10)

    # ── Boundary: single frame ───────────────────────────────────────────
    def test_lru_single_frame(self):
        """1 frame: every unique page access is a fault."""
        algo = LRUPageReplacement(num_frames=1)
        result = algo.run([1, 2, 3, 1, 2])
        self.assertEqual(result["page_faults"], 5)
        self.assertEqual(result["page_hits"], 0)

    # ── Repeated identical page ──────────────────────────────────────────
    def test_lru_repeated_same_page(self):
        """Accessing the same page repeatedly → exactly 1 fault."""
        result = LRUPageReplacement(num_frames=3).run([5, 5, 5, 5, 5])
        self.assertEqual(result["page_faults"], 1)
        self.assertEqual(result["page_hits"], 4)

    # ── Metrics consistency ──────────────────────────────────────────────
    def test_lru_hit_ratio_range(self):
        """hit_ratio must lie in [0.0, 1.0]."""
        result = self.algo.run(CLASSIC_REF)
        self.assertGreaterEqual(result["hit_ratio"], 0.0)
        self.assertLessEqual(result["hit_ratio"], 1.0)

    def test_lru_metrics_sum(self):
        """hits + faults must equal len(reference_string)."""
        result = self.algo.run(CLASSIC_REF)
        self.assertEqual(
            result["page_hits"] + result["page_faults"], len(CLASSIC_REF)
        )

    # ── Snapshot correctness ─────────────────────────────────────────────
    def test_lru_frame_snapshots_length(self):
        """frame_snapshots must have exactly one entry per reference."""
        result = self.algo.run(CLASSIC_REF)
        self.assertEqual(len(result["frame_snapshots"]), len(CLASSIC_REF))

    def test_lru_frame_snapshots_max_size(self):
        """No snapshot must contain more pages than num_frames."""
        result = self.algo.run(CLASSIC_REF)
        for snapshot in result["frame_snapshots"]:
            self.assertLessEqual(len(snapshot), CLASSIC_FRAMES)


# ---------------------------------------------------------------------------
# Optimal tests
# ---------------------------------------------------------------------------
class TestOptimalPageReplacement(unittest.TestCase):
    """Tests for OptimalPageReplacement."""

    def setUp(self):
        self.algo = OptimalPageReplacement(num_frames=CLASSIC_FRAMES)

    # ── Ground-truth fault count ─────────────────────────────────────────
    def test_optimal_known_sequence(self):
        """Verified sequence [1,2,3,4,1,2,5,1,2,3,4,5] with 3 frames → 7 faults."""
        result = self.algo.run(CLASSIC_REF)
        self.assertEqual(result["page_faults"], 7)

    # ── Lower-bound property ─────────────────────────────────────────────
    def test_optimal_lower_bound_vs_fifo(self):
        """Optimal faults must be <= FIFO faults on the same input."""
        fifo_faults = FIFOPageReplacement(CLASSIC_FRAMES).run(CLASSIC_REF)["page_faults"]
        opt_faults = self.algo.run(CLASSIC_REF)["page_faults"]
        self.assertLessEqual(opt_faults, fifo_faults)

    def test_optimal_lower_bound_vs_lru(self):
        """Optimal faults must be <= LRU faults on the same input."""
        lru_faults = LRUPageReplacement(CLASSIC_FRAMES).run(CLASSIC_REF)["page_faults"]
        opt_faults = self.algo.run(CLASSIC_REF)["page_faults"]
        self.assertLessEqual(opt_faults, lru_faults)

    # ── Boundary: single frame ───────────────────────────────────────────
    def test_optimal_single_frame(self):
        """1 frame: every unique page access is a fault."""
        algo = OptimalPageReplacement(num_frames=1)
        result = algo.run([1, 2, 3, 1, 2])
        self.assertEqual(result["page_faults"], 5)
        self.assertEqual(result["page_hits"], 0)

    # ── Repeated identical page ──────────────────────────────────────────
    def test_optimal_repeated_same_page(self):
        """Accessing the same page repeatedly → exactly 1 fault."""
        result = OptimalPageReplacement(num_frames=3).run([3, 3, 3, 3, 3])
        self.assertEqual(result["page_faults"], 1)
        self.assertEqual(result["page_hits"], 4)

    # ── Metrics consistency ──────────────────────────────────────────────
    def test_optimal_metrics_sum(self):
        """hits + faults must equal len(reference_string)."""
        result = self.algo.run(CLASSIC_REF)
        self.assertEqual(
            result["page_hits"] + result["page_faults"], len(CLASSIC_REF)
        )

    # ── Snapshot correctness ─────────────────────────────────────────────
    def test_optimal_frame_snapshots_length(self):
        """frame_snapshots must have exactly one entry per reference."""
        result = self.algo.run(CLASSIC_REF)
        self.assertEqual(len(result["frame_snapshots"]), len(CLASSIC_REF))

    def test_optimal_frame_snapshots_max_size(self):
        """No snapshot must contain more pages than num_frames."""
        result = self.algo.run(CLASSIC_REF)
        for snapshot in result["frame_snapshots"]:
            self.assertLessEqual(len(snapshot), CLASSIC_FRAMES)


# ---------------------------------------------------------------------------
# Cross-algorithm structural tests
# ---------------------------------------------------------------------------
class TestResultStructure(unittest.TestCase):
    """Cross-algorithm tests for result dict completeness and consistency."""

    REQUIRED_KEYS = {
        "algorithm",
        "num_frames",
        "reference_string",
        "page_faults",
        "page_hits",
        "hit_ratio",
        "miss_ratio",
        "execution_time_ms",
        "frame_snapshots",
    }

    def _all_results(self):
        """Return one result dict from each algorithm."""
        ref, frames = CLASSIC_REF, CLASSIC_FRAMES
        return [
            FIFOPageReplacement(frames).run(ref),
            LRUPageReplacement(frames).run(ref),
            OptimalPageReplacement(frames).run(ref),
        ]

    def test_all_required_keys_present(self):
        """Every result dict must contain all required keys."""
        for result in self._all_results():
            with self.subTest(algorithm=result.get("algorithm")):
                self.assertTrue(
                    self.REQUIRED_KEYS.issubset(result.keys()),
                    msg=f"Missing keys: {self.REQUIRED_KEYS - result.keys()}",
                )

    def test_hit_and_miss_ratios_sum_to_one(self):
        """hit_ratio + miss_ratio must equal 1.0 for every algorithm."""
        for result in self._all_results():
            with self.subTest(algorithm=result.get("algorithm")):
                self.assertAlmostEqual(
                    result["hit_ratio"] + result["miss_ratio"],
                    1.0,
                    places=9,
                )

    def test_execution_time_is_non_negative(self):
        """execution_time_ms must be >= 0 for every algorithm."""
        for result in self._all_results():
            with self.subTest(algorithm=result.get("algorithm")):
                self.assertGreaterEqual(result["execution_time_ms"], 0.0)

    def test_algorithm_name_in_result(self):
        """The 'algorithm' field must match the class name prefix."""
        expected = {"FIFO", "LRU", "Optimal"}
        names = {r["algorithm"] for r in self._all_results()}
        self.assertEqual(names, expected)

    def test_invalid_num_frames_raises_value_error(self):
        """Constructing any algorithm with num_frames < 1 must raise ValueError."""
        for AlgoCls in (
            FIFOPageReplacement,
            LRUPageReplacement,
            OptimalPageReplacement,
        ):
            with self.subTest(algo=AlgoCls.__name__):
                with self.assertRaises(ValueError):
                    AlgoCls(num_frames=0)

    def test_reference_string_preserved_in_result(self):
        """The result must contain a copy of the original reference string."""
        for result in self._all_results():
            with self.subTest(algorithm=result.get("algorithm")):
                self.assertEqual(result["reference_string"], CLASSIC_REF)


if __name__ == "__main__":
    unittest.main()
