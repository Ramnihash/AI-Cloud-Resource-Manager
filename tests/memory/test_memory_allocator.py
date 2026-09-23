"""
Unit tests for modules.memory.memory_allocator
===============================================

Run from the repository root::

    .venv\\Scripts\\python.exe -m unittest tests.memory.test_memory_allocator -v

Test coverage:
    1.  First Fit selection logic
    2.  Best Fit selection logic
    3.  Worst Fit selection logic
    4.  Successful allocations
    5.  Failed allocations (no eligible block)
    6.  Exact-size allocation (remaining == 0)
    7.  Multiple sequential allocations (earlier allocations affect later ones)
    8.  Block-size updates after allocation (final_blocks correctness)
    9.  Invalid inputs (TypeError / ValueError)
    10. Consistent result structure across all three algorithms

Reference input used in multi-algorithm comparison tests
(Silberschatz-style textbook example):
    blocks   = [100, 500, 200, 300, 600]
    requests = [212, 417, 112, 426]
"""

import unittest
from modules.memory.memory_allocator import (
    FirstFitAllocator,
    BestFitAllocator,
    WorstFitAllocator,
)









SELECT_BLOCKS   = [50, 150, 100, 200]
SELECT_REQUEST  = 90


MULTI_BLOCKS    = [100, 500, 200, 300, 600]
MULTI_REQUESTS  = [212, 417, 112, 426]





def _all_allocators():
    return [FirstFitAllocator(), BestFitAllocator(), WorstFitAllocator()]





class TestFirstFitAllocator(unittest.TestCase):
    """Tests for FirstFitAllocator selection logic."""

    def setUp(self):
        self.algo = FirstFitAllocator()

    def test_selects_first_eligible_block(self):
        """First Fit picks the first block >= request, not the smallest."""

        result = self.algo.allocate(SELECT_BLOCKS, [SELECT_REQUEST])
        self.assertEqual(result["allocations"][0]["block_index"], 1)

    def test_skips_blocks_that_are_too_small(self):
        """First Fit skips blocks smaller than the request."""
        blocks = [50, 30, 200, 400]
        result = self.algo.allocate(blocks, [100])

        self.assertEqual(result["allocations"][0]["block_index"], 2)

    def test_single_request_success(self):
        """Single request that fits in the first block."""
        result = self.algo.allocate([500], [200])
        self.assertTrue(result["allocations"][0]["allocated"])
        self.assertEqual(result["allocations"][0]["block_index"], 0)

    def test_single_request_fail(self):
        """Single request larger than all blocks."""
        result = self.algo.allocate([100, 200], [500])
        self.assertFalse(result["allocations"][0]["allocated"])

    def test_sequential_requests_see_updated_blocks(self):
        """Second request sees reduced block size from first allocation."""


        result = self.algo.allocate([300, 400], [250, 100])
        allocs = result["allocations"]
        self.assertEqual(allocs[0]["block_index"], 0)
        self.assertEqual(allocs[1]["block_index"], 1)

    def test_multi_request_known_allocation_count(self):
        """Multi-request sequence: verify allocated vs failed counts."""
        result = self.algo.allocate(MULTI_BLOCKS, MULTI_REQUESTS)

        self.assertEqual(result["total_allocated"], 3)
        self.assertEqual(result["total_failed"], 1)





class TestBestFitAllocator(unittest.TestCase):
    """Tests for BestFitAllocator selection logic."""

    def setUp(self):
        self.algo = BestFitAllocator()

    def test_selects_smallest_eligible_block(self):
        """Best Fit picks the smallest block >= request."""

        result = self.algo.allocate(SELECT_BLOCKS, [SELECT_REQUEST])
        self.assertEqual(result["allocations"][0]["block_index"], 2)

    def test_minimises_remaining(self):
        """Best Fit leaves the smallest possible remaining space."""
        blocks = [300, 110, 500, 200]
        result = self.algo.allocate(blocks, [100])


        self.assertEqual(result["allocations"][0]["block_index"], 1)
        self.assertEqual(result["allocations"][0]["remaining"], 10)

    def test_tiebreak_picks_lower_index(self):
        """When two blocks have equal size, Best Fit picks the lower index."""
        blocks = [200, 100, 100]
        result = self.algo.allocate(blocks, [80])

        self.assertEqual(result["allocations"][0]["block_index"], 1)

    def test_single_request_success(self):
        """Single request succeeds and correct block is chosen."""
        result = self.algo.allocate([400, 150, 300], [140])

        self.assertEqual(result["allocations"][0]["block_index"], 1)

    def test_multi_request_all_allocated(self):
        """Multi-request sequence: Best Fit can allocate all 4 requests."""
        result = self.algo.allocate(MULTI_BLOCKS, MULTI_REQUESTS)

        self.assertEqual(result["total_allocated"], 4)
        self.assertEqual(result["total_failed"], 0)





class TestWorstFitAllocator(unittest.TestCase):
    """Tests for WorstFitAllocator selection logic."""

    def setUp(self):
        self.algo = WorstFitAllocator()

    def test_selects_largest_eligible_block(self):
        """Worst Fit picks the largest block >= request."""

        result = self.algo.allocate(SELECT_BLOCKS, [SELECT_REQUEST])
        self.assertEqual(result["allocations"][0]["block_index"], 3)

    def test_maximises_remaining(self):
        """Worst Fit leaves the largest possible remaining space."""
        blocks = [300, 110, 500, 200]
        result = self.algo.allocate(blocks, [100])

        self.assertEqual(result["allocations"][0]["block_index"], 2)
        self.assertEqual(result["allocations"][0]["remaining"], 400)

    def test_tiebreak_picks_lower_index(self):
        """When two blocks have equal size, Worst Fit picks the lower index."""
        blocks = [100, 200, 200]
        result = self.algo.allocate(blocks, [80])

        self.assertEqual(result["allocations"][0]["block_index"], 1)

    def test_single_request_success(self):
        """Single request succeeds and the largest block is chosen."""
        result = self.algo.allocate([400, 150, 300], [140])

        self.assertEqual(result["allocations"][0]["block_index"], 0)

    def test_multi_request_known_allocation_count(self):
        """Multi-request sequence: Worst Fit allocated vs failed count."""
        result = self.algo.allocate(MULTI_BLOCKS, MULTI_REQUESTS)

        self.assertEqual(result["total_allocated"], 3)
        self.assertEqual(result["total_failed"], 1)





class TestAllocationEdgeCases(unittest.TestCase):
    """Edge case tests shared across all three algorithms via subTest."""

    def _all_results(self, blocks, requests):
        return [a.allocate(blocks, requests) for a in _all_allocators()]

    def test_empty_blocks_all_fail(self):
        """With no blocks, every request must fail."""
        for result in self._all_results([], [100, 200]):
            with self.subTest(algorithm=result["algorithm"]):
                self.assertEqual(result["total_allocated"], 0)
                self.assertEqual(result["total_failed"], 2)
                for alloc in result["allocations"]:
                    self.assertFalse(alloc["allocated"])

    def test_empty_requests_no_allocations(self):
        """With no requests, allocations list is empty and counts are zero."""
        for result in self._all_results([100, 200], []):
            with self.subTest(algorithm=result["algorithm"]):
                self.assertEqual(result["allocations"], [])
                self.assertEqual(result["total_allocated"], 0)
                self.assertEqual(result["total_failed"], 0)

    def test_request_larger_than_all_blocks_fails(self):
        """A request bigger than every block must fail without error."""
        for result in self._all_results([50, 80, 30], [200]):
            with self.subTest(algorithm=result["algorithm"]):
                self.assertFalse(result["allocations"][0]["allocated"])
                self.assertIsNone(result["allocations"][0]["block_index"])

    def test_exact_size_allocation_succeeds(self):
        """A request exactly equal to a block size must succeed."""
        for result in self._all_results([100, 200, 300], [200]):
            with self.subTest(algorithm=result["algorithm"]):
                self.assertTrue(result["allocations"][0]["allocated"])

    def test_exact_size_remaining_is_zero(self):
        """After an exact-size allocation, remaining must be 0."""

        for algo in _all_allocators():
            result = algo.allocate([200], [200])
            with self.subTest(algorithm=result["algorithm"]):
                alloc = result["allocations"][0]
                self.assertTrue(alloc["allocated"])
                self.assertEqual(alloc["remaining"], 0)

    def test_zero_size_block_not_reused_after_exact_allocation(self):
        """A block reduced to 0 by exact allocation cannot serve future requests."""


        result = FirstFitAllocator().allocate([100, 200], [100, 50])
        allocs = result["allocations"]
        self.assertEqual(allocs[0]["block_index"], 0)
        self.assertEqual(allocs[0]["remaining"], 0)
        self.assertEqual(allocs[1]["block_index"], 1)

    def test_partial_allocation_some_succeed_some_fail(self):
        """When only some requests can be satisfied, both outcomes appear."""


        for result in self._all_results([200, 300], [80, 150, 500]):
            with self.subTest(algorithm=result["algorithm"]):
                allocs = result["allocations"]
                self.assertTrue(allocs[0]["allocated"])
                self.assertTrue(allocs[1]["allocated"])
                self.assertFalse(allocs[2]["allocated"])

    def test_multiple_requests_compete_for_blocks(self):
        """Sequential requests compete for the same pool of blocks."""

        for result in self._all_results([300, 300], [200, 200]):
            with self.subTest(algorithm=result["algorithm"]):
                self.assertEqual(result["total_allocated"], 2)
                self.assertEqual(result["total_failed"], 0)





class TestBlockUpdates(unittest.TestCase):
    """Verify that block sizes are correctly tracked throughout allocation."""

    def _run(self, algo, blocks, requests):
        return algo.allocate(blocks, requests)

    def test_initial_blocks_not_mutated(self):
        """allocate() must not modify the caller's original blocks list."""
        original = [100, 200, 300]
        blocks_copy = list(original)
        FirstFitAllocator().allocate(blocks_copy, [100])
        self.assertEqual(blocks_copy, original)

    def test_final_blocks_reflects_all_allocations(self):
        """final_blocks must match expected sizes after each allocation."""


        result = FirstFitAllocator().allocate([100, 500, 200], [80, 300])
        self.assertEqual(result["final_blocks"], [20, 200, 200])

    def test_remaining_matches_final_block_entry(self):
        """remaining in each allocation must equal the block's size in final_blocks."""
        result = BestFitAllocator().allocate([200, 500, 100], [90, 400])
        for alloc in result["allocations"]:
            if alloc["allocated"]:
                idx = alloc["block_index"]
                self.assertEqual(
                    alloc["remaining"],
                    result["final_blocks"][idx],
                    msg=f"remaining mismatch at block_index={idx}",
                )

    def test_unallocated_blocks_unchanged_in_final_blocks(self):
        """Blocks that were never chosen must have the same size in final_blocks."""

        result = FirstFitAllocator().allocate([100, 500, 200], [80])
        self.assertEqual(result["final_blocks"][1], 500)
        self.assertEqual(result["final_blocks"][2], 200)

    def test_block_size_before_allocation_recorded(self):
        """block_size in allocation dict must be the size BEFORE reducing."""
        result = WorstFitAllocator().allocate([100, 400], [150])
        alloc = result["allocations"][0]
        self.assertTrue(alloc["allocated"])
        self.assertEqual(alloc["block_size"], 400)
        self.assertEqual(alloc["remaining"], 250)





class TestInvalidInputs(unittest.TestCase):
    """TypeError and ValueError are raised for bad inputs."""

    def _all_algos(self):
        return _all_allocators()

    def test_blocks_not_list_raises_type_error(self):
        for algo in self._all_algos():
            with self.subTest(algorithm=algo.__class__.__name__):
                with self.assertRaises(TypeError):
                    algo.allocate((100, 200), [50])

    def test_requests_not_list_raises_type_error(self):
        for algo in self._all_algos():
            with self.subTest(algorithm=algo.__class__.__name__):
                with self.assertRaises(TypeError):
                    algo.allocate([100, 200], (50,))

    def test_zero_block_size_raises_value_error(self):
        for algo in self._all_algos():
            with self.subTest(algorithm=algo.__class__.__name__):
                with self.assertRaises(ValueError):
                    algo.allocate([100, 0, 200], [50])

    def test_negative_block_size_raises_value_error(self):
        for algo in self._all_algos():
            with self.subTest(algorithm=algo.__class__.__name__):
                with self.assertRaises(ValueError):
                    algo.allocate([100, -50, 200], [50])

    def test_zero_request_size_raises_value_error(self):
        for algo in self._all_algos():
            with self.subTest(algorithm=algo.__class__.__name__):
                with self.assertRaises(ValueError):
                    algo.allocate([100, 200], [50, 0])

    def test_negative_request_size_raises_value_error(self):
        for algo in self._all_algos():
            with self.subTest(algorithm=algo.__class__.__name__):
                with self.assertRaises(ValueError):
                    algo.allocate([100, 200], [-10])





class TestResultStructure(unittest.TestCase):
    """Verify result dict keys, types, and consistency for all algorithms."""

    REQUIRED_RESULT_KEYS = {
        "algorithm", "initial_blocks", "requests",
        "allocations", "total_allocated", "total_failed",
        "final_blocks", "execution_time_ms",
    }

    REQUIRED_ALLOC_KEYS = {
        "request_size", "allocated", "block_index", "block_size", "remaining",
    }

    def _all_results(self):
        blocks   = [100, 500, 200, 300]
        requests = [120, 350]
        return [a.allocate(blocks, requests) for a in _all_allocators()]

    def test_all_required_result_keys_present(self):
        """Every top-level result dict must contain all required keys."""
        for result in self._all_results():
            with self.subTest(algorithm=result["algorithm"]):
                missing = self.REQUIRED_RESULT_KEYS - result.keys()
                self.assertEqual(missing, set(), msg=f"Missing keys: {missing}")

    def test_all_required_allocation_keys_present(self):
        """Every per-request allocation dict must contain all required keys."""
        for result in self._all_results():
            for alloc in result["allocations"]:
                with self.subTest(algorithm=result["algorithm"],
                                  request=alloc["request_size"]):
                    missing = self.REQUIRED_ALLOC_KEYS - alloc.keys()
                    self.assertEqual(missing, set(), msg=f"Missing keys: {missing}")

    def test_total_counts_sum_to_request_count(self):
        """total_allocated + total_failed must equal len(requests)."""
        for result in self._all_results():
            with self.subTest(algorithm=result["algorithm"]):
                self.assertEqual(
                    result["total_allocated"] + result["total_failed"],
                    len(result["requests"]),
                )

    def test_allocations_length_equals_request_count(self):
        """len(allocations) must equal len(requests)."""
        for result in self._all_results():
            with self.subTest(algorithm=result["algorithm"]):
                self.assertEqual(
                    len(result["allocations"]), len(result["requests"])
                )

    def test_algorithm_names_are_correct(self):
        """Each algorithm result must carry its own name."""
        results = self._all_results()
        names = {r["algorithm"] for r in results}
        self.assertEqual(names, {"FirstFit", "BestFit", "WorstFit"})

    def test_initial_blocks_preserved_in_result(self):
        """initial_blocks in result must equal the input blocks list."""
        blocks = [100, 500, 200, 300]
        for algo in _all_allocators():
            result = algo.allocate(blocks, [120])
            with self.subTest(algorithm=result["algorithm"]):
                self.assertEqual(result["initial_blocks"], blocks)

    def test_execution_time_is_non_negative(self):
        """execution_time_ms must be >= 0."""
        for result in self._all_results():
            with self.subTest(algorithm=result["algorithm"]):
                self.assertGreaterEqual(result["execution_time_ms"], 0.0)

    def test_failed_allocation_fields_are_none(self):
        """For a failed allocation, block_index, block_size, remaining are None."""
        for algo in _all_allocators():
            result = algo.allocate([10], [999])
            with self.subTest(algorithm=result["algorithm"]):
                alloc = result["allocations"][0]
                self.assertFalse(alloc["allocated"])
                self.assertIsNone(alloc["block_index"])
                self.assertIsNone(alloc["block_size"])
                self.assertIsNone(alloc["remaining"])

    def test_successful_allocation_fields_are_not_none(self):
        """For a successful allocation, block_index, block_size, remaining are set."""
        for algo in _all_allocators():
            result = algo.allocate([500], [100])
            with self.subTest(algorithm=result["algorithm"]):
                alloc = result["allocations"][0]
                self.assertTrue(alloc["allocated"])
                self.assertIsNotNone(alloc["block_index"])
                self.assertIsNotNone(alloc["block_size"])
                self.assertIsNotNone(alloc["remaining"])


if __name__ == "__main__":
    unittest.main()
