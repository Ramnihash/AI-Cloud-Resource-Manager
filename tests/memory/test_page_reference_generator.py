"""
Unit tests for ml.memory.page_reference_generator
==================================================

Run from the repository root::

    .venv\\Scripts\\python.exe -m unittest tests.memory.test_page_reference_generator -v

Test coverage:
    1.  Uniform pattern — length, page range, reproducibility
    2.  Locality pattern — length, page range, locality property
    3.  Sequential pattern — length, page range, cyclic order
    4.  Seed reproducibility (uniform and locality)
    5.  CSV save — file created, header, row count, valid pages, saved_to field
    6.  In-memory only (save_path=None) — no file, saved_to is None
    7.  Edge cases — length=1, num_pages=1, all three patterns
    8.  Invalid inputs — ValueError for bad num_pages, length, pattern
    9.  Result structure — required keys, types, consistency
"""

import csv
import os
import tempfile
import unittest

from ml.memory.page_reference_generator import (
    PageReferenceGenerator,
    VALID_PATTERNS,
    _HOT_FRACTION,
    _HOT_PROBABILITY,
)






def _make_gen():
    return PageReferenceGenerator()


def _all_results(num_pages=16, length=50, seed=0):
    """Return one result per pattern, using consistent parameters."""
    gen = _make_gen()
    return [
        gen.generate(num_pages=num_pages, length=length, pattern=p, seed=seed)
        for p in VALID_PATTERNS
    ]





class TestUniformPattern(unittest.TestCase):
    def setUp(self):
        self.gen = _make_gen()

    def test_length_correct(self):
        """Generated sequence must have exactly 'length' entries."""
        result = self.gen.generate(num_pages=10, length=100, pattern="uniform", seed=0)
        self.assertEqual(len(result["sequence"]), 100)

    def test_all_pages_in_range(self):
        """Every page reference must be in [0, num_pages-1]."""
        num_pages = 20
        result = self.gen.generate(num_pages=num_pages, length=500, pattern="uniform", seed=1)
        for page in result["sequence"]:
            self.assertGreaterEqual(page, 0)
            self.assertLessEqual(page, num_pages - 1)

    def test_seed_gives_same_sequence(self):
        """Same seed and parameters must produce an identical sequence."""
        r1 = self.gen.generate(num_pages=10, length=50, pattern="uniform", seed=42)
        r2 = self.gen.generate(num_pages=10, length=50, pattern="uniform", seed=42)
        self.assertEqual(r1["sequence"], r2["sequence"])

    def test_different_seeds_give_different_sequences(self):
        """Different seeds must produce different sequences (with high probability)."""
        r1 = self.gen.generate(num_pages=100, length=200, pattern="uniform", seed=0)
        r2 = self.gen.generate(num_pages=100, length=200, pattern="uniform", seed=999)
        self.assertNotEqual(r1["sequence"], r2["sequence"])





class TestLocalityPattern(unittest.TestCase):
    def setUp(self):
        self.gen = _make_gen()

    def test_length_correct(self):
        result = self.gen.generate(num_pages=16, length=200, pattern="locality", seed=0)
        self.assertEqual(len(result["sequence"]), 200)

    def test_all_pages_in_range(self):
        num_pages = 32
        result = self.gen.generate(num_pages=num_pages, length=500, pattern="locality", seed=7)
        for page in result["sequence"]:
            self.assertGreaterEqual(page, 0)
            self.assertLessEqual(page, num_pages - 1)

    def test_locality_property(self):
        """The hot set (top _HOT_FRACTION of pages by frequency) must account
        for substantially more than its proportional share of accesses."""
        num_pages = 20
        length = 1000
        result = self.gen.generate(
            num_pages=num_pages, length=length, pattern="locality", seed=42
        )

        counts = {}
        for p in result["sequence"]:
            counts[p] = counts.get(p, 0) + 1


        hot_size = max(1, int(num_pages * _HOT_FRACTION))
        top_counts = sorted(counts.values(), reverse=True)
        top_share = sum(top_counts[:hot_size]) / length



        self.assertGreater(
            top_share, 0.5,
            msg=(
                f"Top {hot_size} pages cover only {top_share:.1%} of accesses; "
                f"expected >> 25% due to locality (seed=42)"
            ),
        )

    def test_seed_gives_same_sequence(self):
        r1 = self.gen.generate(num_pages=16, length=100, pattern="locality", seed=5)
        r2 = self.gen.generate(num_pages=16, length=100, pattern="locality", seed=5)
        self.assertEqual(r1["sequence"], r2["sequence"])





class TestSequentialPattern(unittest.TestCase):
    def setUp(self):
        self.gen = _make_gen()

    def test_length_correct(self):
        result = self.gen.generate(num_pages=8, length=24, pattern="sequential")
        self.assertEqual(len(result["sequence"]), 24)

    def test_all_pages_in_range(self):
        num_pages = 5
        result = self.gen.generate(num_pages=num_pages, length=20, pattern="sequential")
        for page in result["sequence"]:
            self.assertGreaterEqual(page, 0)
            self.assertLessEqual(page, num_pages - 1)

    def test_exact_cyclic_order(self):
        """Sequential must produce exactly [0, 1, ..., N-1, 0, 1, ...], starting at 0."""
        num_pages = 4
        length = 10
        result = self.gen.generate(num_pages=num_pages, length=length, pattern="sequential")
        expected = [i % num_pages for i in range(length)]

        self.assertEqual(result["sequence"], expected)

    def test_sequential_is_deterministic_regardless_of_seed(self):
        """Sequential pattern is fully deterministic; seed has no effect."""
        r1 = self.gen.generate(num_pages=6, length=18, pattern="sequential", seed=0)
        r2 = self.gen.generate(num_pages=6, length=18, pattern="sequential", seed=999)
        self.assertEqual(r1["sequence"], r2["sequence"])





class TestSeedReproducibility(unittest.TestCase):
    def setUp(self):
        self.gen = _make_gen()

    def test_uniform_same_seed_identical(self):
        r1 = self.gen.generate(num_pages=32, length=200, pattern="uniform", seed=7)
        r2 = self.gen.generate(num_pages=32, length=200, pattern="uniform", seed=7)
        self.assertEqual(r1["sequence"], r2["sequence"])

    def test_locality_same_seed_identical(self):
        r1 = self.gen.generate(num_pages=32, length=200, pattern="locality", seed=7)
        r2 = self.gen.generate(num_pages=32, length=200, pattern="locality", seed=7)
        self.assertEqual(r1["sequence"], r2["sequence"])

    def test_global_random_state_not_affected(self):
        """Generator must not disturb the global random state."""
        import random as stdlib_random
        stdlib_random.seed(123)
        ref = stdlib_random.randint(0, 10_000)

        stdlib_random.seed(123)

        self.gen.generate(num_pages=50, length=500, pattern="uniform", seed=0)
        after = stdlib_random.randint(0, 10_000)

        self.assertEqual(
            ref, after,
            msg="Generator mutated global random state",
        )





class TestSaveToFile(unittest.TestCase):
    def setUp(self):
        self.gen = _make_gen()


        tmp_fd, self.tmp_path = tempfile.mkstemp(suffix=".csv")
        os.close(tmp_fd)
        os.unlink(self.tmp_path)

    def tearDown(self):
        if os.path.exists(self.tmp_path):
            os.unlink(self.tmp_path)

    def _generate_and_save(self, num_pages=10, length=30, pattern="uniform", seed=0):
        return self.gen.generate(
            num_pages=num_pages, length=length,
            pattern=pattern, seed=seed,
            save_path=self.tmp_path,
        )

    def test_file_is_created(self):
        """A CSV file must be created at save_path."""
        self._generate_and_save()
        self.assertTrue(os.path.exists(self.tmp_path))

    def test_csv_has_correct_header(self):
        """First row of the CSV must be ['page_id']."""
        self._generate_and_save()
        with open(self.tmp_path, newline="") as f:
            reader = csv.reader(f)
            header = next(reader)
        self.assertEqual(header, ["page_id"])

    def test_csv_row_count_matches_length(self):
        """Number of data rows must equal 'length'."""
        length = 50
        self._generate_and_save(length=length)
        with open(self.tmp_path, newline="") as f:
            reader = csv.reader(f)
            next(reader)
            data_rows = list(reader)
        self.assertEqual(len(data_rows), length)

    def test_csv_pages_are_valid(self):
        """Every value in the CSV must be an integer in [0, num_pages-1]."""
        num_pages = 12
        self._generate_and_save(num_pages=num_pages)
        with open(self.tmp_path, newline="") as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                page = int(row[0])
                self.assertGreaterEqual(page, 0)
                self.assertLessEqual(page, num_pages - 1)

    def test_saved_to_field_is_path(self):
        """result['saved_to'] must equal the save_path supplied."""
        result = self._generate_and_save()
        self.assertEqual(result["saved_to"], self.tmp_path)





class TestInMemoryOnly(unittest.TestCase):
    def setUp(self):
        self.gen = _make_gen()

    def test_saved_to_is_none(self):
        """When save_path is None, result['saved_to'] must be None."""
        result = self.gen.generate(num_pages=8, length=20, pattern="uniform")
        self.assertIsNone(result["saved_to"])

    def test_no_file_created_in_cwd(self):
        """When save_path is None, no file must be written."""
        before = set(os.listdir("."))
        self.gen.generate(num_pages=8, length=20, pattern="uniform")
        after = set(os.listdir("."))
        self.assertEqual(before, after, msg="Unexpected file(s) created in cwd")





class TestEdgeCases(unittest.TestCase):
    def setUp(self):
        self.gen = _make_gen()

    def test_length_one(self):
        """A sequence of length 1 must work for all patterns."""
        for p in VALID_PATTERNS:
            with self.subTest(pattern=p):
                result = self.gen.generate(num_pages=10, length=1, pattern=p, seed=0)
                self.assertEqual(len(result["sequence"]), 1)

    def test_num_pages_one(self):
        """With num_pages=1, every reference must be page 0."""
        for p in VALID_PATTERNS:
            with self.subTest(pattern=p):
                result = self.gen.generate(num_pages=1, length=20, pattern=p, seed=0)
                self.assertTrue(
                    all(page == 0 for page in result["sequence"]),
                    msg=f"Got non-zero page with num_pages=1, pattern={p!r}",
                )

    def test_large_sequence_completes(self):
        """A large sequence (length=10 000) must complete without error."""
        result = self.gen.generate(num_pages=64, length=10_000, pattern="locality", seed=0)
        self.assertEqual(len(result["sequence"]), 10_000)

    def test_all_patterns_produce_valid_results(self):
        """All three patterns must succeed and return correct length."""
        for p in VALID_PATTERNS:
            with self.subTest(pattern=p):
                result = self.gen.generate(num_pages=16, length=100, pattern=p, seed=1)
                self.assertEqual(len(result["sequence"]), 100)
                self.assertTrue(all(0 <= x <= 15 for x in result["sequence"]))





class TestInvalidInputs(unittest.TestCase):
    def setUp(self):
        self.gen = _make_gen()

    def test_zero_num_pages_raises_value_error(self):
        with self.assertRaises(ValueError):
            self.gen.generate(num_pages=0, length=10, pattern="uniform")

    def test_negative_num_pages_raises_value_error(self):
        with self.assertRaises(ValueError):
            self.gen.generate(num_pages=-5, length=10, pattern="uniform")

    def test_zero_length_raises_value_error(self):
        with self.assertRaises(ValueError):
            self.gen.generate(num_pages=10, length=0, pattern="uniform")

    def test_negative_length_raises_value_error(self):
        with self.assertRaises(ValueError):
            self.gen.generate(num_pages=10, length=-1, pattern="uniform")

    def test_unknown_pattern_raises_value_error(self):
        with self.assertRaises(ValueError):
            self.gen.generate(num_pages=10, length=10, pattern="markov")

    def test_empty_string_pattern_raises_value_error(self):
        with self.assertRaises(ValueError):
            self.gen.generate(num_pages=10, length=10, pattern="")






REQUIRED_KEYS = {"sequence", "num_pages", "length", "pattern", "seed", "saved_to"}


class TestResultStructure(unittest.TestCase):
    def setUp(self):
        self.gen = _make_gen()

    def test_all_required_keys_present(self):
        """Every result dict must contain all six required keys."""
        for result in _all_results():
            with self.subTest(pattern=result["pattern"]):
                missing = REQUIRED_KEYS - result.keys()
                self.assertEqual(missing, set(), msg=f"Missing keys: {missing}")

    def test_sequence_length_matches_length_field(self):
        """len(sequence) must equal result['length']."""
        for result in _all_results(length=75):
            with self.subTest(pattern=result["pattern"]):
                self.assertEqual(len(result["sequence"]), result["length"])

    def test_pattern_field_preserved(self):
        """result['pattern'] must equal the pattern argument passed in."""
        for p in VALID_PATTERNS:
            result = self.gen.generate(num_pages=8, length=10, pattern=p)
            with self.subTest(pattern=p):
                self.assertEqual(result["pattern"], p)

    def test_seed_field_preserved(self):
        """result['seed'] must equal the seed argument passed in."""
        result = self.gen.generate(num_pages=8, length=10, pattern="uniform", seed=77)
        self.assertEqual(result["seed"], 77)

    def test_seed_none_preserved(self):
        """result['seed'] must be None when no seed is supplied."""
        result = self.gen.generate(num_pages=8, length=10, pattern="uniform")
        self.assertIsNone(result["seed"])

    def test_num_pages_field_preserved(self):
        """result['num_pages'] must equal the num_pages argument."""
        result = self.gen.generate(num_pages=32, length=10, pattern="uniform", seed=0)
        self.assertEqual(result["num_pages"], 32)

    def test_sequence_contains_only_ints(self):
        """All elements of the sequence must be plain Python ints."""
        for result in _all_results():
            with self.subTest(pattern=result["pattern"]):
                for page in result["sequence"]:
                    self.assertIsInstance(page, int)


if __name__ == "__main__":
    unittest.main()
