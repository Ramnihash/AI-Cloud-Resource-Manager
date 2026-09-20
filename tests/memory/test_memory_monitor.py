"""
Unit tests for modules.memory.memory_monitor
=============================================

Run from the repository root::

    .venv\\Scripts\\python.exe -m unittest tests.memory.test_memory_monitor -v

Testing strategy:
    Live memory values are non-deterministic.  Tests focus on:

    * Schema — all required keys present in every snapshot
    * Types  — each field has the correct Python type
    * Safe invariants — individual field constraints that hold on all OSes:
        ram_total_bytes > 0
        ram_available_bytes >= 0
        ram_used_bytes >= 0
        ram_free_bytes >= 0
        0 <= ram_percent <= 100
        swap_total_bytes, swap_used_bytes, swap_free_bytes >= 0
        0 <= swap_percent <= 100
    * Structural — collect() returns the right count, CSV row counts, etc.
    * Temporal ordering — timestamps in collect() are non-decreasing
    * Process metrics — rss/vms > 0 and memory_percent >= 0 for a live PID
    * CSV correctness — header matches CSV_COLUMNS, row count matches n

    NOTE: The relationship ram_used + ram_available <= ram_total is NOT
    asserted because psutil's definitions of 'used' and 'available' can
    include OS-managed memory (buffers, cache) differently across platforms.
"""

import csv
import datetime
import os
import tempfile
import unittest

import psutil

from modules.memory.memory_monitor import MemoryMonitor, CSV_COLUMNS


# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

# All 14 keys that every snapshot dict must contain
EXPECTED_SNAPSHOT_KEYS = set(CSV_COLUMNS)

# Use the current process as a safe target for per-process tests
CURRENT_PID = os.getpid()


def _make_mon():
    return MemoryMonitor()


# ---------------------------------------------------------------------------
# 1. take_snapshot — system-wide (pid=None)
# ---------------------------------------------------------------------------
class TestTakeSnapshot(unittest.TestCase):
    """Structural and invariant tests for take_snapshot() without a PID."""

    def setUp(self):
        self.mon = _make_mon()
        self.snap = self.mon.take_snapshot()

    def test_returns_dict(self):
        self.assertIsInstance(self.snap, dict)

    def test_all_required_keys_present(self):
        missing = EXPECTED_SNAPSHOT_KEYS - self.snap.keys()
        self.assertEqual(missing, set(), msg=f"Missing snapshot keys: {missing}")

    def test_no_extra_keys(self):
        extra = self.snap.keys() - EXPECTED_SNAPSHOT_KEYS
        self.assertEqual(extra, set(), msg=f"Unexpected snapshot keys: {extra}")

    def test_timestamp_is_string(self):
        self.assertIsInstance(self.snap["timestamp"], str)

    def test_timestamp_parseable_as_iso(self):
        """Timestamp must be a valid ISO-8601 datetime string."""
        try:
            datetime.datetime.fromisoformat(self.snap["timestamp"])
        except ValueError as exc:
            self.fail(f"timestamp is not valid ISO-8601: {exc}")

    # ── RAM invariants ────────────────────────────────────────────────────

    def test_ram_total_positive(self):
        self.assertGreater(self.snap["ram_total_bytes"], 0)

    def test_ram_available_non_negative(self):
        self.assertGreaterEqual(self.snap["ram_available_bytes"], 0)

    def test_ram_used_non_negative(self):
        self.assertGreaterEqual(self.snap["ram_used_bytes"], 0)

    def test_ram_free_non_negative(self):
        self.assertGreaterEqual(self.snap["ram_free_bytes"], 0)

    def test_ram_percent_range(self):
        self.assertGreaterEqual(self.snap["ram_percent"], 0.0)
        self.assertLessEqual(self.snap["ram_percent"], 100.0)

    def test_ram_percent_is_float(self):
        self.assertIsInstance(self.snap["ram_percent"], float)

    # ── Swap invariants ───────────────────────────────────────────────────

    def test_swap_total_non_negative(self):
        self.assertGreaterEqual(self.snap["swap_total_bytes"], 0)

    def test_swap_used_non_negative(self):
        self.assertGreaterEqual(self.snap["swap_used_bytes"], 0)

    def test_swap_free_non_negative(self):
        self.assertGreaterEqual(self.snap["swap_free_bytes"], 0)

    def test_swap_percent_range(self):
        self.assertGreaterEqual(self.snap["swap_percent"], 0.0)
        self.assertLessEqual(self.snap["swap_percent"], 100.0)

    def test_swap_percent_is_float(self):
        self.assertIsInstance(self.snap["swap_percent"], float)

    # ── Process fields absent when pid=None ───────────────────────────────

    def test_proc_pid_is_none(self):
        self.assertIsNone(self.snap["proc_pid"])

    def test_proc_rss_is_none(self):
        self.assertIsNone(self.snap["proc_rss_bytes"])

    def test_proc_vms_is_none(self):
        self.assertIsNone(self.snap["proc_vms_bytes"])

    def test_proc_memory_percent_is_none(self):
        self.assertIsNone(self.snap["proc_memory_percent"])


# ---------------------------------------------------------------------------
# 2. take_snapshot — with a known live PID
# ---------------------------------------------------------------------------
class TestTakeSnapshotWithProcess(unittest.TestCase):
    """Per-process field tests using the current process as a safe target."""

    def setUp(self):
        self.mon = _make_mon()
        self.snap = self.mon.take_snapshot(pid=CURRENT_PID)

    def test_proc_pid_matches_supplied(self):
        self.assertEqual(self.snap["proc_pid"], CURRENT_PID)

    def test_proc_rss_positive(self):
        """The current process must have at least some resident memory."""
        self.assertGreater(self.snap["proc_rss_bytes"], 0)

    def test_proc_vms_positive(self):
        """The current process must have at least some virtual memory."""
        self.assertGreater(self.snap["proc_vms_bytes"], 0)

    def test_proc_memory_percent_non_negative(self):
        self.assertGreaterEqual(self.snap["proc_memory_percent"], 0.0)

    def test_proc_memory_percent_at_most_100(self):
        self.assertLessEqual(self.snap["proc_memory_percent"], 100.0)

    def test_proc_memory_percent_is_float(self):
        self.assertIsInstance(self.snap["proc_memory_percent"], float)

    def test_invalid_pid_raises_no_such_process(self):
        """A non-existent PID must raise NoSuchProcess (or AccessDenied on Windows).

        PID 999999 is safely above Windows' maximum PID (65535 for most
        configurations) and is guaranteed to not correspond to a live process.
        psutil raises NoSuchProcess for genuinely absent PIDs; on Windows it
        may raise AccessDenied for certain reserved PID ranges, so both are
        accepted here.
        """
        with self.assertRaises((psutil.NoSuchProcess, psutil.AccessDenied)):
            self.mon.take_snapshot(pid=999999)


# ---------------------------------------------------------------------------
# 3. collect — structural tests
# ---------------------------------------------------------------------------
class TestCollect(unittest.TestCase):
    """Tests for collect() structure, count, and ordering."""

    def setUp(self):
        self.mon = _make_mon()

    def test_snapshot_count_matches_n(self):
        result = self.mon.collect(n=3, interval_s=0)
        self.assertEqual(len(result["snapshots"]), 3)

    def test_count_field_matches_n(self):
        result = self.mon.collect(n=4, interval_s=0)
        self.assertEqual(result["count"], 4)

    def test_interval_s_preserved_in_result(self):
        result = self.mon.collect(n=2, interval_s=0)
        self.assertEqual(result["interval_s"], 0.0)

    def test_interval_s_is_float_in_result(self):
        result = self.mon.collect(n=1, interval_s=0)
        self.assertIsInstance(result["interval_s"], float)

    def test_each_snapshot_has_required_keys(self):
        result = self.mon.collect(n=3, interval_s=0)
        for i, snap in enumerate(result["snapshots"]):
            with self.subTest(snapshot_index=i):
                missing = EXPECTED_SNAPSHOT_KEYS - snap.keys()
                self.assertEqual(missing, set())

    def test_timestamps_non_decreasing(self):
        """Timestamps across collected snapshots must be in non-decreasing order."""
        result = self.mon.collect(n=5, interval_s=0)
        timestamps = [
            datetime.datetime.fromisoformat(s["timestamp"])
            for s in result["snapshots"]
        ]
        for i in range(len(timestamps) - 1):
            self.assertLessEqual(
                timestamps[i], timestamps[i + 1],
                msg=f"Timestamp at index {i} is later than index {i + 1}",
            )

    def test_snapshots_is_list(self):
        result = self.mon.collect(n=2, interval_s=0)
        self.assertIsInstance(result["snapshots"], list)

    def test_collect_with_process_populates_proc_fields(self):
        """When pid is supplied, proc fields must be populated in every snapshot."""
        result = self.mon.collect(n=2, interval_s=0, pid=CURRENT_PID)
        for snap in result["snapshots"]:
            self.assertEqual(snap["proc_pid"], CURRENT_PID)
            self.assertIsNotNone(snap["proc_rss_bytes"])
            self.assertIsNotNone(snap["proc_vms_bytes"])
            self.assertIsNotNone(snap["proc_memory_percent"])

    def test_collect_n_equals_one(self):
        """collect() with n=1 must return exactly one snapshot."""
        result = self.mon.collect(n=1, interval_s=0)
        self.assertEqual(result["count"], 1)
        self.assertEqual(len(result["snapshots"]), 1)


# ---------------------------------------------------------------------------
# 4. collect — CSV save
# ---------------------------------------------------------------------------
class TestCollectSaveToFile(unittest.TestCase):
    """CSV output tests using a temporary file."""

    def setUp(self):
        self.mon = _make_mon()
        tmp_fd, self.tmp_path = tempfile.mkstemp(suffix=".csv")
        os.close(tmp_fd)
        os.unlink(self.tmp_path)  # let the monitor create it fresh

    def tearDown(self):
        if os.path.exists(self.tmp_path):
            os.unlink(self.tmp_path)

    def _collect_and_save(self, n=3):
        return self.mon.collect(n=n, interval_s=0, save_path=self.tmp_path)

    def test_file_is_created(self):
        self._collect_and_save()
        self.assertTrue(os.path.exists(self.tmp_path))

    def test_saved_to_field_matches_path(self):
        result = self._collect_and_save()
        self.assertEqual(result["saved_to"], self.tmp_path)

    def test_csv_header_matches_csv_columns(self):
        """CSV header row must contain exactly the columns in CSV_COLUMNS."""
        self._collect_and_save()
        with open(self.tmp_path, newline="") as f:
            reader = csv.reader(f)
            header = next(reader)
        self.assertEqual(set(header), set(CSV_COLUMNS))

    def test_csv_header_column_count(self):
        self._collect_and_save()
        with open(self.tmp_path, newline="") as f:
            reader = csv.reader(f)
            header = next(reader)
        self.assertEqual(len(header), len(CSV_COLUMNS))

    def test_csv_data_row_count_matches_n(self):
        n = 4
        self._collect_and_save(n=n)
        with open(self.tmp_path, newline="") as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            data_rows = list(reader)
        self.assertEqual(len(data_rows), n)

    def test_csv_timestamp_column_parseable(self):
        """Every timestamp value in the CSV must be a valid ISO-8601 string."""
        self._collect_and_save()
        with open(self.tmp_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    datetime.datetime.fromisoformat(row["timestamp"])
                except ValueError as exc:
                    self.fail(f"CSV timestamp not ISO-8601: {exc}")


# ---------------------------------------------------------------------------
# 5. collect — in-memory only (save_path=None)
# ---------------------------------------------------------------------------
class TestCollectInMemoryOnly(unittest.TestCase):
    """Tests confirming that no file is written when save_path is omitted."""

    def setUp(self):
        self.mon = _make_mon()

    def test_saved_to_is_none(self):
        result = self.mon.collect(n=2, interval_s=0)
        self.assertIsNone(result["saved_to"])

    def test_no_unexpected_file_created(self):
        before = set(os.listdir("."))
        self.mon.collect(n=2, interval_s=0)
        after = set(os.listdir("."))
        self.assertEqual(before, after, msg="Unexpected file(s) created in cwd")


# ---------------------------------------------------------------------------
# 6. collect — memory value invariants across multiple snapshots
# ---------------------------------------------------------------------------
class TestMemoryConstraints(unittest.TestCase):
    """Safe per-field invariants verified across a small collection."""

    def setUp(self):
        self.mon = _make_mon()
        self.result = self.mon.collect(n=3, interval_s=0)
        self.snapshots = self.result["snapshots"]

    def test_ram_total_positive_all_snapshots(self):
        for i, snap in enumerate(self.snapshots):
            with self.subTest(snapshot=i):
                self.assertGreater(snap["ram_total_bytes"], 0)

    def test_ram_fields_non_negative_all_snapshots(self):
        for i, snap in enumerate(self.snapshots):
            for field in ("ram_available_bytes", "ram_used_bytes", "ram_free_bytes"):
                with self.subTest(snapshot=i, field=field):
                    self.assertGreaterEqual(snap[field], 0)

    def test_ram_percent_range_all_snapshots(self):
        for i, snap in enumerate(self.snapshots):
            with self.subTest(snapshot=i):
                self.assertGreaterEqual(snap["ram_percent"], 0.0)
                self.assertLessEqual(snap["ram_percent"], 100.0)

    def test_swap_fields_non_negative_all_snapshots(self):
        for i, snap in enumerate(self.snapshots):
            for field in ("swap_total_bytes", "swap_used_bytes", "swap_free_bytes"):
                with self.subTest(snapshot=i, field=field):
                    self.assertGreaterEqual(snap[field], 0)

    def test_swap_percent_range_all_snapshots(self):
        for i, snap in enumerate(self.snapshots):
            with self.subTest(snapshot=i):
                self.assertGreaterEqual(snap["swap_percent"], 0.0)
                self.assertLessEqual(snap["swap_percent"], 100.0)


# ---------------------------------------------------------------------------
# 7. Invalid inputs
# ---------------------------------------------------------------------------
class TestInvalidInputs(unittest.TestCase):

    def setUp(self):
        self.mon = _make_mon()

    def test_n_zero_raises_value_error(self):
        with self.assertRaises(ValueError):
            self.mon.collect(n=0, interval_s=0)

    def test_n_negative_raises_value_error(self):
        with self.assertRaises(ValueError):
            self.mon.collect(n=-1, interval_s=0)

    def test_interval_negative_raises_value_error(self):
        with self.assertRaises(ValueError):
            self.mon.collect(n=1, interval_s=-0.1)


# ---------------------------------------------------------------------------
# 8. Result structure
# ---------------------------------------------------------------------------
class TestResultStructure(unittest.TestCase):
    """Verify the top-level collect() result dict structure."""

    REQUIRED_COLLECT_KEYS = {"snapshots", "count", "interval_s", "saved_to"}

    def setUp(self):
        self.mon = _make_mon()

    def test_collect_result_has_required_keys(self):
        result = self.mon.collect(n=2, interval_s=0)
        missing = self.REQUIRED_COLLECT_KEYS - result.keys()
        self.assertEqual(missing, set(), msg=f"Missing keys: {missing}")

    def test_collect_no_extra_keys(self):
        result = self.mon.collect(n=2, interval_s=0)
        extra = result.keys() - self.REQUIRED_COLLECT_KEYS
        self.assertEqual(extra, set(), msg=f"Unexpected keys: {extra}")

    def test_take_snapshot_result_is_plain_snapshot_dict(self):
        """take_snapshot() returns a snapshot dict, not a wrapper dict."""
        snap = self.mon.take_snapshot()
        # Should have snapshot keys, NOT collect() wrapper keys
        self.assertIn("timestamp", snap)
        self.assertIn("ram_total_bytes", snap)
        self.assertNotIn("snapshots", snap)
        self.assertNotIn("count", snap)

    def test_csv_columns_constant_has_correct_count(self):
        """CSV_COLUMNS must list exactly 14 fields (matches snapshot dict)."""
        self.assertEqual(len(CSV_COLUMNS), 14)

    def test_csv_columns_matches_snapshot_keys(self):
        """CSV_COLUMNS must match the keys of a real snapshot exactly."""
        snap = self.mon.take_snapshot()
        self.assertEqual(set(CSV_COLUMNS), set(snap.keys()))


if __name__ == "__main__":
    unittest.main()
