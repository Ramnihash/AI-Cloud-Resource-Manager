"""Tests for CP3.7 telemetry sequence preparation."""

import unittest

import numpy as np
import pandas as pd

from ml.device.sequence import TELEMETRY_FEATURES, create_telemetry_sequences


def _make_dataframe(
    device_ids: list[str],
    n_rows_per_device: int,
    *,
    reverse_timestamps: bool = False,
) -> pd.DataFrame:
    """Build a minimal deterministic telemetry DataFrame for testing."""
    records = []
    base_values = {
        "temperature": 30.0,
        "error_rate": 0.01,
        "response_latency": 5.0,
        "queue_length": 2.0,
        "io_activity": 10.0,
    }
    for device_id in device_ids:
        timestamps = list(range(n_rows_per_device))
        if reverse_timestamps:
            timestamps = list(reversed(timestamps))
        for t in timestamps:
            row = {"device_id": device_id, "timestamp": t}
            for feat in TELEMETRY_FEATURES:

                row[feat] = base_values[feat] + float(t)
            records.append(row)
    return pd.DataFrame(records)


class TestFeatureOrdering(unittest.TestCase):
    """Feature column order must match TELEMETRY_FEATURES exactly."""

    def test_feature_order(self):
        df = _make_dataframe(["d1"], 5)
        result = create_telemetry_sequences(df, window_size=3)

        first_window_first_step = result[0, 0, :]
        expected = [
            30.0 + 0,
            0.01 + 0,
            5.0 + 0,
            2.0 + 0,
            10.0 + 0,
        ]
        np.testing.assert_allclose(first_window_first_step, expected)


class TestOutputShape(unittest.TestCase):
    """Output shape must be (n_windows, window_size, 5)."""

    def test_single_device_shape(self):

        df = _make_dataframe(["d1"], 10)
        result = create_telemetry_sequences(df, window_size=4)
        self.assertEqual(result.shape, (7, 4, 5))

    def test_exact_minimum_shape(self):

        df = _make_dataframe(["d1"], 2)
        result = create_telemetry_sequences(df, window_size=2)
        self.assertEqual(result.shape, (1, 2, 5))


class TestSlidingWindow(unittest.TestCase):
    """Consecutive windows must overlap correctly."""

    def test_sliding_overlap(self):
        df = _make_dataframe(["d1"], 5)
        result = create_telemetry_sequences(df, window_size=3)

        self.assertEqual(result.shape[0], 3)


        np.testing.assert_array_equal(result[0, 1:], result[1, :-1])


class TestMultipleDevices(unittest.TestCase):
    """Each device must contribute its own independent windows."""

    def test_two_devices_combined_windows(self):

        df = _make_dataframe(["d1", "d2"], 5)
        result = create_telemetry_sequences(df, window_size=3)
        self.assertEqual(result.shape, (6, 3, 5))

    def test_three_devices(self):
        df = _make_dataframe(["a", "b", "c"], 4)
        result = create_telemetry_sequences(df, window_size=2)

        self.assertEqual(result.shape, (9, 2, 5))


class TestTimestampSorting(unittest.TestCase):
    """Rows supplied out-of-order must be sorted before window creation."""

    def test_unsorted_timestamps_are_sorted(self):

        df_reversed = _make_dataframe(["d1"], 5, reverse_timestamps=True)
        df_sorted = _make_dataframe(["d1"], 5, reverse_timestamps=False)

        result_reversed = create_telemetry_sequences(df_reversed, window_size=3)
        result_sorted = create_telemetry_sequences(df_sorted, window_size=3)

        np.testing.assert_array_equal(result_reversed, result_sorted)


class TestNoCrossDeviceWindows(unittest.TestCase):
    """A window must never contain records from two different devices."""

    def test_no_cross_device_windows(self):

        records = []
        for t in range(3):
            for dev in ["a", "b"]:
                row = {"device_id": dev, "timestamp": t}
                for feat in TELEMETRY_FEATURES:
                    row[feat] = float(t) if dev == "a" else float(t + 100)
                records.append(row)
        df = pd.DataFrame(records)

        result = create_telemetry_sequences(df, window_size=3)


        self.assertEqual(result.shape, (2, 3, 5))



        self.assertTrue((result[0] < 100).all())
        self.assertTrue((result[1] >= 100).all())


class TestValidationErrors(unittest.TestCase):
    """Invalid inputs must raise ValueError (or TypeError for wrong types)."""

    def setUp(self):
        self.valid_df = _make_dataframe(["d1"], 5)

    def test_missing_column(self):
        df = self.valid_df.drop(columns=["temperature"])
        with self.assertRaises(ValueError):
            create_telemetry_sequences(df, window_size=3)

    def test_window_size_below_2(self):
        with self.assertRaises(ValueError):
            create_telemetry_sequences(self.valid_df, window_size=1)

    def test_window_size_zero(self):
        with self.assertRaises(ValueError):
            create_telemetry_sequences(self.valid_df, window_size=0)

    def test_window_size_negative(self):
        with self.assertRaises(ValueError):
            create_telemetry_sequences(self.valid_df, window_size=-3)

    def test_window_size_float(self):
        with self.assertRaises((ValueError, TypeError)):
            create_telemetry_sequences(self.valid_df, window_size=3.0)

    def test_window_size_string(self):
        with self.assertRaises((ValueError, TypeError)):
            create_telemetry_sequences(self.valid_df, window_size="3")

    def test_insufficient_records(self):

        df = _make_dataframe(["d1"], 2)
        with self.assertRaises(ValueError):
            create_telemetry_sequences(df, window_size=10)

    def test_empty_dataframe(self):
        df = pd.DataFrame(columns=["device_id", "timestamp"] + TELEMETRY_FEATURES)
        with self.assertRaises(ValueError):
            create_telemetry_sequences(df, window_size=3)


if __name__ == "__main__":
    unittest.main()
