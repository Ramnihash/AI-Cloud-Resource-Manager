"""Unit tests for the device telemetry dataset generator in data.device.generate_dataset."""

from pathlib import Path
import tempfile
import unittest

import pandas as pd

from data.device.generate_dataset import (
    DATASET_COLUMNS,
    generate_device_telemetry_dataset,
    save_device_telemetry_dataset,
)
from modules.device.abnormal import AbnormalPattern
from modules.device.fleet import create_default_fleet


class TestGenerateDataset(unittest.TestCase):
    """Test suite covering the simulated device telemetry dataset generator."""

    def test_default_generation_record_count(self):
        """1. Default generation produces exactly 1500 records."""
        df = generate_device_telemetry_dataset()
        self.assertEqual(len(df), 1500)

    def test_dataset_columns_and_order(self):
        """2. Dataset has exactly the expected 9 columns and correct order."""
        df = generate_device_telemetry_dataset(normal_samples=4, abnormal_samples_per_pattern=2)
        expected_columns = [
            "device_id",
            "timestamp",
            "temperature",
            "error_rate",
            "response_latency",
            "queue_length",
            "io_activity",
            "label",
            "pattern",
        ]
        self.assertEqual(list(df.columns), expected_columns)
        self.assertEqual(list(df.columns), DATASET_COLUMNS)

    def test_default_normal_records_label_and_pattern(self):
        """3. Default normal records are labeled 'normal' and pattern 'normal'."""
        df = generate_device_telemetry_dataset()
        normal_subset = df.iloc[:1000]

        self.assertEqual(len(normal_subset), 1000)
        self.assertTrue((normal_subset["label"] == "normal").all())
        self.assertTrue((normal_subset["pattern"] == "normal").all())
        self.assertEqual((df["label"] == "normal").sum(), 1000)

    def test_all_five_abnormal_patterns_represented(self):
        """4. All five abnormal patterns are represented."""
        df = generate_device_telemetry_dataset()
        abnormal_subset = df[df["label"] == "abnormal"]
        expected_patterns = {pattern.value for pattern in AbnormalPattern}

        self.assertEqual(set(abnormal_subset["pattern"].unique()), expected_patterns)
        self.assertEqual(len(expected_patterns), 5)

    def test_each_abnormal_pattern_record_count(self):
        """5. Each abnormal pattern has exactly 100 records with default configuration."""
        df = generate_device_telemetry_dataset()
        abnormal_subset = df[df["label"] == "abnormal"]

        for pattern in AbnormalPattern:
            with self.subTest(pattern=pattern.value):
                count = (abnormal_subset["pattern"] == pattern.value).sum()
                self.assertEqual(count, 100)

    def test_custom_sample_counts(self):
        """6. Custom normal_samples and abnormal_samples_per_pattern values work correctly."""
        normal_count = 25
        abnormal_per_pattern = 10
        total_expected = normal_count + (abnormal_per_pattern * len(AbnormalPattern))

        df = generate_device_telemetry_dataset(
            normal_samples=normal_count,
            abnormal_samples_per_pattern=abnormal_per_pattern,
        )

        self.assertEqual(len(df), total_expected)
        self.assertEqual((df["label"] == "normal").sum(), normal_count)
        self.assertEqual((df["label"] == "abnormal").sum(), abnormal_per_pattern * len(AbnormalPattern))

        for pattern in AbnormalPattern:
            with self.subTest(pattern=pattern.value):
                self.assertEqual((df["pattern"] == pattern.value).sum(), abnormal_per_pattern)

    def test_negative_normal_samples_raises_value_error(self):
        """7. Negative normal_samples raises ValueError."""
        with self.assertRaises(ValueError):
            generate_device_telemetry_dataset(normal_samples=-1)

    def test_negative_abnormal_samples_per_pattern_raises_value_error(self):
        """8. Negative abnormal_samples_per_pattern raises ValueError."""
        with self.assertRaises(ValueError):
            generate_device_telemetry_dataset(abnormal_samples_per_pattern=-5)

    def test_non_positive_timestamp_step_raises_value_error(self):
        """9. timestamp_step <= 0 raises ValueError."""
        with self.assertRaises(ValueError):
            generate_device_telemetry_dataset(timestamp_step=0.0)

        with self.assertRaises(ValueError):
            generate_device_telemetry_dataset(timestamp_step=-1.0)

    def test_generation_determinism(self):
        """10. Generation is deterministic for identical arguments."""
        df1 = generate_device_telemetry_dataset(
            normal_samples=20,
            abnormal_samples_per_pattern=5,
            timestamp_start=100.0,
            timestamp_step=0.5,
        )
        df2 = generate_device_telemetry_dataset(
            normal_samples=20,
            abnormal_samples_per_pattern=5,
            timestamp_start=100.0,
            timestamp_step=0.5,
        )

        pd.testing.assert_frame_equal(df1, df2)

    def test_save_device_telemetry_dataset(self):
        """11. save_device_telemetry_dataset() creates CSV and preserves expected schema and row count."""
        df = generate_device_telemetry_dataset(
            normal_samples=16,
            abnormal_samples_per_pattern=4,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "output" / "telemetry.csv"
            saved_file = save_device_telemetry_dataset(df, csv_path)

            self.assertTrue(saved_file.exists())
            loaded_df = pd.read_csv(saved_file)

            self.assertEqual(len(loaded_df), len(df))
            self.assertEqual(list(loaded_df.columns), list(df.columns))


            incomplete_df = df.drop(columns=["label"])
            with self.assertRaises(ValueError):
                save_device_telemetry_dataset(incomplete_df, Path(tmp_dir) / "invalid.csv")

    def test_device_objects_not_mutated(self):
        """12. The dataset generation process does not mutate the Device objects used by the fleet."""
        fleet_before = create_default_fleet()
        initial_states = [device.to_dict() for device in fleet_before]

        generate_device_telemetry_dataset(
            normal_samples=50,
            abnormal_samples_per_pattern=10,
        )

        fleet_after = create_default_fleet()
        after_states = [device.to_dict() for device in fleet_after]

        self.assertEqual(initial_states, after_states)


if __name__ == "__main__":
    unittest.main()
