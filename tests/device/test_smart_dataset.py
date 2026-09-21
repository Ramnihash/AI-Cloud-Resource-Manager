"""Tests for CP3.8 SMART dataset utilities."""

from pathlib import Path
import tempfile
import unittest

import pandas as pd

from ml.device.smart.dataset import (
    load_prepared_dataset,
    split_features_target,
)


class TestSmartDataset(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_dir = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _write_dataset(self, dataframe: pd.DataFrame) -> Path:
        path = self.test_dir / "prepared_smart.csv"
        dataframe.to_csv(path, index=False)
        return path

    def test_load_prepared_dataset_success(self):
        dataframe = pd.DataFrame(
            {
                "smart_5_raw": [10.0, 20.0],
                "smart_9_raw": [100.0, 200.0],
                "failure": [0, 1],
            }
        )

        path = self._write_dataset(dataframe)

        result = load_prepared_dataset(path)

        pd.testing.assert_frame_equal(result, dataframe)

    def test_load_prepared_dataset_missing_file(self):
        path = self.test_dir / "missing.csv"

        with self.assertRaises(ValueError):
            load_prepared_dataset(path)

    def test_load_prepared_dataset_wrong_extension(self):
        path = self.test_dir / "prepared_smart.txt"
        path.write_text("smart_5_raw,failure\n10,0\n")

        with self.assertRaises(ValueError):
            load_prepared_dataset(path)

    def test_load_prepared_dataset_missing_target(self):
        dataframe = pd.DataFrame(
            {
                "smart_5_raw": [10.0, 20.0],
            }
        )

        path = self._write_dataset(dataframe)

        with self.assertRaises(ValueError):
            load_prepared_dataset(path)

    def test_load_prepared_dataset_missing_smart_features(self):
        dataframe = pd.DataFrame(
            {
                "temperature": [30.0, 31.0],
                "failure": [0, 1],
            }
        )

        path = self._write_dataset(dataframe)

        with self.assertRaises(ValueError):
            load_prepared_dataset(path)

    def test_split_features_and_target(self):
        dataframe = pd.DataFrame(
            {
                "smart_9_raw": [100.0, 200.0],
                "smart_5_raw": [10.0, 20.0],
                "failure": [0, 1],
            }
        )

        features, target = split_features_target(dataframe)

        self.assertEqual(
            list(features.columns),
            ["smart_5_raw", "smart_9_raw"],
        )
        self.assertEqual(
            list(target),
            [0, 1],
        )

    def test_split_ignores_non_smart_columns(self):
        dataframe = pd.DataFrame(
            {
                "date": ["2024-01-01", "2024-01-02"],
                "serial_number": ["A", "B"],
                "smart_5_raw": [10.0, 20.0],
                "smart_9_raw": [100.0, 200.0],
                "failure": [0, 1],
            }
        )

        features, target = split_features_target(dataframe)

        self.assertEqual(
            list(features.columns),
            ["smart_5_raw", "smart_9_raw"],
        )
        self.assertEqual(
            list(target),
            [0, 1],
        )

    def test_split_missing_target(self):
        dataframe = pd.DataFrame(
            {
                "smart_5_raw": [10.0, 20.0],
            }
        )

        with self.assertRaises(ValueError):
            split_features_target(dataframe)

    def test_split_missing_smart_features(self):
        dataframe = pd.DataFrame(
            {
                "temperature": [30.0, 31.0],
                "failure": [0, 1],
            }
        )

        with self.assertRaises(ValueError):
            split_features_target(dataframe)

    def test_split_empty_dataset(self):
        dataframe = pd.DataFrame(
            columns=["smart_5_raw", "failure"]
        )

        with self.assertRaises(ValueError):
            split_features_target(dataframe)


if __name__ == "__main__":
    unittest.main()