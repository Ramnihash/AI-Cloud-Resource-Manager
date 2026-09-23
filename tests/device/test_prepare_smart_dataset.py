"""Unit tests for the SMART/Backblaze dataset preparation pipeline (CP2.5)."""

from pathlib import Path
import tempfile
import unittest

import pandas as pd

from data.device.smart.prepare_dataset import (
    prepare_smart_dataset,
    save_prepared_dataset,
)


class TestPrepareSmartDataset(unittest.TestCase):
    """Test suite covering SMART / Backblaze data preparation."""

    def setUp(self):
        """Create a temporary working directory for test fixtures."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_dir = Path(self.temp_dir.name)

    def tearDown(self):
        """Clean up the temporary directory."""
        self.temp_dir.cleanup()

    def _create_csv_fixture(self, data: dict, filename: str = "backblaze_sample.csv") -> Path:
        """Helper to write a dictionary of column data to a temporary CSV."""
        df = pd.DataFrame(data)
        csv_path = self.test_dir / filename
        df.to_csv(csv_path, index=False)
        return csv_path

    def test_valid_dataset_preparation(self):
        """1. Valid dataset preparation produces expected rows and columns."""
        fixture_data = {
            "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "serial_number": ["SN001", "SN002", "SN003"],
            "model": ["ST4000DM000", "ST4000DM000", "HGST4000"],
            "capacity_bytes": [4000000000, 4000000000, 4000000000],
            "smart_5_raw": [0, 5, 10],
            "smart_187_raw": [0, 0, 1],
            "failure": [0, 0, 1],
        }
        csv_file = self._create_csv_fixture(fixture_data)
        result = prepare_smart_dataset(csv_file)

        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 3)
        self.assertIn("smart_5_raw", result.columns)
        self.assertIn("smart_187_raw", result.columns)
        self.assertIn("failure", result.columns)

    def test_smart_columns_discovered(self):
        """2. SMART columns starting with 'smart_' are automatically discovered."""
        fixture_data = {
            "serial_number": ["A1", "A2"],
            "smart_9_raw": [100, 200],
            "smart_194_raw": [30, 32],
            "smart_197_raw": [0, 0],
            "failure": [0, 0],
        }
        csv_file = self._create_csv_fixture(fixture_data)
        result = prepare_smart_dataset(csv_file)

        expected_features = ["smart_194_raw", "smart_197_raw", "smart_9_raw"]
        for feature in expected_features:
            self.assertIn(feature, result.columns)

    def test_metadata_is_excluded(self):
        """3. Drive identifiers and metadata are excluded from the output."""
        fixture_data = {
            "date": ["2024-01-01", "2024-01-02"],
            "serial_number": ["DRIVE_1", "DRIVE_2"],
            "model": ["Model_X", "Model_Y"],
            "capacity_bytes": [1000000, 2000000],
            "smart_5_raw": [0, 2],
            "failure": [0, 1],
        }
        csv_file = self._create_csv_fixture(fixture_data)
        result = prepare_smart_dataset(csv_file)

        excluded = ["date", "serial_number", "model", "capacity_bytes"]
        for col in excluded:
            self.assertNotIn(col, result.columns)

    def test_numeric_conversion_handles_strings_and_corrupt_values(self):
        """4. Numeric conversion converts valid strings and coerces corrupt strings to NaN."""
        fixture_data = {
            "smart_5_raw": ["10", "20", "corrupt"],
            "smart_9_raw": [100.0, "200", None],
            "failure": [0, 0, 1],
        }
        csv_file = self._create_csv_fixture(fixture_data)
        result = prepare_smart_dataset(csv_file)

        self.assertTrue(pd.api.types.is_float_dtype(result["smart_5_raw"]))
        self.assertTrue(pd.api.types.is_float_dtype(result["smart_9_raw"]))

        self.assertEqual(result["smart_5_raw"].isna().sum(), 0)
        self.assertEqual(result["smart_9_raw"].isna().sum(), 0)

    def test_missing_smart_values_imputed_with_median(self):
        """5. Missing SMART values are imputed using the column's median."""
        fixture_data = {

            "smart_5_raw": [10.0, None, 30.0],
            "failure": [0, 0, 1],
        }
        csv_file = self._create_csv_fixture(fixture_data)
        result = prepare_smart_dataset(csv_file)

        imputed_value = result.loc[1, "smart_5_raw"]
        self.assertEqual(imputed_value, 20.0)

    def test_failure_column_retained_with_binary_integers(self):
        """6. Target column 'failure' is retained with integer 0/1 values."""
        fixture_data = {
            "smart_1_raw": [50, 60],
            "failure": [0, 1],
        }
        csv_file = self._create_csv_fixture(fixture_data)
        result = prepare_smart_dataset(csv_file)

        self.assertIn("failure", result.columns)
        self.assertEqual(list(result["failure"]), [0, 1])
        self.assertTrue(pd.api.types.is_integer_dtype(result["failure"]))

    def test_output_column_order_is_deterministic(self):
        """7. Output columns are deterministically sorted SMART features followed by failure."""
        fixture_data = {
            "smart_z_raw": [1, 2],
            "smart_a_raw": [3, 4],
            "smart_m_raw": [5, 6],
            "failure": [0, 0],
        }
        csv_file = self._create_csv_fixture(fixture_data)
        result = prepare_smart_dataset(csv_file)

        expected_order = ["smart_a_raw", "smart_m_raw", "smart_z_raw", "failure"]
        self.assertEqual(list(result.columns), expected_order)

    def test_missing_failure_column_raises_value_error(self):
        """8. Missing target column 'failure' raises ValueError."""
        fixture_data = {
            "smart_5_raw": [0, 1],
            "serial_number": ["A", "B"],
        }
        csv_file = self._create_csv_fixture(fixture_data)
        with self.assertRaises(ValueError) as ctx:
            prepare_smart_dataset(csv_file)
        self.assertIn("failure", str(ctx.exception).lower())

    def test_no_smart_columns_raises_value_error(self):
        """9. Absence of SMART feature columns raises ValueError."""
        fixture_data = {
            "serial_number": ["A", "B"],
            "failure": [0, 1],
        }
        csv_file = self._create_csv_fixture(fixture_data)
        with self.assertRaises(ValueError) as ctx:
            prepare_smart_dataset(csv_file)
        self.assertIn("smart", str(ctx.exception).lower())

    def test_non_binary_failure_labels_raise_value_error(self):
        """10. Non-binary values in 'failure' column raise ValueError."""

        fixture_data_invalid_num = {
            "smart_5_raw": [0, 1],
            "failure": [0, 2],
        }
        csv_file1 = self._create_csv_fixture(fixture_data_invalid_num, "bad_num.csv")
        with self.assertRaises(ValueError):
            prepare_smart_dataset(csv_file1)


        fixture_data_string = {
            "smart_5_raw": [0, 1],
            "failure": ["healthy", "failed"],
        }
        csv_file2 = self._create_csv_fixture(fixture_data_string, "bad_str.csv")
        with self.assertRaises(ValueError):
            prepare_smart_dataset(csv_file2)

    def test_save_prepared_dataset_creates_file(self):
        """11. save_prepared_dataset writes CSV file with index=False and creates directories."""
        df = pd.DataFrame({
            "smart_5_raw": [0.0, 1.0],
            "failure": [0, 1],
        })
        output_csv = self.test_dir / "nested" / "output_smart.csv"
        saved_path = save_prepared_dataset(df, output_csv)

        self.assertTrue(saved_path.exists())
        loaded = pd.read_csv(saved_path)
        self.assertEqual(len(loaded), 2)
        self.assertEqual(list(loaded.columns), ["smart_5_raw", "failure"])

    def test_invalid_input_paths_raise_value_error(self):
        """12. Non-existent file and non-CSV extensions raise ValueError."""
        non_existent = self.test_dir / "does_not_exist.csv"
        with self.assertRaises(ValueError):
            prepare_smart_dataset(non_existent)

        txt_file = self.test_dir / "sample.txt"
        txt_file.write_text("hello", encoding="utf-8")
        with self.assertRaises(ValueError):
            prepare_smart_dataset(txt_file)


if __name__ == "__main__":
    unittest.main()
