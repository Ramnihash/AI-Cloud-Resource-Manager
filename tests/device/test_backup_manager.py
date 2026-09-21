"""Tests for the backup_manager orchestration layer (CP4.5)."""

import unittest

from modules.device.backup_manager import handle_failure_risk
from modules.device.backup import BackupResult


class TestBackupManager(unittest.TestCase):
    def test_below_threshold_not_triggered(self):
        result = handle_failure_risk("dev1", 0.5, threshold=0.7)
        self.assertIsInstance(result, BackupResult)
        self.assertFalse(result.triggered)
        self.assertFalse(result.success)
        self.assertIsNone(result.backup_location)

    def test_equal_threshold_triggered(self):
        result = handle_failure_risk("dev2", 0.7, threshold=0.7)
        self.assertTrue(result.triggered)
        self.assertTrue(result.success)
        self.assertEqual(result.backup_location, "cloud://backup/dev2")

    def test_above_threshold_triggered(self):
        result = handle_failure_risk("dev3", 0.9, threshold=0.7)
        self.assertTrue(result.triggered)
        self.assertTrue(result.success)
        self.assertEqual(result.backup_location, "cloud://backup/dev3")

    def test_invalid_probability_low(self):
        with self.assertRaises(ValueError):
            handle_failure_risk("dev4", -0.1, threshold=0.7)

    def test_invalid_probability_high(self):
        with self.assertRaises(ValueError):
            handle_failure_risk("dev5", 1.1, threshold=0.7)

    def test_invalid_threshold_low(self):
        with self.assertRaises(ValueError):
            handle_failure_risk("dev6", 0.5, threshold=-0.2)

    def test_invalid_threshold_high(self):
        with self.assertRaises(ValueError):
            handle_failure_risk("dev7", 0.5, threshold=1.5)

    def test_result_fields_preserved(self):
        result = handle_failure_risk("dev8", 0.8, threshold=0.7)
        self.assertEqual(result.device_id, "dev8")
        self.assertTrue(result.triggered)
        self.assertTrue(result.success)
        self.assertEqual(result.backup_location, "cloud://backup/dev8")
        self.assertIn("completed", result.message.lower())


if __name__ == "__main__":
    unittest.main()
