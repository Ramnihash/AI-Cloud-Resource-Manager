"""Tests for CP4.5 simulated cloud backup action."""

import unittest

from modules.device.backup import (
    BackupResult,
    create_backup,
    should_backup,
)


class TestBackupLogic(unittest.TestCase):
    def test_should_backup_below_threshold(self):
        self.assertFalse(should_backup(0.5, threshold=0.7))

    def test_should_backup_equal_threshold(self):
        self.assertTrue(should_backup(0.7, threshold=0.7))

    def test_should_backup_above_threshold(self):
        self.assertTrue(should_backup(0.9, threshold=0.7))

    def test_invalid_failure_probability_low(self):
        with self.assertRaises(ValueError):
            should_backup(-0.1, threshold=0.7)

    def test_invalid_failure_probability_high(self):
        with self.assertRaises(ValueError):
            should_backup(1.1, threshold=0.7)

    def test_invalid_threshold_low(self):
        with self.assertRaises(ValueError):
            should_backup(0.5, threshold=-0.2)

    def test_invalid_threshold_high(self):
        with self.assertRaises(ValueError):
            should_backup(0.5, threshold=1.5)

    def test_create_backup_not_triggered(self):
        result = create_backup("dev123", 0.4, threshold=0.7)
        self.assertIsInstance(result, BackupResult)
        self.assertFalse(result.triggered)
        self.assertFalse(result.success)
        self.assertIsNone(result.backup_location)
        self.assertIn("not triggered", result.message.lower())

    def test_create_backup_triggered(self):
        device_id = "dev456"
        result = create_backup(device_id, 0.8, threshold=0.7)
        self.assertIsInstance(result, BackupResult)
        self.assertTrue(result.triggered)
        self.assertTrue(result.success)
        self.assertEqual(result.device_id, device_id)
        self.assertEqual(result.backup_location, f"cloud://backup/{device_id}")
        self.assertIn("completed", result.message.lower())


if __name__ == "__main__":
    unittest.main()
