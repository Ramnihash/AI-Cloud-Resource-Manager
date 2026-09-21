"""Tests for CP4.5 device recovery simulation (backup device rerouting)."""

import unittest

from modules.device.recovery import RecoveryResult, recover_device


class TestDeviceRecovery(unittest.TestCase):
    # Healthy statuses – no recovery needed
    def test_healthy_device(self):
        result = recover_device("dev1", "healthy")
        self.assertIsInstance(result, RecoveryResult)
        self.assertFalse(result.recovered)
        self.assertFalse(result.io_blocked)
        self.assertFalse(result.rerouted)
        self.assertIsNone(result.backup_device_id)
        self.assertIn("healthy", result.message.lower())

    def test_available_device(self):
        result = recover_device("dev2", "available")
        self.assertFalse(result.recovered)
        self.assertFalse(result.io_blocked)
        self.assertFalse(result.rerouted)
        self.assertIsNone(result.backup_device_id)
        self.assertIn("healthy", result.message.lower())

    def test_normal_device(self):
        result = recover_device("dev3", "normal")
        self.assertFalse(result.recovered)
        self.assertFalse(result.io_blocked)
        self.assertFalse(result.rerouted)
        self.assertIsNone(result.backup_device_id)
        self.assertIn("healthy", result.message.lower())

    # Unhealthy with backup – should block and reroute
    def test_unhealthy_with_backup(self):
        result = recover_device("dev4", "unhealthy", backup_device_id="bk1")
        self.assertTrue(result.recovered)
        self.assertTrue(result.io_blocked)
        self.assertTrue(result.rerouted)
        self.assertEqual(result.backup_device_id, "bk1")
        self.assertIn("rerouted", result.message.lower())

    def test_degrading_with_backup(self):
        result = recover_device("dev5", "degrading", backup_device_id="bk2")
        self.assertTrue(result.recovered)
        self.assertTrue(result.io_blocked)
        self.assertTrue(result.rerouted)
        self.assertEqual(result.backup_device_id, "bk2")
        self.assertIn("rerouted", result.message.lower())

    def test_failed_with_backup(self):
        result = recover_device("dev6", "failed", backup_device_id="bk3")
        self.assertTrue(result.recovered)
        self.assertTrue(result.io_blocked)
        self.assertTrue(result.rerouted)
        self.assertEqual(result.backup_device_id, "bk3")
        self.assertIn("rerouted", result.message.lower())

    # Unhealthy without backup – block only
    def test_unhealthy_without_backup(self):
        result = recover_device("dev7", "unhealthy")
        self.assertFalse(result.recovered)
        self.assertTrue(result.io_blocked)
        self.assertFalse(result.rerouted)
        self.assertIsNone(result.backup_device_id)
        self.assertIn("no backup", result.message.lower())

    # Validation errors
    def test_invalid_status(self):
        with self.assertRaises(ValueError):
            recover_device("dev8", "unknown_status")

    def test_empty_device_id(self):
        with self.assertRaises(ValueError):
            recover_device("   ", "healthy")

    def test_empty_device_status(self):
        with self.assertRaises(ValueError):
            recover_device("dev9", "   ")

    def test_empty_backup_id_when_provided(self):
        with self.assertRaises(ValueError):
            recover_device("dev10", "unhealthy", backup_device_id="   ")


if __name__ == "__main__":
    unittest.main()
