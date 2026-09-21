"""Tests for CP4.5 predictive action orchestration layer."""

import unittest

from modules.device.predictive_action import (
    PredictiveActionResult,
    handle_predictive_action,
)


class TestPredictiveAction(unittest.TestCase):
    # 1. High risk + degrading + backup device
    def test_high_risk_degrading_with_backup(self):
        result = handle_predictive_action(
            device_id="disk_0",
            failure_probability=0.85,
            device_status="degrading",
            backup_device_id="disk_backup",
            threshold=0.7,
        )
        self.assertIsInstance(result, PredictiveActionResult)
        self.assertEqual(result.device_id, "disk_0")
        self.assertAlmostEqual(result.failure_probability, 0.85)
        # backup
        self.assertTrue(result.backup_result.triggered)
        self.assertTrue(result.backup_result.success)
        # recovery
        self.assertTrue(result.recovery_result.io_blocked)
        self.assertTrue(result.recovery_result.rerouted)
        self.assertEqual(result.recovery_result.backup_device_id, "disk_backup")

    # 2. High risk + failed + backup device
    def test_high_risk_failed_with_backup(self):
        result = handle_predictive_action(
            device_id="disk_1",
            failure_probability=0.95,
            device_status="failed",
            backup_device_id="disk_spare",
        )
        self.assertTrue(result.backup_result.triggered)
        self.assertTrue(result.recovery_result.recovered)
        self.assertTrue(result.recovery_result.rerouted)
        self.assertEqual(result.recovery_result.backup_device_id, "disk_spare")

    # 3. Low risk + healthy device – no actions taken
    def test_low_risk_healthy(self):
        result = handle_predictive_action(
            device_id="disk_2",
            failure_probability=0.30,
            device_status="healthy",
        )
        self.assertFalse(result.backup_result.triggered)
        self.assertFalse(result.recovery_result.recovered)
        self.assertFalse(result.recovery_result.io_blocked)
        self.assertFalse(result.recovery_result.rerouted)

    # 4. Probability exactly equal to threshold → backup triggered
    def test_probability_equal_to_threshold(self):
        result = handle_predictive_action(
            device_id="disk_3",
            failure_probability=0.7,
            device_status="healthy",
            threshold=0.7,
        )
        self.assertTrue(result.backup_result.triggered)

    # 5. Invalid failure probability
    def test_invalid_failure_probability(self):
        with self.assertRaises(ValueError):
            handle_predictive_action(
                device_id="disk_4",
                failure_probability=1.5,
                device_status="healthy",
            )

    # 6. Invalid device status
    def test_invalid_device_status(self):
        with self.assertRaises(ValueError):
            handle_predictive_action(
                device_id="disk_5",
                failure_probability=0.5,
                device_status="unknown_status",
            )

    # 7. Invalid threshold
    def test_invalid_threshold(self):
        with self.assertRaises(ValueError):
            handle_predictive_action(
                device_id="disk_6",
                failure_probability=0.5,
                device_status="healthy",
                threshold=1.5,
            )

    # 8. Empty / whitespace device_id
    def test_empty_device_id(self):
        with self.assertRaises(ValueError):
            handle_predictive_action(
                device_id="   ",
                failure_probability=0.5,
                device_status="healthy",
            )

    # 9. Empty backup_device_id when supplied
    def test_empty_backup_device_id(self):
        with self.assertRaises(ValueError):
            handle_predictive_action(
                device_id="disk_7",
                failure_probability=0.8,
                device_status="degrading",
                backup_device_id="   ",
            )


if __name__ == "__main__":
    unittest.main()
