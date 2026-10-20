"""Tests for the anomaly detector serving helpers (no PyTorch required)."""

import os
import tempfile
import unittest

from ml.device import anomaly_service
from modules.device.fleet import create_default_fleet


class TestBuildWindow(unittest.TestCase):
    def setUp(self) -> None:
        self.disk = create_default_fleet()[0]

    def test_shape_and_sample_count(self) -> None:
        window, samples = anomaly_service.build_window(self.disk, None, 10, 0)
        self.assertEqual(window.shape, (10, 5))
        self.assertEqual(len(samples), 10)

    def test_normal_window_is_deterministic(self) -> None:
        a, _ = anomaly_service.build_window(self.disk, None, 10, 7)
        b, _ = anomaly_service.build_window(self.disk, None, 10, 7)
        self.assertTrue((a == b).all())

    def test_fault_changes_only_the_second_half(self) -> None:
        normal, _ = anomaly_service.build_window(self.disk, None, 10, 0)
        faulty, samples = anomaly_service.build_window(self.disk, "high_temperature", 10, 0)
        self.assertTrue((normal[:5] == faulty[:5]).all())
        self.assertGreater(faulty[9][0], normal[9][0] + 40)
        self.assertGreater(samples[9]["temperature"], 70)

    def test_every_pattern_is_supported(self) -> None:
        for pattern in anomaly_service.FAULT_PATTERNS:
            window, _ = anomaly_service.build_window(self.disk, pattern, 10, 0)
            self.assertEqual(window.shape, (10, 5))


class TestRiskMapping(unittest.TestCase):
    def test_threshold_maps_to_backup_threshold(self) -> None:
        self.assertAlmostEqual(anomaly_service.risk_from_score(2.0, 2.0), anomaly_service.BACKUP_THRESHOLD)

    def test_is_capped_and_monotonic(self) -> None:
        self.assertEqual(anomaly_service.risk_from_score(1000.0, 1.0), 1.0)
        self.assertEqual(anomaly_service.risk_from_score(1e9, 1.0), 1.0)
        mid = anomaly_service.risk_from_score(10.0, 1.0)
        self.assertTrue(anomaly_service.BACKUP_THRESHOLD < mid < 1.0)
        self.assertLess(anomaly_service.risk_from_score(10.0, 1.0), anomaly_service.risk_from_score(100.0, 1.0))
        self.assertLess(anomaly_service.risk_from_score(0.1, 1.0), anomaly_service.risk_from_score(0.5, 1.0))
        self.assertEqual(anomaly_service.risk_from_score(0.0, 1.0), 0.0)


class TestLoadBundle(unittest.TestCase):
    def test_missing_model_file(self) -> None:
        missing = os.path.join(tempfile.gettempdir(), "no_such_anomaly_model.pt")
        with self.assertRaises(anomaly_service.ModelUnavailableError):
            anomaly_service.load_bundle(missing)

    def test_corrupt_model_file(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            corrupt = os.path.join(folder, "corrupt.pt")
            with open(corrupt, "wb") as handle:
                handle.write(b"this is not a saved model")
            with self.assertRaises(anomaly_service.ModelUnavailableError):
                anomaly_service.load_bundle(corrupt)


if __name__ == "__main__":
    unittest.main()
