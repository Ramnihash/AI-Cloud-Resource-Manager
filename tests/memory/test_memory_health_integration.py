
import unittest

from ml.memory.memory_pressure_forecaster import MemoryPressureForecaster
from ml.memory.memory_leak_detector import MemoryLeakDetector
from modules.memory.memory_health_action import MemoryHealthActionLayer


class TestMemoryHealthIntegration(unittest.TestCase):

    def setUp(self):
        self.forecaster = MemoryPressureForecaster(window_size=3)
        self.detector = MemoryLeakDetector(
            window_size=5,
            min_growth_bytes_per_step=1024.0,
            min_r2=0.75,
        )
        self.action_layer = MemoryHealthActionLayer(
            pressure_threshold=85.0
        )

    def snapshots(self, ram_values, rss_values, pid=1234):
        return [
            {
                "ram_percent": ram,
                "proc_rss_bytes": rss,
                "proc_pid": pid,
            }
            for ram, rss in zip(ram_values, rss_values)
        ]

    def evaluate_scenario(self, snapshots):
        self.forecaster.train(snapshots[:-3])

        pressure = self.forecaster.predict(snapshots[-3:])
        leak = self.detector.detect(snapshots[-5:])

        action = self.action_layer.evaluate_and_act(
            leak_detection=leak,
            pressure_prediction=pressure,
        )

        return action, pressure, leak

    def test_normal_memory(self):
        data = self.snapshots(
            [50.0] * 12,
            [20_000_000] * 12,
        )

        action, pressure, leak = self.evaluate_scenario(data)

        self.assertFalse(leak["is_leak_detected"])
        self.assertLess(pressure["predicted_ram_percent"], 85.0)
        self.assertEqual(action["action"], "NO_ACTION")

    def test_high_memory_pressure(self):
        data = self.snapshots(
            [80.0, 81.0, 82.0, 83.0, 84.0, 85.0,
             86.0, 87.0, 88.0, 89.0, 90.0, 91.0],
            [20_000_000] * 12,
        )

        action, pressure, leak = self.evaluate_scenario(data)

        self.assertFalse(leak["is_leak_detected"])
        self.assertGreaterEqual(
            pressure["predicted_ram_percent"], 85.0
        )
        self.assertEqual(
            action["action"], "ALERT_HIGH_PRESSURE"
        )

    def test_memory_leak(self):
        data = self.snapshots(
            [50.0] * 12,
            [20_000_000 + i * 100_000 for i in range(12)],
        )

        action, pressure, leak = self.evaluate_scenario(data)

        self.assertTrue(leak["is_leak_detected"])
        self.assertEqual(action["action"], "RESTART_PROCESS")
        self.assertEqual(action["pid"], 1234)

    def test_leak_priority_over_high_pressure(self):
        data = self.snapshots(
            [80.0, 81.0, 82.0, 83.0, 84.0, 85.0,
             86.0, 87.0, 88.0, 89.0, 90.0, 91.0],
            [20_000_000 + i * 100_000 for i in range(12)],
        )

        action, pressure, leak = self.evaluate_scenario(data)

        self.assertTrue(leak["is_leak_detected"])
        self.assertGreaterEqual(
            pressure["predicted_ram_percent"], 85.0
        )
        self.assertEqual(action["action"], "RESTART_PROCESS")


if __name__ == "__main__":
    unittest.main()
