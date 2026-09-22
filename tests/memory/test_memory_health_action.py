import unittest
from modules.memory.memory_health_action import MemoryHealthActionLayer

class TestMemoryHealthActionLayer(unittest.TestCase):
    def setUp(self):
        self.action_layer = MemoryHealthActionLayer(pressure_threshold=85.0)

    def test_evaluate_and_act_leak_detected(self):
        leak_detection = {
            "is_leak_detected": True,
            "proc_pid": 1234,
            "growth_rate_bytes_per_step": 2048.0,
            "trend_r2_score": 0.85
        }
        # Provide both to ensure leak takes priority
        pressure_prediction = {
            "predicted_ram_percent": 90.0
        }
        
        result = self.action_layer.evaluate_and_act(leak_detection, pressure_prediction)
        
        self.assertEqual(result["action"], "RESTART_PROCESS")
        self.assertEqual(result["pid"], 1234)
        self.assertTrue("MemoryLeakDetector" in result["source"])
        self.assertEqual(len(self.action_layer.action_log), 1)

    def test_evaluate_and_act_high_pressure(self):
        leak_detection = {
            "is_leak_detected": False,
            "proc_pid": 5678,
            "growth_rate_bytes_per_step": 100.0,
            "trend_r2_score": 0.1
        }
        pressure_prediction = {
            "predicted_ram_percent": 88.5
        }
        
        result = self.action_layer.evaluate_and_act(leak_detection, pressure_prediction)
        
        self.assertEqual(result["action"], "ALERT_HIGH_PRESSURE")
        self.assertEqual(result["predicted_ram_percent"], 88.5)
        self.assertTrue("MemoryPressureForecaster" in result["source"])
        self.assertEqual(len(self.action_layer.action_log), 1)

    def test_evaluate_and_act_no_action(self):
        leak_detection = {
            "is_leak_detected": False,
            "proc_pid": 9999
        }
        pressure_prediction = {
            "predicted_ram_percent": 50.0
        }
        
        result = self.action_layer.evaluate_and_act(leak_detection, pressure_prediction)
        
        self.assertEqual(result["action"], "NO_ACTION")
        self.assertEqual(result["pid"], 9999)
        self.assertEqual(result["predicted_ram_percent"], 50.0)

    def test_evaluate_and_act_missing_inputs(self):
        result = self.action_layer.evaluate_and_act(None, None)
        
        self.assertEqual(result["action"], "NO_ACTION")
        self.assertEqual(result["source"], "None")
        self.assertIsNone(result["pid"])
        self.assertIsNone(result["predicted_ram_percent"])

    def test_invalid_leak_detection_type(self):
        with self.assertRaises(TypeError):
            self.action_layer.evaluate_and_act(leak_detection="not a dict")

    def test_invalid_pressure_prediction_type(self):
        with self.assertRaises(TypeError):
            self.action_layer.evaluate_and_act(pressure_prediction="not a dict")

    def test_invalid_threshold_type(self):
        with self.assertRaises(TypeError):
            MemoryHealthActionLayer(pressure_threshold="90")

    def test_invalid_threshold_bounds(self):
        with self.assertRaises(ValueError):
            MemoryHealthActionLayer(pressure_threshold=150.0)
            
    def test_clear_log(self):
        self.action_layer.evaluate_and_act(None, None)
        self.assertEqual(len(self.action_layer.action_log), 1)
        self.action_layer.clear_log()
        self.assertEqual(len(self.action_layer.action_log), 0)

if __name__ == "__main__":
    unittest.main()
