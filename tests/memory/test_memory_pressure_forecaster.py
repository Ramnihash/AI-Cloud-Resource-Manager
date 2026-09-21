import unittest
from ml.memory.memory_pressure_forecaster import MemoryPressureForecaster

class TestMemoryPressureForecaster(unittest.TestCase):
    def setUp(self):
        self.default_window = 5
        self.forecaster = MemoryPressureForecaster(window_size=self.default_window)
        
        # Valid dummy sequence for training
        self.train_snapshots = [{"ram_percent": 10.0 + i} for i in range(20)]
        self.test_snapshots = [{"ram_percent": 30.0 + i} for i in range(10)]

    def test_construction_and_window_size(self):
        # 1. Construction and window_size validation
        self.assertEqual(self.forecaster.window_size, 5)
        self.assertFalse(self.forecaster._trained)
        
        with self.assertRaises(ValueError):
            MemoryPressureForecaster(window_size=0)
        with self.assertRaises(ValueError):
            MemoryPressureForecaster(window_size=-1)
        with self.assertRaises(ValueError):
            MemoryPressureForecaster(window_size=True)
        with self.assertRaises(ValueError):
            MemoryPressureForecaster(window_size=False)
        with self.assertRaises(ValueError):
            MemoryPressureForecaster(window_size="5")
            
    def test_training(self):
        # 2. Training
        result = self.forecaster.train(self.train_snapshots)
        self.assertTrue(self.forecaster._trained)
        self.assertEqual(result["window_size"], 5)
        self.assertEqual(result["training_samples"], 20 - 5)

    def test_prediction_structure_and_range(self):
        # 3. Prediction structure and range
        self.forecaster.train(self.train_snapshots)
        context = self.train_snapshots[-5:]
        result = self.forecaster.predict(context)
        
        self.assertIn("predicted_ram_percent", result)
        pred = result["predicted_ram_percent"]
        self.assertIsInstance(pred, float)
        self.assertTrue(0.0 <= pred <= 100.0)

        # Test clipping
        # A sequence descending rapidly could predict < 0
        desc_forecaster = MemoryPressureForecaster(window_size=3)
        desc_seq = [{"ram_percent": 10.0 - i * 5} for i in range(10)]  # 10, 5, 0, -5... (values clamped below in loop to stay valid for train)
        clamped_desc = [{"ram_percent": max(0.0, 10.0 - i * 5)} for i in range(10)]
        
        # If we just feed it an artificially steep drop, linear regression might predict < 0
        steep_drop = [{"ram_percent": 90.0}, {"ram_percent": 70.0}, {"ram_percent": 50.0}, {"ram_percent": 30.0}, {"ram_percent": 10.0}]
        desc_forecaster.train(steep_drop)
        # Context is [70, 50, 30, 10], next would be -10
        context_drop = steep_drop[-3:]
        pred_clamped = desc_forecaster.predict(context_drop)
        self.assertGreaterEqual(pred_clamped["predicted_ram_percent"], 0.0)

    def test_evaluation_metrics(self):
        # 4. Evaluation metrics
        self.forecaster.train(self.train_snapshots)
        result = self.forecaster.evaluate(self.test_snapshots)
        self.assertIn("MAE", result)
        self.assertIn("MSE", result)
        self.assertIn("total_predictions", result)
        self.assertEqual(result["total_predictions"], 10 - 5)
        self.assertIsInstance(result["MAE"], float)
        self.assertIsInstance(result["MSE"], float)

    def test_predict_evaluate_before_training(self):
        # 5. Predict/evaluate before training
        with self.assertRaises(RuntimeError):
            self.forecaster.predict(self.train_snapshots[:5])
        with self.assertRaises(RuntimeError):
            self.forecaster.evaluate(self.test_snapshots)

    def test_invalid_missing_snapshot_fields(self):
        # 6. Invalid/missing snapshot fields
        with self.assertRaises(TypeError):
            self.forecaster.train("not a list")
            
        with self.assertRaises(TypeError):
            self.forecaster.train(["not a dict"] * 10)
            
        with self.assertRaises(ValueError):
            self.forecaster.train([{"wrong_key": 10.0}] * 10)
            
        with self.assertRaises(TypeError):
            self.forecaster.train([{"ram_percent": "10.0"}] * 10)
            
        with self.assertRaises(ValueError):
            self.forecaster.train([{"ram_percent": 101.0}] * 10)
            
        with self.assertRaises(ValueError):
            self.forecaster.train([{"ram_percent": -1.0}] * 10)

    def test_insufficient_data(self):
        # 7. Insufficient data
        with self.assertRaises(ValueError):
            self.forecaster.train(self.train_snapshots[:5]) # Need window_size + 1 (6)
            
        self.forecaster.train(self.train_snapshots)
        
        with self.assertRaises(ValueError):
            self.forecaster.predict(self.train_snapshots[:4]) # Need exactly 5
            
        with self.assertRaises(ValueError):
            self.forecaster.predict(self.train_snapshots[:6]) # Need exactly 5
            
        with self.assertRaises(ValueError):
            self.forecaster.evaluate(self.test_snapshots[:5]) # Need at least 6

    def test_deterministic_increasing_trend(self):
        # 8. Deterministic increasing RAM trend
        # A perfectly linear trend: y = 5 + 2*t
        trend = [{"ram_percent": 5.0 + 2.0 * i} for i in range(20)]
        self.forecaster.train(trend)
        
        # Test sequence continues the trend
        test_trend = [{"ram_percent": 5.0 + 2.0 * i} for i in range(20, 30)]
        eval_res = self.forecaster.evaluate(test_trend)
        
        # Linear regression should fit perfectly
        self.assertAlmostEqual(eval_res["MAE"], 0.0, places=5)
        self.assertAlmostEqual(eval_res["MSE"], 0.0, places=5)
        
        # Context prediction check
        context = test_trend[:5] # t=20 to t=24
        # Expected next at t=25 is 5 + 2*25 = 55
        pred_res = self.forecaster.predict(context)
        self.assertAlmostEqual(pred_res["predicted_ram_percent"], 55.0, places=5)

    def test_train_test_separation(self):
        # 9. Train/test separation
        # Evaluate on a completely separate dataset without sharing data points
        self.forecaster.train(self.train_snapshots)
        eval_res = self.forecaster.evaluate(self.test_snapshots)
        # Verify it works (metrics computed) and it didn't crash
        self.assertGreaterEqual(eval_res["MAE"], 0.0)

    def test_retraining_behavior(self):
        # 10. Retraining behavior
        # Train on increasing trend
        inc = [{"ram_percent": 10.0 + i} for i in range(15)]
        self.forecaster.train(inc)
        pred_inc = self.forecaster.predict(inc[-5:])["predicted_ram_percent"]
        # Next should be 10 + 14 + 1 = 25
        self.assertAlmostEqual(pred_inc, 25.0, places=5)
        
        # Retrain on decreasing trend
        dec = [{"ram_percent": 50.0 - i} for i in range(15)]
        self.forecaster.train(dec)
        pred_dec = self.forecaster.predict(dec[-5:])["predicted_ram_percent"]
        # Next should be 50 - 14 - 1 = 35
        self.assertAlmostEqual(pred_dec, 35.0, places=5)


if __name__ == "__main__":
    unittest.main()
