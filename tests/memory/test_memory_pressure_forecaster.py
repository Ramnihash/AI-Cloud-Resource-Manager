import unittest
from ml.memory.memory_pressure_forecaster import MemoryPressureForecaster

class TestMemoryPressureForecaster(unittest.TestCase):
    def setUp(self):
        self.default_window = 5
        self.forecaster = MemoryPressureForecaster(window_size=self.default_window)


        self.train_snapshots = [{"ram_percent": 10.0 + i} for i in range(20)]
        self.test_snapshots = [{"ram_percent": 30.0 + i} for i in range(10)]

    def test_construction_and_window_size(self):

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

        result = self.forecaster.train(self.train_snapshots)
        self.assertTrue(self.forecaster._trained)
        self.assertEqual(result["window_size"], 5)
        self.assertEqual(result["training_samples"], 20 - 5)

    def test_prediction_structure_and_range(self):

        self.forecaster.train(self.train_snapshots)
        context = self.train_snapshots[-5:]
        result = self.forecaster.predict(context)

        self.assertIn("predicted_ram_percent", result)
        pred = result["predicted_ram_percent"]
        self.assertIsInstance(pred, float)
        self.assertTrue(0.0 <= pred <= 100.0)



        desc_forecaster = MemoryPressureForecaster(window_size=3)
        desc_seq = [{"ram_percent": 10.0 - i * 5} for i in range(10)]
        clamped_desc = [{"ram_percent": max(0.0, 10.0 - i * 5)} for i in range(10)]


        steep_drop = [{"ram_percent": 90.0}, {"ram_percent": 70.0}, {"ram_percent": 50.0}, {"ram_percent": 30.0}, {"ram_percent": 10.0}]
        desc_forecaster.train(steep_drop)

        context_drop = steep_drop[-3:]
        pred_clamped = desc_forecaster.predict(context_drop)
        self.assertGreaterEqual(pred_clamped["predicted_ram_percent"], 0.0)

    def test_evaluation_metrics(self):

        self.forecaster.train(self.train_snapshots)
        result = self.forecaster.evaluate(self.test_snapshots)
        self.assertIn("MAE", result)
        self.assertIn("MSE", result)
        self.assertIn("total_predictions", result)
        self.assertEqual(result["total_predictions"], 10 - 5)
        self.assertIsInstance(result["MAE"], float)
        self.assertIsInstance(result["MSE"], float)

    def test_predict_evaluate_before_training(self):

        with self.assertRaises(RuntimeError):
            self.forecaster.predict(self.train_snapshots[:5])
        with self.assertRaises(RuntimeError):
            self.forecaster.evaluate(self.test_snapshots)

    def test_invalid_missing_snapshot_fields(self):

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

        with self.assertRaises(ValueError):
            self.forecaster.train(self.train_snapshots[:5])

        self.forecaster.train(self.train_snapshots)

        with self.assertRaises(ValueError):
            self.forecaster.predict(self.train_snapshots[:4])

        with self.assertRaises(ValueError):
            self.forecaster.predict(self.train_snapshots[:6])

        with self.assertRaises(ValueError):
            self.forecaster.evaluate(self.test_snapshots[:5])

    def test_deterministic_increasing_trend(self):


        trend = [{"ram_percent": 5.0 + 2.0 * i} for i in range(20)]
        self.forecaster.train(trend)


        test_trend = [{"ram_percent": 5.0 + 2.0 * i} for i in range(20, 30)]
        eval_res = self.forecaster.evaluate(test_trend)


        self.assertAlmostEqual(eval_res["MAE"], 0.0, places=5)
        self.assertAlmostEqual(eval_res["MSE"], 0.0, places=5)


        context = test_trend[:5]

        pred_res = self.forecaster.predict(context)
        self.assertAlmostEqual(pred_res["predicted_ram_percent"], 55.0, places=5)

    def test_train_test_separation(self):


        self.forecaster.train(self.train_snapshots)
        eval_res = self.forecaster.evaluate(self.test_snapshots)

        self.assertGreaterEqual(eval_res["MAE"], 0.0)

    def test_retraining_behavior(self):


        inc = [{"ram_percent": 10.0 + i} for i in range(15)]
        self.forecaster.train(inc)
        pred_inc = self.forecaster.predict(inc[-5:])["predicted_ram_percent"]

        self.assertAlmostEqual(pred_inc, 25.0, places=5)


        dec = [{"ram_percent": 50.0 - i} for i in range(15)]
        self.forecaster.train(dec)
        pred_dec = self.forecaster.predict(dec[-5:])["predicted_ram_percent"]

        self.assertAlmostEqual(pred_dec, 35.0, places=5)


if __name__ == "__main__":
    unittest.main()
