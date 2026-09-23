"""Unit tests for CPUBurstPredictor regression model."""

import math
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np

from data.cpu.generator import CPUWorkloadGenerator, WorkloadConfig
from ml.cpu.burst_predictor import CPUBurstPredictor
from modules.cpu.process import Process
from modules.cpu.sjf import SJFScheduler


class TestCPUBurstPredictor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):

        cls.config = WorkloadConfig(
            num_processes=2000,
            correlate_burst_to_features=True,
            noise_level=0.2,
            seed=42,
        )
        cls.generator = CPUWorkloadGenerator(config=cls.config)
        cls.workload = cls.generator.generate()
        cls.df = cls.generator.to_dataframe(cls.workload)

        cls.X = cls.df[CPUBurstPredictor.FEATURE_NAMES].to_numpy(dtype=np.float64)
        cls.y = cls.df["burst_time"].to_numpy(dtype=np.float64)

        cls.predictor = CPUBurstPredictor(random_state=42)
        cls.metrics = cls.predictor.fit(cls.X, cls.y, test_size=0.20, random_state=42)

    def test_model_training_succeeds_and_metrics_returned(self):
        self.assertTrue(self.predictor.is_fitted)
        self.assertIsNotNone(self.predictor.model)

        expected_keys = {"mae", "mse", "rmse", "r2"}
        self.assertEqual(set(self.metrics.keys()), expected_keys)


        for key, val in self.metrics.items():
            self.assertTrue(np.isfinite(val))
            self.assertIsInstance(val, float)

        self.assertGreater(self.metrics["r2"], 0.5)
        self.assertGreater(self.metrics["mae"], 0.0)

    def test_predictions_are_finite_and_strictly_positive(self):
        preds = self.predictor.predict(self.X[:50])
        self.assertEqual(len(preds), 50)
        self.assertTrue(np.all(np.isfinite(preds)))
        self.assertTrue(np.all(preds > 0.0))
        self.assertTrue(np.all(preds >= self.predictor.min_prediction_floor))

    def test_predictions_have_correct_shape(self):

        pred_single = self.predictor.predict(self.X[0])
        self.assertEqual(pred_single.shape, (1,))


        pred_batch = self.predictor.predict(self.X[:10])
        self.assertEqual(pred_batch.shape, (10,))

    def test_reproducibility_with_seed(self):
        p1 = CPUBurstPredictor(random_state=42)
        p1.fit(self.X, self.y, test_size=0.20, random_state=42)

        p2 = CPUBurstPredictor(random_state=42)
        p2.fit(self.X, self.y, test_size=0.20, random_state=42)

        preds1 = p1.predict(self.X[:20])
        preds2 = p2.predict(self.X[:20])
        np.testing.assert_array_almost_equal(preds1, preds2)

    def test_predict_process_works_with_generated_process(self):
        proc = self.workload[0]
        pred_val = self.predictor.predict_process(proc)

        self.assertIsInstance(pred_val, float)
        self.assertGreater(pred_val, 0.0)
        self.assertTrue(math.isfinite(pred_val))

    def test_predict_workload_populates_predicted_burst(self):
        sample_workload = [p.copy() for p in self.workload[:10]]
        for p in sample_workload:
            self.assertIsNone(p.predicted_burst_time)

        predicted_list = self.predictor.predict_workload(sample_workload, inplace=True)
        self.assertEqual(len(predicted_list), 10)

        for p in predicted_list:
            self.assertIsNotNone(p.predicted_burst_time)
            self.assertGreater(p.predicted_burst_time, 0.0)
            self.assertIsInstance(p.predicted_burst_time, float)

    def test_true_burst_time_remains_unchanged(self):
        sample = [p.copy() for p in self.workload[:10]]
        original_bursts = [p.burst_time for p in sample]

        self.predictor.predict_workload(sample, inplace=True)

        for p, orig_burst in zip(sample, original_bursts):
            self.assertEqual(p.burst_time, orig_burst)

    def test_pid_arrival_priority_do_not_affect_prediction(self):
        proc1 = Process(
            pid="P1",
            arrival_time=0.0,
            burst_time=10.0,
            priority=1,
            metadata={
                "instruction_count": 25000.0,
                "cpu_intensity": 0.7,
                "io_wait_ratio": 0.2,
                "process_complexity": 2.0,
            },
        )
        proc2 = Process(
            pid="PROCESS_DIFFERENT",
            arrival_time=99.0,
            burst_time=10.0,
            priority=9,
            metadata={
                "instruction_count": 25000.0,
                "cpu_intensity": 0.7,
                "io_wait_ratio": 0.2,
                "process_complexity": 2.0,
            },
        )

        pred1 = self.predictor.predict_process(proc1)
        pred2 = self.predictor.predict_process(proc2)
        self.assertEqual(pred1, pred2)

    def test_missing_ml_metadata_uses_training_medians(self):

        proc_empty = Process(pid="P_EMPTY", arrival_time=0.0, burst_time=5.0, priority=1)
        pred_empty = self.predictor.predict_process(proc_empty)


        median_features = [
            self.predictor.feature_medians[feat]
            for feat in CPUBurstPredictor.FEATURE_NAMES
        ]
        expected_pred = round(float(self.predictor.predict([median_features])[0]), 2)

        self.assertEqual(pred_empty, expected_pred)

    def test_unfitted_predictor_fails_safely(self):
        unfitted = CPUBurstPredictor()
        proc = Process(pid="P1", arrival_time=0.0, burst_time=5.0)

        with self.assertRaises(RuntimeError):
            unfitted.predict([[1000.0, 0.5, 0.2, 1.5]])

        with self.assertRaises(RuntimeError):
            unfitted.predict_process(proc)

        with self.assertRaises(RuntimeError):
            unfitted.predict_workload([proc])

        with self.assertRaises(RuntimeError):
            unfitted.save("dummy.joblib")

    def test_save_and_load_produces_equivalent_predictions(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = Path(tmp_dir) / "burst_predictor.joblib"
            self.predictor.save(model_path)
            self.assertTrue(model_path.exists())

            loaded_predictor = CPUBurstPredictor.load(model_path)
            self.assertTrue(loaded_predictor.is_fitted)
            self.assertEqual(loaded_predictor.feature_medians, self.predictor.feature_medians)
            self.assertEqual(loaded_predictor.evaluation_metrics, self.predictor.evaluation_metrics)

            orig_preds = self.predictor.predict(self.X[:15])
            loaded_preds = loaded_predictor.predict(self.X[:15])
            np.testing.assert_array_almost_equal(orig_preds, loaded_preds)

    def test_load_nonexistent_or_invalid_artifact_raises(self):
        with self.assertRaises(FileNotFoundError):
            CPUBurstPredictor.load("non_existent_path.joblib")

        with tempfile.TemporaryDirectory() as tmp_dir:
            corrupt_path = Path(tmp_dir) / "corrupt.joblib"
            joblib.dump({"corrupted": True}, corrupt_path)
            with self.assertRaises(ValueError):
                CPUBurstPredictor.load(corrupt_path)

    def test_sjf_integration_with_predicted_burst(self):

        workload = [p.copy() for p in self.workload[:5]]
        self.predictor.predict_workload(workload, inplace=True)

        for p in workload:
            self.assertIsNotNone(p.predicted_burst_time)


        scheduler = SJFScheduler(use_predicted_burst=True)
        result = scheduler.schedule(workload)

        self.assertEqual(len(result.processes), 5)
        self.assertEqual(result.metrics.completed_process_count, 5)
        self.assertGreater(result.metrics.total_time, 0.0)
        self.assertGreater(result.metrics.cpu_utilization, 0.0)

    def test_train_on_synthetic_workload_convenience_method(self):
        predictor = CPUBurstPredictor(random_state=42)
        metrics = predictor.train_on_synthetic_workload(num_processes=100, seed=42)
        self.assertTrue(predictor.is_fitted)
        self.assertIn("mae", metrics)
        self.assertIn("r2", metrics)


if __name__ == "__main__":
    unittest.main()
