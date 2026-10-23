"""Unit tests for CP3.2 CPU workload/job-mix classifier."""

from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np

from data.cpu.generator import CPUWorkloadGenerator, WorkloadConfig
from data.cpu.workload_classifier_data import (
    CLASS_BALANCED,
    WORKLOAD_CLASSES,
)
from ml.cpu.burst_predictor import CPUBurstPredictor
from ml.cpu.workload_classifier import ClassificationResult, CPUWorkloadClassifier
from modules.cpu.process import Process


class TestCPUWorkloadClassifier(unittest.TestCase):
    """Test suite for CPUWorkloadClassifier (CP3.2 Step 2)."""

    @classmethod
    def setUpClass(cls) -> None:
        """Fit one classifier instance using the synthetic dataset generator for class-wide tests."""
        cls.classifier = CPUWorkloadClassifier(random_state=42)
        cls.metrics = cls.classifier.train_on_synthetic_dataset(seed=42)


    def test_model_training_succeeds_and_is_fitted(self) -> None:
        self.assertTrue(self.classifier.is_fitted)
        self.assertIsNotNone(self.classifier.model)
        self.assertEqual(len(self.classifier.classes_), 4)
        for c in WORKLOAD_CLASSES:
            self.assertIn(c, self.classifier.classes_)


    def test_actual_accuracy_meets_target_criteria(self) -> None:
        accuracy = self.metrics["accuracy"]
        self.assertIsInstance(accuracy, float)
        self.assertGreaterEqual(
            accuracy,
            0.90,
            f"Expected accuracy >= 0.90, got {accuracy:.4f}",
        )


    def test_actual_macro_f1_meets_target_criteria(self) -> None:
        macro_f1 = self.metrics["macro_f1"]
        self.assertIsInstance(macro_f1, float)
        self.assertGreaterEqual(
            macro_f1,
            0.90,
            f"Expected macro_f1 >= 0.90, got {macro_f1:.4f}",
        )
        self.assertTrue(self.metrics["target_criteria_met"])


    def test_confusion_matrix_structure_and_classes(self) -> None:
        cm_dict = self.metrics["confusion_matrix"]
        self.assertIn("labels", cm_dict)
        self.assertIn("matrix", cm_dict)
        self.assertEqual(len(cm_dict["labels"]), 4)

        matrix = np.asarray(cm_dict["matrix"])
        self.assertEqual(matrix.shape, (4, 4))

        self.assertEqual(int(np.sum(matrix)), self.metrics["test_size"])


    def test_exact_feature_isolation(self) -> None:
        expected_features = [
            "mean_predicted_burst_time",
            "std_predicted_burst_time",
            "mean_cpu_intensity",
            "mean_io_wait_ratio",
            "priority_variance",
        ]
        self.assertEqual(CPUWorkloadClassifier.FEATURE_NAMES, expected_features)


        with self.assertRaises(ValueError):
            self.classifier.predict(np.zeros((1, 4)))
        with self.assertRaises(ValueError):
            self.classifier.predict(np.zeros((1, 6)))


    def test_no_true_burst_time_leakage(self) -> None:
        forbidden = ["burst_time", "arrival_time", "pid", "priority"]
        for feat in forbidden:
            self.assertNotIn(feat, CPUWorkloadClassifier.FEATURE_NAMES)


        procs = [
            Process(
                pid="P1",
                arrival_time=0.0,
                burst_time=99999.0,
                priority=2,
                predicted_burst_time=5.0,
                metadata={"cpu_intensity": 0.5, "io_wait_ratio": 0.2},
            )
        ]
        features = self.classifier._extract_workload_features(procs)
        self.assertEqual(features["mean_predicted_burst_time"], 5.0)
        self.assertNotIn("burst_time", features)


    def test_input_immutability(self) -> None:
        procs = [
            Process(
                pid=f"P{i}",
                arrival_time=float(i),
                burst_time=10.0 + i,
                priority=i % 4 + 1,
                predicted_burst_time=12.0,
                metadata={"cpu_intensity": 0.6, "io_wait_ratio": 0.3},
            )
            for i in range(12)
        ]
        snapshots = [copy.deepcopy(p) for p in procs]

        result = self.classifier.predict_workload(procs)
        self.assertIsInstance(result, ClassificationResult)


        for original, after in zip(snapshots, procs):
            self.assertEqual(original.pid, after.pid)
            self.assertEqual(original.arrival_time, after.arrival_time)
            self.assertEqual(original.burst_time, after.burst_time)
            self.assertEqual(original.priority, after.priority)
            self.assertEqual(original.predicted_burst_time, after.predicted_burst_time)
            self.assertEqual(original.metadata, after.metadata)


    def test_deterministic_predictions(self) -> None:
        features = [[5.0, 1.5, 0.3, 0.75, 0.5]]
        pred1 = self.classifier.predict(features)
        pred2 = self.classifier.predict(features)
        self.assertEqual(list(pred1), list(pred2))


    def test_probability_sums_to_one(self) -> None:
        X = np.array([
            [5.0, 1.5, 0.3, 0.75, 0.5],
            [70.0, 15.0, 0.85, 0.15, 0.5],
            [6.0, 2.0, 0.35, 0.25, 6.0],
            [5.0, 1.5, 0.35, 0.30, 0.8],
        ])
        probas = self.classifier.predict_proba(X)
        self.assertEqual(probas.shape, (4, 4))
        for row in probas:
            self.assertAlmostEqual(float(np.sum(row)), 1.0, places=5)


    def test_confidence_calculation(self) -> None:
        X = np.array([[5.0, 1.5, 0.3, 0.75, 0.5]])
        probas = self.classifier.predict_proba(X)[0]
        results = self.classifier.predict_with_confidence(X)
        self.assertEqual(len(results), 1)

        pred_class, confidence, is_fallback = results[0]
        expected_confidence = round(float(np.max(probas)), 4)
        self.assertEqual(confidence, expected_confidence)
        self.assertFalse(is_fallback)


    def test_low_confidence_fallback(self) -> None:

        fallback_clf = CPUWorkloadClassifier(min_confidence=0.999, random_state=42)
        fallback_clf.model = self.classifier.model
        fallback_clf.classes_ = self.classifier.classes_
        fallback_clf.feature_medians = self.classifier.feature_medians
        fallback_clf.is_fitted = True

        # Boundary batch between the classes (the same shape as the FALLBACK evaluation scenario),
        # so the forest's vote is split and the confidence is below the threshold.
        bursts = [8.0, 8.5, 9.0, 9.5, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                  16.0, 17.0, 18.0, 19.0, 20.0, 20.5, 21.0, 21.5, 22.0, 22.5]
        procs = [
            Process(
                pid=f"P{i}",
                arrival_time=i * 0.5,
                burst_time=b,
                priority=1 if i < 9 else 4,
                predicted_burst_time=b,
                metadata={"cpu_intensity": 0.675, "io_wait_ratio": 0.548},
            )
            for i, b in enumerate(bursts)
        ]
        result = fallback_clf.predict_workload(procs)
        self.assertTrue(result.is_fallback)
        self.assertEqual(result.effective_class, CLASS_BALANCED)


    def test_unfitted_guards(self) -> None:
        unfitted = CPUWorkloadClassifier()
        self.assertFalse(unfitted.is_fitted)

        with self.assertRaises(RuntimeError):
            unfitted.predict(np.zeros((1, 5)))
        with self.assertRaises(RuntimeError):
            unfitted.predict_proba(np.zeros((1, 5)))
        with self.assertRaises(RuntimeError):
            unfitted.predict_workload([Process("P1", 0.0, 5.0, 1)])
        with self.assertRaises(RuntimeError):
            unfitted.save("dummy.joblib")


    def test_save_and_load_persistence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            artifact_path = Path(tmp_dir) / "test_classifier.joblib"
            saved_path = self.classifier.save(artifact_path)
            self.assertTrue(saved_path.exists())

            loaded = CPUWorkloadClassifier.load(saved_path)
            self.assertTrue(loaded.is_fitted)
            self.assertEqual(loaded.FEATURE_NAMES, self.classifier.FEATURE_NAMES)
            self.assertEqual(loaded.classes_, self.classifier.classes_)
            self.assertEqual(loaded.feature_medians, self.classifier.feature_medians)
            self.assertEqual(
                loaded.evaluation_metrics["accuracy"],
                self.classifier.evaluation_metrics["accuracy"],
            )


            X = np.array([[70.0, 15.0, 0.85, 0.15, 0.5]])
            np.testing.assert_array_equal(self.classifier.predict(X), loaded.predict(X))
            np.testing.assert_allclose(
                self.classifier.predict_proba(X),
                loaded.predict_proba(X),
                rtol=1e-5,
            )


    def test_corrupt_artifact_rejection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            corrupt_path = Path(tmp_dir) / "corrupt.joblib"
            joblib.dump({"wrong_payload": 123}, corrupt_path)

            with self.assertRaises(ValueError):
                CPUWorkloadClassifier.load(corrupt_path)

        with self.assertRaises(FileNotFoundError):
            CPUWorkloadClassifier.load("non_existent_file.joblib")


    def test_missing_predicted_burst_handling(self) -> None:
        procs = [
            Process(
                pid=f"P{i}",
                arrival_time=0.0,
                burst_time=10.0,
                priority=2,
                predicted_burst_time=None,
                metadata={
                    "instruction_count": 20000.0,
                    "cpu_intensity": 0.5,
                    "io_wait_ratio": 0.3,
                    "process_complexity": 2.0,
                },
            )
            for i in range(10)
        ]


        with self.assertRaises(ValueError):
            self.classifier.predict_workload(procs, burst_predictor=None)


        predictor = CPUBurstPredictor(random_state=42)
        predictor.train_on_synthetic_workload(seed=42)

        result = self.classifier.predict_workload(procs, burst_predictor=predictor)
        self.assertIsInstance(result, ClassificationResult)


        for p in procs:
            self.assertIsNone(p.predicted_burst_time)


    def test_end_to_end_predictor_to_classifier(self) -> None:
        generator = CPUWorkloadGenerator(
            config=WorkloadConfig(
                num_processes=15,
                correlate_burst_to_features=True,
                seed=42,
            )
        )
        workload = generator.generate()

        predictor = CPUBurstPredictor(random_state=42)
        predictor.train_on_synthetic_workload(seed=42)


        predictor.predict_workload(workload, inplace=True)
        for p in workload:
            self.assertIsNotNone(p.predicted_burst_time)


        result = self.classifier.predict_workload(workload)
        self.assertIsInstance(result, ClassificationResult)
        self.assertIn(result.predicted_class, WORKLOAD_CLASSES)
        self.assertIn(result.effective_class, WORKLOAD_CLASSES)
        self.assertGreaterEqual(result.confidence, 0.0)
        self.assertLessEqual(result.confidence, 1.0)
        self.assertEqual(len(result.features_used), 5)


    def test_invalid_and_empty_inputs_raise(self) -> None:
        with self.assertRaises(ValueError):
            self.classifier.predict_workload([])


if __name__ == "__main__":
    unittest.main()
