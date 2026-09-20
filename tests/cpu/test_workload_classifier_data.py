"""Unit tests for CP3.2 CPU workload classification dataset generation and feature extraction."""

from __future__ import annotations

import unittest
from collections import Counter

import numpy as np
import pandas as pd

from data.cpu.workload_classifier_data import (
    ALL_DATASET_COLUMNS,
    CLASS_BALANCED,
    CLASS_CPU_BOUND,
    CLASS_IO_BOUND,
    CLASS_PRIORITY_CRITICAL,
    CLASSIFIER_FEATURE_NAMES,
    METADATA_COLUMNS,
    TARGET_COLUMN,
    WORKLOAD_CLASSES,
    WorkloadClassificationDatasetGenerator,
    WorkloadClassificationSample,
    extract_batch_features,
    label_batch,
    samples_to_dataframe,
)
from ml.cpu.burst_predictor import CPUBurstPredictor
from modules.cpu.process import Process


class TestWorkloadClassifierData(unittest.TestCase):
    """Test suite for CP3.2 workload classification dataset generation and feature extraction."""

    @classmethod
    def setUpClass(cls) -> None:
        """Generate the locked 1,000-sample dataset once for class-wide verification."""
        cls.predictor = CPUBurstPredictor(random_state=42)
        cls.predictor.train_on_synthetic_workload(seed=42)
        cls.generator = WorkloadClassificationDatasetGenerator(
            predictor=cls.predictor,
            seed=42,
            target_per_class=250,
            min_batch_size=10,
            max_batch_size=30,
        )
        cls.samples = cls.generator.generate_dataset(seed=42, target_per_class=250)
        cls.df = cls.generator.to_dataframe(cls.samples)

    # 1. Exactly 1,000 samples are generated
    def test_exactly_1000_samples_generated(self) -> None:
        self.assertEqual(len(self.samples), 1000)
        self.assertEqual(len(self.df), 1000)

    # 2. Exactly 250 samples exist for each class
    def test_exactly_250_samples_per_class(self) -> None:
        counts = Counter(s.label for s in self.samples)
        for class_name in WORKLOAD_CLASSES:
            self.assertEqual(
                counts[class_name],
                250,
                f"Expected 250 samples for {class_name}, got {counts[class_name]}",
            )
        df_counts = self.df["label"].value_counts().to_dict()
        for class_name in WORKLOAD_CLASSES:
            self.assertEqual(df_counts[class_name], 250)

    # 3. Every batch size is between 10 and 30
    def test_batch_sizes_between_10_and_30(self) -> None:
        batch_sizes = [s.batch_size for s in self.samples]
        self.assertGreaterEqual(min(batch_sizes), 10)
        self.assertLessEqual(max(batch_sizes), 30)
        self.assertGreaterEqual(self.df["batch_size"].min(), 10)
        self.assertLessEqual(self.df["batch_size"].max(), 30)

    # 4. Same seed produces identical dataset
    def test_same_seed_produces_identical_dataset(self) -> None:
        gen1 = WorkloadClassificationDatasetGenerator(
            predictor=self.predictor,
            seed=42,
            target_per_class=5,
        )
        gen2 = WorkloadClassificationDatasetGenerator(
            predictor=self.predictor,
            seed=42,
            target_per_class=5,
        )
        samples1 = gen1.generate_dataset(seed=42, target_per_class=5)
        samples2 = gen2.generate_dataset(seed=42, target_per_class=5)

        self.assertEqual(len(samples1), len(samples2))
        for s1, s2 in zip(samples1, samples2):
            self.assertEqual(s1.batch_size, s2.batch_size)
            self.assertEqual(s1.label, s2.label)
            self.assertAlmostEqual(s1.mean_predicted_burst_time, s2.mean_predicted_burst_time, places=4)
            self.assertAlmostEqual(s1.std_predicted_burst_time, s2.std_predicted_burst_time, places=4)
            self.assertAlmostEqual(s1.mean_cpu_intensity, s2.mean_cpu_intensity, places=4)
            self.assertAlmostEqual(s1.mean_io_wait_ratio, s2.mean_io_wait_ratio, places=4)
            self.assertAlmostEqual(s1.priority_variance, s2.priority_variance, places=4)

    # 5. Different seed produces different dataset
    def test_different_seed_produces_different_dataset(self) -> None:
        gen = WorkloadClassificationDatasetGenerator(
            predictor=self.predictor,
            seed=42,
            target_per_class=5,
        )
        samples_seed42 = gen.generate_dataset(seed=42, target_per_class=5)
        samples_seed99 = gen.generate_dataset(seed=99, target_per_class=5)

        self.assertEqual(len(samples_seed42), 20)
        self.assertEqual(len(samples_seed99), 20)

        # Confirm differences in generated features across different seeds
        burst_means_42 = [s.mean_predicted_burst_time for s in samples_seed42]
        burst_means_99 = [s.mean_predicted_burst_time for s in samples_seed99]
        self.assertNotEqual(burst_means_42, burst_means_99)

    # 6. Labeling rules are correct under standard conditions
    def test_labeling_rules_basic(self) -> None:
        # PRIORITY_CRITICAL
        procs_pri = [
            Process(f"P{i}", 0.0, 5.0, pri, metadata={"cpu_intensity": 0.3, "io_wait_ratio": 0.2})
            for i, pri in enumerate([1, 1, 5, 5] * 3)  # variance = 4.0 >= 2.0, has priority 1
        ]
        self.assertEqual(label_batch(procs_pri), CLASS_PRIORITY_CRITICAL)

        # IO_BOUND
        procs_io = [
            Process(f"P{i}", 0.0, 5.0, 2, metadata={"cpu_intensity": 0.3, "io_wait_ratio": 0.75})
            for i in range(12)  # io >= 0.60, no priority 1
        ]
        self.assertEqual(label_batch(procs_io), CLASS_IO_BOUND)

        # CPU_BOUND
        procs_cpu = [
            Process(f"P{i}", 0.0, 20.0, 2, metadata={"cpu_intensity": 0.85, "io_wait_ratio": 0.2})
            for i in range(12)  # cpu >= 0.70, burst >= 15.0, no priority 1
        ]
        self.assertEqual(label_batch(procs_cpu), CLASS_CPU_BOUND)

        # BALANCED
        procs_bal = [
            Process(f"P{i}", 0.0, 8.0, 2, metadata={"cpu_intensity": 0.45, "io_wait_ratio": 0.35})
            for i in range(12)
        ]
        self.assertEqual(label_batch(procs_bal), CLASS_BALANCED)

    # 7. PRIORITY_CRITICAL takes precedence over IO_BOUND and CPU_BOUND when multiple conditions match
    def test_priority_critical_precedence_over_io_and_cpu(self) -> None:
        # Meets PRIORITY_CRITICAL (var >= 2.0, pri==1), IO_BOUND (io >= 0.60), and CPU_BOUND (cpu >= 0.70, burst >= 15)
        procs = [
            Process(f"P{i}", 0.0, 25.0, pri, metadata={"cpu_intensity": 0.90, "io_wait_ratio": 0.80})
            for i, pri in enumerate([1, 1, 5, 5] * 3)
        ]
        self.assertEqual(label_batch(procs), CLASS_PRIORITY_CRITICAL)

    # 8. IO_BOUND takes precedence over CPU_BOUND when both match and priority-critical does not
    def test_io_bound_precedence_over_cpu_bound(self) -> None:
        # Meets IO_BOUND (io >= 0.60) and CPU_BOUND (cpu >= 0.70, burst >= 15.0), but no priority 1 (pri=3)
        procs = [
            Process(f"P{i}", 0.0, 25.0, 3, metadata={"cpu_intensity": 0.85, "io_wait_ratio": 0.75})
            for i in range(12)
        ]
        self.assertEqual(label_batch(procs), CLASS_IO_BOUND)

    # 9. CPU_BOUND is correctly detected when IO_BOUND and PRIORITY_CRITICAL do not match
    def test_cpu_bound_detected(self) -> None:
        procs = [
            Process(f"P{i}", 0.0, 18.0, 3, metadata={"cpu_intensity": 0.80, "io_wait_ratio": 0.25})
            for i in range(12)
        ]
        self.assertEqual(label_batch(procs), CLASS_CPU_BOUND)

    # 10. BALANCED is the fallback when no conditions match
    def test_balanced_fallback(self) -> None:
        procs = [
            Process(f"P{i}", 0.0, 10.0, 3, metadata={"cpu_intensity": 0.50, "io_wait_ratio": 0.40})
            for i in range(12)
        ]
        self.assertEqual(label_batch(procs), CLASS_BALANCED)

    # Threshold Boundary Tests
    def test_boundary_priority_variance_exactly_2_0_with_priority_1(self) -> None:
        # [1, 3, 3, 5] has mean 3.0, diffs [-2, 0, 0, 2], squared diffs [4, 0, 0, 4], variance 8/4 = 2.0
        procs = [
            Process(f"P{i}", 0.0, 5.0, pri, metadata={"cpu_intensity": 0.3, "io_wait_ratio": 0.2})
            for i, pri in enumerate([1, 3, 3, 5] * 3)
        ]
        self.assertEqual(label_batch(procs), CLASS_PRIORITY_CRITICAL)

    def test_boundary_priority_variance_exactly_2_0_without_priority_1(self) -> None:
        # [2, 4, 4, 6] has mean 4.0, diffs [-2, 0, 0, 2], squared diffs [4, 0, 0, 4], variance 8/4 = 2.0
        # No process has priority == 1
        procs = [
            Process(f"P{i}", 0.0, 5.0, pri, metadata={"cpu_intensity": 0.3, "io_wait_ratio": 0.2})
            for i, pri in enumerate([2, 4, 4, 6] * 3)
        ]
        self.assertNotEqual(label_batch(procs), CLASS_PRIORITY_CRITICAL)
        self.assertEqual(label_batch(procs), CLASS_BALANCED)

    def test_boundary_priority_variance_below_2_0(self) -> None:
        # [1, 1, 3, 3, 3, 3, 3, 3, 5, 5] has mean 3.0, squared diffs [4, 4, 0*6, 4, 4] = 16, var = 1.6 < 2.0
        # Has priority 1, but variance is 1.6 < 2.0
        procs = [
            Process(f"P{i}", 0.0, 5.0, pri, metadata={"cpu_intensity": 0.3, "io_wait_ratio": 0.2})
            for i, pri in enumerate([1, 1, 3, 3, 3, 3, 3, 3, 5, 5])
        ]
        self.assertNotEqual(label_batch(procs), CLASS_PRIORITY_CRITICAL)
        self.assertEqual(label_batch(procs), CLASS_BALANCED)

    def test_boundary_mean_io_wait_ratio_exactly_0_60(self) -> None:
        procs = [
            Process(f"P{i}", 0.0, 5.0, 3, metadata={"cpu_intensity": 0.3, "io_wait_ratio": 0.60})
            for i in range(12)
        ]
        self.assertEqual(label_batch(procs), CLASS_IO_BOUND)

    def test_boundary_mean_io_wait_ratio_below_0_60(self) -> None:
        procs = [
            Process(f"P{i}", 0.0, 5.0, 3, metadata={"cpu_intensity": 0.3, "io_wait_ratio": 0.5999})
            for i in range(12)
        ]
        self.assertNotEqual(label_batch(procs), CLASS_IO_BOUND)
        self.assertEqual(label_batch(procs), CLASS_BALANCED)

    def test_boundary_mean_cpu_intensity_exactly_0_70_and_burst_15_0(self) -> None:
        procs = [
            Process(f"P{i}", 0.0, 15.0, 3, metadata={"cpu_intensity": 0.70, "io_wait_ratio": 0.2})
            for i in range(12)
        ]
        self.assertEqual(label_batch(procs), CLASS_CPU_BOUND)

    def test_boundary_cpu_intensity_0_70_burst_below_15_0(self) -> None:
        procs = [
            Process(f"P{i}", 0.0, 14.99, 3, metadata={"cpu_intensity": 0.70, "io_wait_ratio": 0.2})
            for i in range(12)
        ]
        self.assertNotEqual(label_batch(procs), CLASS_CPU_BOUND)
        self.assertEqual(label_batch(procs), CLASS_BALANCED)

    def test_boundary_cpu_intensity_below_0_70_burst_15_0(self) -> None:
        procs = [
            Process(f"P{i}", 0.0, 15.0, 3, metadata={"cpu_intensity": 0.6999, "io_wait_ratio": 0.2})
            for i in range(12)
        ]
        self.assertNotEqual(label_batch(procs), CLASS_CPU_BOUND)
        self.assertEqual(label_batch(procs), CLASS_BALANCED)

    # 11. All five classifier features are present
    def test_all_five_classifier_features_present(self) -> None:
        expected_features = [
            "mean_predicted_burst_time",
            "std_predicted_burst_time",
            "mean_cpu_intensity",
            "mean_io_wait_ratio",
            "priority_variance",
        ]
        self.assertEqual(CLASSIFIER_FEATURE_NAMES, expected_features)
        for feat in expected_features:
            self.assertIn(feat, self.df.columns)
            self.assertFalse(self.df[feat].isnull().any())
            self.assertTrue(np.all(np.isfinite(self.df[feat].to_numpy())))

    # 12. pid/arrival_time/true burst_time are not classifier features
    def test_pid_arrival_true_burst_not_classifier_features(self) -> None:
        forbidden = ["pid", "arrival_time", "burst_time"]
        for col in forbidden:
            self.assertNotIn(col, CLASSIFIER_FEATURE_NAMES)

        X, y = WorkloadClassificationDatasetGenerator.get_feature_matrix_and_labels(self.df)
        self.assertEqual(X.shape, (1000, 5))
        self.assertEqual(len(y), 1000)

    # 13. predicted_burst_time is actually produced through CPUBurstPredictor
    def test_predicted_burst_time_produced_through_cpu_burst_predictor(self) -> None:
        # Create unpredicted processes
        procs = [
            Process(
                pid=f"P{i}",
                arrival_time=0.0,
                burst_time=10.0,
                priority=2,
                metadata={
                    "instruction_count": 20000.0,
                    "cpu_intensity": 0.5,
                    "io_wait_ratio": 0.3,
                    "process_complexity": 2.0,
                },
            )
            for i in range(10)
        ]
        # Attempting extraction before prediction must raise ValueError
        with self.assertRaises(ValueError):
            extract_batch_features(procs)

        # Run prediction via CPUBurstPredictor
        self.predictor.predict_workload(procs, inplace=True)
        for p in procs:
            self.assertIsNotNone(p.predicted_burst_time)
            self.assertGreater(p.predicted_burst_time, 0.0)

        # Extraction must now succeed
        features = extract_batch_features(procs)
        expected_mean = round(float(np.mean([p.predicted_burst_time for p in procs])), 4)
        self.assertEqual(features["mean_predicted_burst_time"], expected_mean)

    # Train / Test Stratified Split Support
    def test_stratified_train_test_split(self) -> None:
        X_train, X_test, y_train, y_test = WorkloadClassificationDatasetGenerator.get_train_test_split(
            self.df, test_size=0.20, random_state=42
        )
        self.assertEqual(X_train.shape, (800, 5))
        self.assertEqual(X_test.shape, (200, 5))

        train_counts = Counter(y_train)
        test_counts = Counter(y_test)
        for c in WORKLOAD_CLASSES:
            self.assertEqual(train_counts[c], 200)
            self.assertEqual(test_counts[c], 50)

    # Edge cases and validation
    def test_empty_workload_raises(self) -> None:
        with self.assertRaises(ValueError):
            label_batch([])
        with self.assertRaises(ValueError):
            extract_batch_features([])

    def test_sample_dataclass_and_dataframe_conversion(self) -> None:
        sample = WorkloadClassificationSample(
            batch_id="batch_0001",
            batch_size=15,
            mean_predicted_burst_time=12.5,
            std_predicted_burst_time=2.3,
            mean_cpu_intensity=0.75,
            mean_io_wait_ratio=0.25,
            priority_variance=1.8,
            label=CLASS_CPU_BOUND,
        )
        d = sample.to_dict()
        self.assertEqual(d["batch_id"], "batch_0001")
        self.assertEqual(d["label"], CLASS_CPU_BOUND)

        df = samples_to_dataframe([sample])
        self.assertEqual(list(df.columns), ALL_DATASET_COLUMNS)
        self.assertEqual(len(df), 1)

    def test_candidate_config_broadened_boundary_ranges(self) -> None:
        """Verify that candidate generation bounds reach the broadened lower limits."""
        import random
        rng = random.Random(42)

        cpu_instr_lows = []
        cpu_comp_lows = []
        io_wait_lows = []

        for _ in range(100):
            cpu_cfg = self.generator._generate_candidate_config(CLASS_CPU_BOUND, rng)
            cpu_instr_lows.append(cpu_cfg.instruction_count_range[0])
            cpu_comp_lows.append(cpu_cfg.process_complexity_range[0])

            io_cfg = self.generator._generate_candidate_config(CLASS_IO_BOUND, rng)
            io_wait_lows.append(io_cfg.io_wait_ratio_range[0])

        # CPU_BOUND: instruction_count lower reaches ~15000, complexity lower reaches ~1.5
        self.assertLess(min(cpu_instr_lows), 20000.0)
        self.assertGreaterEqual(min(cpu_instr_lows), 15000.0)
        self.assertLess(min(cpu_comp_lows), 1.8)
        self.assertGreaterEqual(min(cpu_comp_lows), 1.5)

        # IO_BOUND: io_wait_ratio lower reaches 0.60
        self.assertAlmostEqual(min(io_wait_lows), 0.60, places=2)


if __name__ == "__main__":
    unittest.main()
