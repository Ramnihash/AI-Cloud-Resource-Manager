"""Unit tests for CPU evaluation engine (CP3.3)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ml.cpu.burst_predictor import CPUBurstPredictor
from ml.cpu.workload_classifier import CPUWorkloadClassifier
from modules.cpu.evaluation import (
    evaluate_burst_predictor,
    evaluate_workload_classifier,
    generate_evaluation_plots,
    generate_evaluation_scenarios,
    generate_markdown_report,
    run_full_evaluation,
    run_scheduler_benchmark,
)
from modules.cpu.meta_scheduler import CPUMetaScheduler


class TestCPUEvaluation(unittest.TestCase):
    """Test suite for CP3.3 evaluation pipeline."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.predictor = CPUBurstPredictor(random_state=42)
        cls.predictor.train_on_synthetic_workload(num_processes=200, seed=42)

        cls.classifier = CPUWorkloadClassifier(random_state=42)
        cls.classifier.train_on_synthetic_dataset(seed=42, target_per_class=30)

        cls.meta_scheduler = CPUMetaScheduler(
            classifier=cls.classifier,
            burst_predictor=cls.predictor,
        )


    def test_burst_predictor_evaluation_structure(self) -> None:
        result = evaluate_burst_predictor(seed=42, num_processes=100)
        metrics = result["metrics"]
        self.assertIn("mae", metrics)
        self.assertIn("mse", metrics)
        self.assertIn("rmse", metrics)
        self.assertIn("r2", metrics)
        self.assertGreaterEqual(metrics["mae"], 0.0)
        self.assertGreaterEqual(metrics["mse"], 0.0)
        self.assertGreaterEqual(metrics["rmse"], 0.0)
        self.assertIn("y_test", result["plot_data"])
        self.assertIn("y_pred", result["plot_data"])


    def test_workload_classifier_evaluation_structure(self) -> None:
        result = evaluate_workload_classifier(
            predictor=self.predictor,
            seed=42,
            target_per_class=20,
        )
        metrics = result["metrics"]
        self.assertIn("accuracy", metrics)
        self.assertIn("macro_f1", metrics)
        self.assertIn("confusion_matrix", metrics)
        self.assertIn("classification_report", metrics)
        self.assertIn("confidence_statistics", metrics)
        self.assertGreaterEqual(metrics["accuracy"], 0.0)
        self.assertLessEqual(metrics["accuracy"], 1.0)
        self.assertGreaterEqual(metrics["macro_f1"], 0.0)
        self.assertLessEqual(metrics["macro_f1"], 1.0)
        self.assertEqual(len(metrics["confusion_matrix"]["labels"]), 4)


    def test_scenarios_generation_and_sizing(self) -> None:
        scenarios = generate_evaluation_scenarios(predictor=self.predictor, seed=42)
        self.assertEqual(
            set(scenarios.keys()),
            {"CPU_BOUND", "IO_BOUND", "PRIORITY_CRITICAL", "BALANCED", "FALLBACK"},
        )
        for sc_name, procs in scenarios.items():
            self.assertEqual(len(procs), 20, f"Scenario {sc_name} must contain exactly 20 processes")


    def test_scenario_5_low_confidence_fallback(self) -> None:
        scenarios = generate_evaluation_scenarios(predictor=self.predictor, seed=42)
        fb_procs = scenarios["FALLBACK"]
        res = self.meta_scheduler.schedule(fb_procs)


        self.assertLess(res.confidence, 0.40)
        self.assertTrue(res.is_fallback)
        self.assertEqual(res.workload_class, "BALANCED")
        self.assertEqual(res.selected_scheduler, "FCFS")


    def test_run_scheduler_benchmark_grid(self) -> None:
        scenarios = generate_evaluation_scenarios(predictor=self.predictor, seed=42)
        benchmark_data = run_scheduler_benchmark(
            meta_scheduler=self.meta_scheduler,
            scenarios=scenarios,
        )
        expected_schedulers = {"FCFS", "SJF", "Round Robin", "Priority", "Meta-Scheduler"}
        for sc_name, sc_data in benchmark_data.items():
            self.assertEqual(set(sc_data["algorithms"].keys()), expected_schedulers)
            for sched_name, metrics in sc_data["algorithms"].items():
                self.assertGreaterEqual(metrics["avg_waiting_time"], 0.0)
                self.assertGreaterEqual(metrics["avg_turnaround_time"], 0.0)
                self.assertGreaterEqual(metrics["avg_response_time"], 0.0)
                self.assertGreater(metrics["cpu_utilization"], 0.0)
                self.assertGreater(metrics["makespan"], 0.0)


    def test_generate_markdown_report(self) -> None:
        eval_data = {
            "burst_predictor": {
                "metrics": {
                    "mae": 1.25,
                    "mse": 2.5,
                    "rmse": 1.58,
                    "r2": 0.89,
                    "train_size": 1600,
                    "test_size": 400,
                    "total_samples": 2000,
                    "inference_time_per_sample_us": 12.5,
                    "reference_targets": {
                        "mae_target": 2.0,
                        "r2_target": 0.85,
                        "mae_target_met": True,
                        "r2_target_met": True,
                    },
                }
            },
            "workload_classifier": {
                "metrics": {
                    "accuracy": 0.98,
                    "macro_f1": 0.98,
                    "total_batches": 1000,
                    "train_batches": 800,
                    "test_batches": 200,
                    "classes": ["BALANCED", "CPU_BOUND", "IO_BOUND", "PRIORITY_CRITICAL"],
                    "confidence_statistics": {"mean": 0.98, "std": 0.05, "min": 0.52, "max": 1.0},
                    "fallback_test_count": 0,
                    "fallback_test_rate": 0.0,
                    "classification_report": {
                        "BALANCED": {"precision": 0.98, "recall": 0.95, "f1-score": 0.96, "support": 50},
                        "CPU_BOUND": {"precision": 1.0, "recall": 1.0, "f1-score": 1.0, "support": 50},
                        "IO_BOUND": {"precision": 1.0, "recall": 1.0, "f1-score": 1.0, "support": 50},
                        "PRIORITY_CRITICAL": {"precision": 0.96, "recall": 0.98, "f1-score": 0.97, "support": 50},
                    },
                    "confusion_matrix": {
                        "labels": ["BALANCED", "CPU_BOUND", "IO_BOUND", "PRIORITY_CRITICAL"],
                        "matrix": [[48, 0, 0, 2], [0, 50, 0, 0], [0, 0, 50, 0], [1, 0, 0, 49]],
                    },
                    "reference_targets": {
                        "accuracy_target": 0.90,
                        "macro_f1_target": 0.90,
                        "accuracy_target_met": True,
                        "macro_f1_target_met": True,
                    },
                }
            },
            "benchmark_scenarios": {
                "CPU_BOUND": {
                    "ground_truth_label": "CPU_BOUND",
                    "meta_decision": {
                        "predicted_class": "CPU_BOUND",
                        "confidence": 0.95,
                        "is_fallback": False,
                        "workload_class": "CPU_BOUND",
                        "selected_scheduler": "SJF",
                    },
                    "algorithms": {
                        "Meta-Scheduler": {
                            "selected_by_meta": True,
                            "avg_waiting_time": 25.0,
                            "avg_turnaround_time": 45.0,
                            "avg_response_time": 25.0,
                            "cpu_utilization": 98.0,
                            "makespan": 150.0,
                            "scheduling_overhead_ms": 1.5,
                        }
                    },
                }
            },
        }
        md = generate_markdown_report(eval_data)
        self.assertIn("# CP3.3 — AI-Assisted CPU Scheduling Evaluation Report", md)
        self.assertIn("Burst Time Prediction", md)
        self.assertIn("Confusion Matrix", md)


    def test_run_full_evaluation_file_generation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            plots = generate_evaluation_plots(
                {
                    "burst_predictor": {
                        "plot_data": {"y_test": [10.0, 20.0, 30.0], "y_pred": [11.0, 19.5, 31.0]}
                    },
                    "workload_classifier": {
                        "metrics": {
                            "confusion_matrix": {
                                "labels": ["BAL", "CPU", "IO", "PRI"],
                                "matrix": [[10, 0, 0, 0], [0, 10, 0, 0], [0, 0, 10, 0], [0, 0, 0, 10]],
                            }
                        }
                    },
                    "benchmark_scenarios": {
                        "S1": {
                            "algorithms": {
                                "FCFS": {"avg_turnaround_time": 50.0, "avg_response_time": 20.0},
                                "SJF": {"avg_turnaround_time": 40.0, "avg_response_time": 18.0},
                                "Round Robin": {"avg_turnaround_time": 45.0, "avg_response_time": 10.0},
                                "Priority": {"avg_turnaround_time": 48.0, "avg_response_time": 19.0},
                                "Meta-Scheduler": {"avg_turnaround_time": 40.0, "avg_response_time": 18.0},
                            }
                        }
                    },
                },
                Path(tmp_dir),
            )
            self.assertEqual(len(plots), 4)
            for p in plots:
                self.assertTrue(Path(p).exists())


if __name__ == "__main__":
    unittest.main()
