"""Unit and integration tests for CPUMetaScheduler (CP3.2 Step 3)."""

from __future__ import annotations

import unittest
from typing import Any, Optional

from ml.cpu.burst_predictor import CPUBurstPredictor
from ml.cpu.workload_classifier import (
    CLASS_BALANCED,
    CLASS_CPU_BOUND,
    CLASS_IO_BOUND,
    CLASS_PRIORITY_CRITICAL,
    ClassificationResult,
    CPUWorkloadClassifier,
)
from modules.cpu.base_scheduler import BaseScheduler, ScheduleResult
from modules.cpu.fcfs import FCFSScheduler
from modules.cpu.meta_scheduler import (
    CLASS_TO_SCHEDULER_NAME,
    CPUMetaScheduler,
    MetaScheduleResult,
)
from modules.cpu.metrics import GanttSlice
from modules.cpu.priority import PriorityScheduler
from modules.cpu.process import Process
from modules.cpu.round_robin import RoundRobinScheduler
from modules.cpu.sjf import SJFScheduler


class StubWorkloadClassifier(CPUWorkloadClassifier):
    """Deterministic test double for CPUWorkloadClassifier."""

    def __init__(
        self,
        predicted_class: str = CLASS_BALANCED,
        confidence: float = 0.95,
        is_fallback: bool = False,
        effective_class: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.is_fitted = True
        self.stub_predicted_class = predicted_class
        self.stub_confidence = confidence
        self.stub_is_fallback = is_fallback
        self.stub_effective_class = effective_class or (
            CLASS_BALANCED if is_fallback else predicted_class
        )

    def predict_workload(
        self,
        processes: list[Process],
        burst_predictor: Optional[CPUBurstPredictor] = None,
    ) -> ClassificationResult:
        return ClassificationResult(
            predicted_class=self.stub_predicted_class,
            confidence=self.stub_confidence,
            class_probabilities={self.stub_predicted_class: self.stub_confidence},
            is_fallback=self.stub_is_fallback,
            effective_class=self.stub_effective_class,
            features_used={
                "mean_predicted_burst_time": 10.0,
                "std_predicted_burst_time": 2.0,
                "mean_cpu_intensity": 0.5,
                "mean_io_wait_ratio": 0.5,
                "priority_variance": 1.0,
            },
        )


class RecordingMockScheduler(BaseScheduler):
    """Test scheduler that records execution invocations."""

    def __init__(self, name: str = "MockScheduler") -> None:
        super().__init__(name=name)
        self.invocations: list[dict[str, Any]] = []

    def _run_schedule(
        self, processes: list[Process], **kwargs: Any
    ) -> tuple[list[Process], list[GanttSlice]]:
        self.invocations.append({"processes": [p.copy() for p in processes], "kwargs": kwargs})
        completed = []
        slices = []
        t = 0.0
        for p in processes:
            p.start_time = t
            p.completion_time = t + p.burst_time
            p.remaining_time = 0.0
            p.waiting_time = 0.0
            p.turnaround_time = p.burst_time
            p.response_time = 0.0
            slices.append(GanttSlice(t, t + p.burst_time, p.pid))
            t += p.burst_time
            completed.append(p)
        return completed, slices


class TestCPUMetaSchedulerUnit(unittest.TestCase):
    """Focused unit tests for CPUMetaScheduler."""

    def setUp(self) -> None:
        self.default_workload = [
            Process("P1", arrival_time=0.0, burst_time=10.0, priority=2, predicted_burst_time=10.0),
            Process("P2", arrival_time=0.0, burst_time=4.0, priority=1, predicted_burst_time=4.0),
        ]


    def test_cpu_bound_maps_to_sjf(self) -> None:
        clf = StubWorkloadClassifier(predicted_class=CLASS_CPU_BOUND)
        meta = CPUMetaScheduler(classifier=clf)
        result = meta.schedule(self.default_workload)

        self.assertEqual(result.selected_scheduler, "SJF")
        self.assertEqual(result.workload_class, CLASS_CPU_BOUND)
        self.assertEqual(result.predicted_class, CLASS_CPU_BOUND)
        self.assertFalse(result.is_fallback)
        self.assertEqual(result.schedule_result.algorithm_name, "SJF")


    def test_io_bound_maps_to_round_robin(self) -> None:
        clf = StubWorkloadClassifier(predicted_class=CLASS_IO_BOUND)
        meta = CPUMetaScheduler(classifier=clf)
        result = meta.schedule(self.default_workload)

        self.assertEqual(result.selected_scheduler, "Round Robin")
        self.assertEqual(result.workload_class, CLASS_IO_BOUND)
        self.assertEqual(result.predicted_class, CLASS_IO_BOUND)
        self.assertFalse(result.is_fallback)
        self.assertEqual(result.schedule_result.algorithm_name, "Round Robin")


    def test_priority_critical_maps_to_priority(self) -> None:
        clf = StubWorkloadClassifier(predicted_class=CLASS_PRIORITY_CRITICAL)
        meta = CPUMetaScheduler(classifier=clf)
        result = meta.schedule(self.default_workload)

        self.assertEqual(result.selected_scheduler, "Priority")
        self.assertEqual(result.workload_class, CLASS_PRIORITY_CRITICAL)
        self.assertEqual(result.predicted_class, CLASS_PRIORITY_CRITICAL)
        self.assertFalse(result.is_fallback)
        self.assertEqual(result.schedule_result.algorithm_name, "Priority")


    def test_balanced_maps_to_fcfs(self) -> None:
        clf = StubWorkloadClassifier(predicted_class=CLASS_BALANCED)
        meta = CPUMetaScheduler(classifier=clf)
        result = meta.schedule(self.default_workload)

        self.assertEqual(result.selected_scheduler, "FCFS")
        self.assertEqual(result.workload_class, CLASS_BALANCED)
        self.assertEqual(result.predicted_class, CLASS_BALANCED)
        self.assertFalse(result.is_fallback)
        self.assertEqual(result.schedule_result.algorithm_name, "FCFS")


    def test_low_confidence_fallback_maps_to_fcfs(self) -> None:
        clf = StubWorkloadClassifier(
            predicted_class=CLASS_CPU_BOUND,
            confidence=0.35,
            is_fallback=True,
            effective_class=CLASS_BALANCED,
        )
        meta = CPUMetaScheduler(classifier=clf)
        result = meta.schedule(self.default_workload)

        self.assertTrue(result.is_fallback)
        self.assertEqual(result.predicted_class, CLASS_CPU_BOUND)
        self.assertEqual(result.workload_class, CLASS_BALANCED)
        self.assertEqual(result.confidence, 0.35)
        self.assertEqual(result.selected_scheduler, "FCFS")
        self.assertEqual(result.schedule_result.algorithm_name, "FCFS")


    def test_selected_scheduler_is_actually_invoked(self) -> None:

        clf_sjf = StubWorkloadClassifier(CLASS_CPU_BOUND)
        meta_sjf = CPUMetaScheduler(classifier=clf_sjf)
        result_sjf = meta_sjf.schedule(self.default_workload)
        execution_pids = [s.pid for s in result_sjf.gantt_chart if not s.is_idle]
        self.assertEqual(execution_pids, ["P2", "P1"])


        clf_pri = StubWorkloadClassifier(CLASS_PRIORITY_CRITICAL)
        meta_pri = CPUMetaScheduler(classifier=clf_pri)
        result_pri = meta_pri.schedule(self.default_workload)
        execution_pids_pri = [s.pid for s in result_pri.gantt_chart if not s.is_idle]
        self.assertEqual(execution_pids_pri, ["P2", "P1"])


        rr_workload = [
            Process("P1", 0.0, 4.0, 1, predicted_burst_time=4.0),
            Process("P2", 0.0, 4.0, 1, predicted_burst_time=4.0),
        ]
        clf_rr = StubWorkloadClassifier(CLASS_IO_BOUND)
        meta_rr = CPUMetaScheduler(classifier=clf_rr, default_time_quantum=2.0)
        result_rr = meta_rr.schedule(rr_workload)
        rr_slices = [s.pid for s in result_rr.gantt_chart if not s.is_idle]
        self.assertEqual(rr_slices, ["P1", "P2", "P1", "P2"])


    def test_caller_workload_remains_unchanged(self) -> None:
        original_processes = [
            Process(
                pid="P1",
                arrival_time=0.0,
                burst_time=12.0,
                priority=3,
                predicted_burst_time=None,
                metadata={"cpu_intensity": 0.8, "io_wait_ratio": 0.2},
            ),
            Process(
                pid="P2",
                arrival_time=1.0,
                burst_time=5.0,
                priority=1,
                predicted_burst_time=None,
                metadata={"cpu_intensity": 0.7, "io_wait_ratio": 0.3},
            ),
        ]

        original_dicts = [p.to_dict() for p in original_processes]

        predictor = CPUBurstPredictor(random_state=42)
        predictor.train_on_synthetic_workload(seed=42, num_processes=30)

        meta = CPUMetaScheduler(
            classifier=StubWorkloadClassifier(CLASS_CPU_BOUND),
            burst_predictor=predictor,
        )
        meta.schedule(original_processes)


        for p, original_dict in zip(original_processes, original_dicts):
            self.assertEqual(p.to_dict(), original_dict)
            self.assertIsNone(p.start_time)
            self.assertIsNone(p.completion_time)
            self.assertEqual(p.waiting_time, 0.0)
            self.assertEqual(p.turnaround_time, 0.0)
            self.assertEqual(p.remaining_time, p.burst_time)
            self.assertIsNone(p.predicted_burst_time)


    def test_missing_predicted_burst_with_supplied_predictor_works(self) -> None:
        workload = [
            Process(
                pid="P1",
                arrival_time=0.0,
                burst_time=15.0,
                priority=2,
                predicted_burst_time=None,
                metadata={
                    "instruction_count": 25000.0,
                    "cpu_intensity": 0.8,
                    "io_wait_ratio": 0.2,
                    "process_complexity": 2.5,
                },
            )
        ]
        predictor = CPUBurstPredictor(random_state=42)
        predictor.train_on_synthetic_workload(seed=42, num_processes=30)

        meta = CPUMetaScheduler(
            classifier=StubWorkloadClassifier(CLASS_BALANCED),
            burst_predictor=predictor,
        )
        result = meta.schedule(workload)


        self.assertEqual(result.selected_scheduler, "FCFS")

        self.assertIsNone(workload[0].predicted_burst_time)

        self.assertIsNotNone(result.processes[0].predicted_burst_time)


    def test_missing_predicted_burst_without_predictor_raises_value_error(self) -> None:
        workload = [
            Process("P1", 0.0, 5.0, 1, predicted_burst_time=None),
        ]
        meta = CPUMetaScheduler(
            classifier=StubWorkloadClassifier(CLASS_BALANCED),
            burst_predictor=None,
        )
        with self.assertRaises(ValueError) as ctx:
            meta.schedule(workload)
        self.assertIn("missing predicted_burst_time", str(ctx.exception))


    def test_empty_workload_raises_value_error(self) -> None:
        meta = CPUMetaScheduler(classifier=StubWorkloadClassifier())
        with self.assertRaises(ValueError) as ctx:
            meta.schedule([])
        self.assertIn("empty process workload", str(ctx.exception))


    def test_invalid_process_type_raises_type_error(self) -> None:
        meta = CPUMetaScheduler(classifier=StubWorkloadClassifier())
        with self.assertRaises(TypeError):
            meta.schedule([{"pid": "P1", "burst_time": 5.0}])

        with self.assertRaises(TypeError):
            meta.schedule(["invalid_process_string"])


    def test_unfitted_classifier_raises_runtime_error(self) -> None:
        unfitted = CPUWorkloadClassifier()
        self.assertFalse(unfitted.is_fitted)

        meta = CPUMetaScheduler(classifier=unfitted)
        with self.assertRaises(RuntimeError) as ctx:
            meta.schedule(self.default_workload)
        self.assertIn("CPUWorkloadClassifier is not fitted", str(ctx.exception))

        with self.assertRaises(RuntimeError):
            meta.classify_workload(self.default_workload)


    def test_unfitted_burst_predictor_raises_runtime_error(self) -> None:
        unfitted_pred = CPUBurstPredictor()
        self.assertFalse(unfitted_pred.is_fitted)

        meta = CPUMetaScheduler(
            classifier=StubWorkloadClassifier(),
            burst_predictor=unfitted_pred,
        )
        with self.assertRaises(RuntimeError) as ctx:
            meta.schedule(self.default_workload)
        self.assertIn("CPUBurstPredictor is not fitted", str(ctx.exception))

        with self.assertRaises(RuntimeError):
            meta.classify_workload(self.default_workload)


    def test_missing_custom_scheduler_raises_clear_key_error(self) -> None:
        custom_schedulers = {"FCFS": FCFSScheduler()}
        meta = CPUMetaScheduler(
            classifier=StubWorkloadClassifier(CLASS_CPU_BOUND),
            schedulers=custom_schedulers,
        )
        with self.assertRaises(KeyError) as ctx:
            meta.schedule(self.default_workload)
        self.assertIn("SJF", str(ctx.exception))


    def test_result_structure_and_serialization(self) -> None:
        clf = StubWorkloadClassifier(CLASS_CPU_BOUND, confidence=0.92)
        meta = CPUMetaScheduler(classifier=clf)
        result = meta.schedule(self.default_workload)

        self.assertIsInstance(result, MetaScheduleResult)
        self.assertEqual(result.algorithm_name, result.selected_scheduler)
        self.assertEqual(result.processes, result.schedule_result.processes)
        self.assertEqual(result.gantt_chart, result.schedule_result.gantt_chart)
        self.assertEqual(result.metrics, result.schedule_result.metrics)
        self.assertEqual(result.confidence, 0.92)

        data = result.to_dict()
        self.assertIn("selected_scheduler", data)
        self.assertIn("workload_class", data)
        self.assertIn("predicted_class", data)
        self.assertIn("confidence", data)
        self.assertIn("is_fallback", data)
        self.assertIn("classification_result", data)
        self.assertIn("schedule_result", data)
        self.assertIn("scheduler_kwargs_used", data)
        self.assertEqual(data["selected_scheduler"], "SJF")


    def test_injected_custom_schedulers_work_correctly(self) -> None:
        mock_sjf = RecordingMockScheduler(name="MockSJF")
        custom_registry = {
            "SJF": mock_sjf,
            "Round Robin": RoundRobinScheduler(),
            "Priority": PriorityScheduler(),
            "FCFS": FCFSScheduler(),
        }
        meta = CPUMetaScheduler(
            classifier=StubWorkloadClassifier(CLASS_CPU_BOUND),
            schedulers=custom_registry,
        )
        result = meta.schedule(self.default_workload)

        self.assertEqual(result.selected_scheduler, "MockSJF")
        self.assertEqual(len(mock_sjf.invocations), 1)
        self.assertEqual(len(mock_sjf.invocations[0]["processes"]), 2)


    def test_consecutive_runs_do_not_leak_scheduler_state(self) -> None:
        clf = StubWorkloadClassifier(CLASS_IO_BOUND)
        meta = CPUMetaScheduler(classifier=clf)

        workload_1 = [
            Process("A1", 0.0, 4.0, 1, predicted_burst_time=4.0),
            Process("A2", 0.0, 4.0, 1, predicted_burst_time=4.0),
        ]
        workload_2 = [
            Process("B1", 0.0, 6.0, 1, predicted_burst_time=6.0),
            Process("B2", 0.0, 2.0, 1, predicted_burst_time=2.0),
            Process("B3", 0.0, 1.0, 1, predicted_burst_time=1.0),
        ]

        result_1 = meta.schedule(workload_1)
        result_2 = meta.schedule(workload_2)

        self.assertEqual(len(result_1.processes), 2)
        self.assertEqual(len(result_2.processes), 3)

        pids_1 = {p.pid for p in result_1.processes}
        pids_2 = {p.pid for p in result_2.processes}
        self.assertEqual(pids_1, {"A1", "A2"})
        self.assertEqual(pids_2, {"B1", "B2", "B3"})
        self.assertTrue(pids_1.isdisjoint(pids_2))


    def test_scheduler_kwargs_filtering_and_validation(self) -> None:
        clf_rr = StubWorkloadClassifier(CLASS_IO_BOUND)
        meta_rr = CPUMetaScheduler(classifier=clf_rr)


        result_rr = meta_rr.schedule(self.default_workload, time_quantum=3.5)
        self.assertEqual(result_rr.scheduler_kwargs_used, {"time_quantum": 3.5})


        with self.assertRaises(ValueError):
            meta_rr.schedule(self.default_workload, time_quantum=-1.0)


        clf_sjf = StubWorkloadClassifier(CLASS_CPU_BOUND)
        meta_sjf = CPUMetaScheduler(classifier=clf_sjf)
        result_sjf = meta_sjf.schedule(self.default_workload, time_quantum=3.5)
        self.assertNotIn("time_quantum", result_sjf.scheduler_kwargs_used)


class TestCPUMetaSchedulerIntegration(unittest.TestCase):
    """End-to-end integration test with trained ML models."""

    @classmethod
    def setUpClass(cls) -> None:

        cls.burst_predictor = CPUBurstPredictor(random_state=42)
        cls.burst_predictor.train_on_synthetic_workload(seed=42, num_processes=40)


        cls.classifier = CPUWorkloadClassifier(random_state=42)
        cls.classifier.train_on_synthetic_dataset(seed=42, target_per_class=50)

    def test_end_to_end_pipeline(self) -> None:
        """Process workload → burst predictor → classifier → meta-scheduler → classical scheduler → metrics."""
        meta = CPUMetaScheduler(
            classifier=self.classifier,
            burst_predictor=self.burst_predictor,
        )


        raw_workload = [
            Process(
                pid=f"Proc_{i}",
                arrival_time=float(i * 0.5),
                burst_time=float(10.0 + i * 2.0),
                priority=2,
                metadata={
                    "instruction_count": float(20000 + i * 1000),
                    "cpu_intensity": 0.85,
                    "io_wait_ratio": 0.15,
                    "process_complexity": 2.5,
                },
            )
            for i in range(12)
        ]


        for p in raw_workload:
            self.assertIsNone(p.predicted_burst_time)


        result = meta.schedule(raw_workload)


        self.assertIn(result.selected_scheduler, ["SJF", "Round Robin", "Priority", "FCFS"])
        self.assertIn(result.workload_class, [CLASS_CPU_BOUND, CLASS_IO_BOUND, CLASS_PRIORITY_CRITICAL, CLASS_BALANCED])
        self.assertGreaterEqual(result.confidence, 0.0)
        self.assertLessEqual(result.confidence, 1.0)


        self.assertEqual(len(result.processes), 12)
        self.assertGreater(len(result.gantt_chart), 0)
        self.assertEqual(result.metrics.completed_process_count, 12)
        self.assertGreater(result.metrics.cpu_utilization, 0.0)
        self.assertGreaterEqual(result.metrics.avg_waiting_time, 0.0)
        self.assertGreaterEqual(result.metrics.avg_turnaround_time, 0.0)


        for p in raw_workload:
            self.assertIsNone(p.predicted_burst_time)
            self.assertIsNone(p.start_time)
            self.assertIsNone(p.completion_time)


if __name__ == "__main__":
    unittest.main()
