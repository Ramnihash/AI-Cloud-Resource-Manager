"""Unit tests for CPU AI Action Loop simulation engine (CP4.1)."""

from __future__ import annotations

import copy
import unittest
from dataclasses import FrozenInstanceError

from data.cpu.generator import CPUWorkloadGenerator, WorkloadConfig
from ml.cpu.burst_predictor import CPUBurstPredictor
from ml.cpu.workload_classifier import CPUWorkloadClassifier
from modules.cpu.action_loop import ActionLoopStep, CPUActionLoop
from modules.cpu.base_scheduler import BaseScheduler, ScheduleResult
from modules.cpu.fcfs import FCFSScheduler
from modules.cpu.meta_scheduler import CPUMetaScheduler, MetaScheduleResult
from modules.cpu.process import Process
from modules.cpu.round_robin import RoundRobinScheduler
from modules.cpu.sjf import SJFScheduler


class TestCPUActionLoop(unittest.TestCase):
    """Test suite validating the CPUActionLoop orchestration engine."""

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

    def _create_sample_process(
        self,
        pid: str,
        arrival: float = 0.0,
        burst: float = 10.0,
        priority: int = 1,
        cpu_intensity: float = 0.8,
        io_wait_ratio: float = 0.1,
    ) -> Process:
        """Helper to create test processes with required metadata."""
        return Process(
            pid=pid,
            arrival_time=arrival,
            burst_time=burst,
            priority=priority,
            metadata={
                "instruction_count": burst * 1000.0,
                "cpu_intensity": cpu_intensity,
                "io_wait_ratio": io_wait_ratio,
                "process_complexity": 1.5,
            },
        )

    def _create_fallback_batch(self) -> list[Process]:
        """Generate a 20-process boundary batch with confidence < 0.40."""
        bursts = [
            8.0, 8.5, 9.0, 9.5, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
            16.0, 17.0, 18.0, 19.0, 20.0, 20.5, 21.0, 21.5, 22.0, 22.5,
        ]
        processes: list[Process] = []
        for i in range(20):
            b = bursts[i]
            pri = 1 if i < 9 else 4
            cpu = 0.715
            io = 0.552
            comp = 2.5
            ic = b * 2500.0 * (1.0 + io) / (comp * cpu)
            processes.append(
                Process(
                    pid=f"FB_{i}",
                    arrival_time=float(i * 0.5),
                    burst_time=b,
                    priority=pri,
                    metadata={
                        "cpu_intensity": cpu,
                        "io_wait_ratio": io,
                        "instruction_count": ic,
                        "process_complexity": comp,
                    },
                )
            )
        return processes


    def test_initialization_defaults_and_guards(self) -> None:
        loop = CPUActionLoop(meta_scheduler=self.meta_scheduler)
        self.assertIsInstance(loop.baseline_scheduler, FCFSScheduler)
        self.assertEqual(len(loop.get_history()), 0)

        custom_baseline = SJFScheduler()
        loop_custom = CPUActionLoop(
            meta_scheduler=self.meta_scheduler,
            baseline_scheduler=custom_baseline,
        )
        self.assertIs(loop_custom.baseline_scheduler, custom_baseline)


        with self.assertRaises(ValueError):
            CPUActionLoop(meta_scheduler=None)

        with self.assertRaises(TypeError):
            CPUActionLoop(meta_scheduler="invalid")

        with self.assertRaises(TypeError):
            CPUActionLoop(
                meta_scheduler=self.meta_scheduler,
                baseline_scheduler="invalid",
            )


    def test_single_step_execution(self) -> None:
        loop = CPUActionLoop(meta_scheduler=self.meta_scheduler)
        batch = [
            self._create_sample_process("P1", burst=25.0, cpu_intensity=0.9, io_wait_ratio=0.05),
            self._create_sample_process("P2", burst=35.0, cpu_intensity=0.85, io_wait_ratio=0.08),
            self._create_sample_process("P3", burst=20.0, cpu_intensity=0.92, io_wait_ratio=0.04),
        ]

        step = loop.step(batch)
        self.assertIsInstance(step, ActionLoopStep)
        self.assertEqual(step.step_index, 0)
        self.assertEqual(step.process_count, 3)
        self.assertIsInstance(step.meta_result, MetaScheduleResult)
        self.assertIsNone(step.baseline_result)
        self.assertIsNone(step.action_delta)


        self.assertIn(step.selected_scheduler, ["SJF", "Round Robin", "Priority", "FCFS"])
        self.assertIn(step.workload_class, ["CPU_BOUND", "IO_BOUND", "PRIORITY_CRITICAL", "BALANCED"])
        self.assertFalse(step.is_fallback)
        self.assertGreaterEqual(step.confidence, 0.40)
        self.assertGreaterEqual(step.metrics.avg_waiting_time, 0.0)
        self.assertGreaterEqual(step.metrics.avg_turnaround_time, 0.0)
        self.assertGreater(step.metrics.cpu_utilization, 0.0)
        self.assertEqual(len(step.processes), 3)
        self.assertGreater(len(step.gantt_chart), 0)


        d = step.to_dict()
        self.assertEqual(d["step_index"], 0)
        self.assertEqual(d["process_count"], 3)
        self.assertIn("meta_result", d)
        self.assertIsNone(d["baseline_result"])
        self.assertIsNone(d["action_delta"])


    def test_multiple_workload_steps_and_run(self) -> None:
        loop = CPUActionLoop(meta_scheduler=self.meta_scheduler)
        b1 = [self._create_sample_process("B1_1", burst=15.0)]
        b2 = [self._create_sample_process("B2_1", burst=12.0), self._create_sample_process("B2_2", burst=8.0)]
        b3 = [self._create_sample_process("B3_1", burst=20.0)]

        steps = loop.run([b1, b2, b3])
        self.assertEqual(len(steps), 3)
        self.assertEqual(steps[0].step_index, 0)
        self.assertEqual(steps[1].step_index, 1)
        self.assertEqual(steps[2].step_index, 2)
        self.assertEqual(steps[0].process_count, 1)
        self.assertEqual(steps[1].process_count, 2)
        self.assertEqual(steps[2].process_count, 1)

        history = loop.get_history()
        self.assertEqual(len(history), 3)
        self.assertEqual([s.step_index for s in history], [0, 1, 2])

        with self.assertRaises(ValueError):
            loop.run(None)


    def test_history_isolation(self) -> None:
        loop = CPUActionLoop(meta_scheduler=self.meta_scheduler)
        batch = [self._create_sample_process("P1")]
        loop.step(batch)

        history = loop.get_history()
        self.assertEqual(len(history), 1)


        history.clear()
        self.assertEqual(len(loop.get_history()), 1)


        step = loop.get_history()[0]
        with self.assertRaises(FrozenInstanceError):
            step.step_index = 999


    def test_reset(self) -> None:
        loop = CPUActionLoop(meta_scheduler=self.meta_scheduler)
        b1 = [self._create_sample_process("P1")]
        b2 = [self._create_sample_process("P2")]
        loop.run([b1, b2])
        self.assertEqual(len(loop.get_history()), 2)

        loop.reset()
        self.assertEqual(len(loop.get_history()), 0)


        next_step = loop.step(b1)
        self.assertEqual(next_step.step_index, 0)
        self.assertEqual(len(loop.get_history()), 1)


    def test_fallback_propagation(self) -> None:
        loop = CPUActionLoop(meta_scheduler=self.meta_scheduler)
        fb_batch = self._create_fallback_batch()

        step = loop.step(fb_batch)
        self.assertTrue(step.is_fallback)
        self.assertLess(step.confidence, 0.40)
        self.assertEqual(step.workload_class, "BALANCED")
        self.assertEqual(step.selected_scheduler, "FCFS")


    def test_optional_baseline_comparison(self) -> None:
        loop = CPUActionLoop(meta_scheduler=self.meta_scheduler)
        batch = [
            self._create_sample_process("P1", burst=10.0, arrival=0.0),
            self._create_sample_process("P2", burst=20.0, arrival=1.0),
        ]


        step_no_base = loop.step(batch, compare_with_baseline=False)
        self.assertIsNone(step_no_base.baseline_result)
        self.assertIsNone(step_no_base.action_delta)


        step_base = loop.step(batch, compare_with_baseline=True)
        self.assertIsNotNone(step_base.baseline_result)
        self.assertIsInstance(step_base.baseline_result, ScheduleResult)
        self.assertEqual(step_base.baseline_result.algorithm_name, "FCFS")

        self.assertIsNotNone(step_base.action_delta)
        delta = step_base.action_delta
        self.assertIn("waiting_time_delta", delta)
        self.assertIn("turnaround_time_delta", delta)
        self.assertIn("response_time_delta", delta)


        meta_m = step_base.metrics
        base_m = step_base.baseline_result.metrics
        self.assertAlmostEqual(delta["waiting_time_delta"], meta_m.avg_waiting_time - base_m.avg_waiting_time, places=4)
        self.assertAlmostEqual(delta["turnaround_time_delta"], meta_m.avg_turnaround_time - base_m.avg_turnaround_time, places=4)
        self.assertAlmostEqual(delta["response_time_delta"], meta_m.avg_response_time - base_m.avg_response_time, places=4)


        rr_loop = CPUActionLoop(
            meta_scheduler=self.meta_scheduler,
            baseline_scheduler=RoundRobinScheduler(time_quantum=2.0),
        )
        step_rr = rr_loop.step(batch, compare_with_baseline=True)
        self.assertEqual(step_rr.baseline_result.algorithm_name, "Round Robin")


    def test_workload_immutability(self) -> None:
        loop = CPUActionLoop(meta_scheduler=self.meta_scheduler)
        original_processes = [
            self._create_sample_process("P1", burst=15.0, arrival=0.0),
            self._create_sample_process("P2", burst=25.0, arrival=2.0),
        ]
        processes_input = [p.copy() for p in original_processes]

        step = loop.step(processes_input, compare_with_baseline=True)


        self.assertEqual(len(processes_input), len(original_processes))
        for orig, current in zip(original_processes, processes_input):
            self.assertEqual(orig.pid, current.pid)
            self.assertEqual(orig.burst_time, current.burst_time)
            self.assertEqual(orig.arrival_time, current.arrival_time)
            self.assertIsNone(current.start_time)
            self.assertIsNone(current.completion_time)
            self.assertEqual(current.waiting_time, 0.0)
            self.assertEqual(current.turnaround_time, 0.0)
            self.assertEqual(current.response_time, 0.0)


        for p in step.processes:
            self.assertIsNotNone(p.start_time)
            self.assertIsNotNone(p.completion_time)


    def test_invalid_inputs_and_unfitted_guards(self) -> None:
        loop = CPUActionLoop(meta_scheduler=self.meta_scheduler)


        with self.assertRaises(ValueError):
            loop.step([])


        with self.assertRaises(TypeError):
            loop.step(["not_a_process"])


        unfitted_clf = CPUWorkloadClassifier(random_state=42)
        unfitted_meta = CPUMetaScheduler(
            classifier=unfitted_clf,
            burst_predictor=self.predictor,
        )
        unfitted_loop = CPUActionLoop(meta_scheduler=unfitted_meta)
        batch = [self._create_sample_process("P1")]
        with self.assertRaises(RuntimeError):
            unfitted_loop.step(batch)


    def test_get_summary(self) -> None:
        loop = CPUActionLoop(meta_scheduler=self.meta_scheduler)


        empty_sum = loop.get_summary()
        self.assertEqual(empty_sum["total_steps"], 0)
        self.assertEqual(empty_sum["total_processes"], 0)
        self.assertEqual(empty_sum["fallback_count"], 0)
        self.assertIsNone(empty_sum["mean_action_delta"])


        b1 = [self._create_sample_process("P1", burst=20.0), self._create_sample_process("P2", burst=10.0)]
        b2 = [self._create_sample_process("P3", burst=5.0)]
        fb = self._create_fallback_batch()

        loop.step(b1, compare_with_baseline=True)
        loop.step(b2, compare_with_baseline=True)
        loop.step(fb, compare_with_baseline=False)

        summary = loop.get_summary()
        self.assertEqual(summary["total_steps"], 3)
        self.assertEqual(summary["total_processes"], 2 + 1 + 20)
        self.assertIn("scheduler_distribution", summary)
        self.assertIn("scheduler_percentages", summary)
        self.assertEqual(summary["fallback_count"], 1)
        self.assertAlmostEqual(summary["fallback_rate"], 1.0 / 3.0, places=4)

        mean_metrics = summary["mean_metrics"]
        self.assertIn("avg_waiting_time", mean_metrics)
        self.assertIn("avg_turnaround_time", mean_metrics)
        self.assertIn("avg_response_time", mean_metrics)
        self.assertIn("cpu_utilization", mean_metrics)

        action_delta = summary["mean_action_delta"]
        self.assertIsNotNone(action_delta)
        self.assertIn("mean_waiting_time_delta", action_delta)
        self.assertIn("mean_turnaround_time_delta", action_delta)
        self.assertIn("mean_response_time_delta", action_delta)


if __name__ == "__main__":
    unittest.main()
