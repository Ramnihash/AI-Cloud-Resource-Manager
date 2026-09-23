"""Unit test suite for CP5.3 IntegratedExecutionManager."""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

import psutil

from ml.cpu.burst_predictor import CPUBurstPredictor
from ml.cpu.workload_classifier import CPUWorkloadClassifier
from modules.cpu.action_loop import CPUActionLoop
from modules.cpu.meta_scheduler import CPUMetaScheduler
from modules.cpu.process import Process
from modules.device.device import Device, DeviceType
from modules.integration.integrated_execution import (
    IntegratedExecutionManager,
    IntegratedExecutionStep,
)
from modules.memory.memory_monitor import MemoryMonitor
from modules.monitoring.shared_monitor import SharedMonitor


class TestIntegratedExecutionManager(unittest.TestCase):
    """Test suite validating IntegratedExecutionManager coordination, consistency, and lifecycle."""

    @classmethod
    def setUpClass(cls) -> None:
        """Initialize trained models and CPUActionLoop for fast, deterministic testing."""
        predictor = CPUBurstPredictor(random_state=42)
        predictor.train_on_synthetic_workload(num_processes=50, seed=42)
        classifier = CPUWorkloadClassifier(random_state=42)
        classifier.train_on_synthetic_dataset(seed=42, target_per_class=10)
        meta = CPUMetaScheduler(classifier=classifier, burst_predictor=predictor)
        cls.cpu_action_loop = CPUActionLoop(meta_scheduler=meta)

    def setUp(self) -> None:
        self.cpu_action_loop.reset()

    def test_manager_initialization_defaults(self) -> None:
        """Manager initializes cleanly with default components and empty history."""
        manager = IntegratedExecutionManager()
        self.assertIsInstance(manager.memory_monitor, MemoryMonitor)
        self.assertIsInstance(manager.shared_monitor, SharedMonitor)
        self.assertEqual(len(manager.get_history()), 0)

    def test_injected_component_consistency(self) -> None:
        """Manager and SharedMonitor strictly share the same CPU and Memory instances."""
        mem = MemoryMonitor()
        shared = SharedMonitor(cpu_action_loop=self.cpu_action_loop, memory_monitor=mem)
        manager = IntegratedExecutionManager(
            cpu_action_loop=self.cpu_action_loop,
            memory_monitor=mem,
            shared_monitor=shared,
        )


        self.assertIs(manager.cpu_action_loop, self.cpu_action_loop)
        self.assertIs(manager.shared_monitor.cpu_action_loop, self.cpu_action_loop)
        self.assertIs(manager.cpu_action_loop, manager.shared_monitor.cpu_action_loop)

        self.assertIs(manager.memory_monitor, mem)
        self.assertIs(manager.shared_monitor.memory_monitor, mem)
        self.assertIs(manager.memory_monitor, manager.shared_monitor.memory_monitor)

    def test_injected_component_reconciliation_when_shared_monitor_provided(self) -> None:
        """When shared_monitor is provided without cpu_action_loop, manager adopts it."""
        shared = SharedMonitor(cpu_action_loop=self.cpu_action_loop)
        manager = IntegratedExecutionManager(shared_monitor=shared)
        self.assertIs(manager.cpu_action_loop, self.cpu_action_loop)
        self.assertIs(manager.cpu_action_loop, manager.shared_monitor.cpu_action_loop)

    def test_cpu_only_coordinated_step(self) -> None:
        """A coordinated step with process workload only populates CPU, Memory, and empty devices."""
        manager = IntegratedExecutionManager(cpu_action_loop=self.cpu_action_loop)
        procs = [
            Process(pid="P1", arrival_time=0.0, burst_time=8.0),
            Process(pid="P2", arrival_time=1.0, burst_time=4.0),
        ]
        step = manager.step(processes=procs, compare_with_baseline=False)

        self.assertIsInstance(step, IntegratedExecutionStep)
        self.assertEqual(step.step_index, 0)
        self.assertIn("timestamp", step.to_dict())


        self.assertEqual(step.cpu_result["process_count"], 2)
        self.assertIn("selected_scheduler", step.cpu_result)
        self.assertIn("schedule_result", step.cpu_result["meta_result"])
        self.assertIn("metrics", step.cpu_result["meta_result"]["schedule_result"])


        self.assertIn("ram_percent", step.memory_result)
        self.assertIn("swap_percent", step.memory_result)


        self.assertEqual(step.device_results, [])


        self.assertEqual(step.system_snapshot["cpu"]["total_steps"], 1)
        self.assertEqual(step.system_snapshot["status"], "ok")

    def test_cpu_and_memory_observation_with_pid(self) -> None:
        """Supplying valid PID observes process-level memory in step result."""
        manager = IntegratedExecutionManager(cpu_action_loop=self.cpu_action_loop)
        my_pid = os.getpid()
        procs = [Process(pid="P1", arrival_time=0.0, burst_time=5.0)]
        step = manager.step(processes=procs, pid=my_pid)

        self.assertEqual(step.memory_result["proc_pid"], my_pid)
        self.assertIsNotNone(step.memory_result["proc_rss_bytes"])
        self.assertEqual(step.system_snapshot["memory"]["proc_pid"], my_pid)

    def test_high_risk_device_action(self) -> None:
        """High failure-probability device action triggers backup and recovery."""
        manager = IntegratedExecutionManager(cpu_action_loop=self.cpu_action_loop)
        procs = [Process(pid="P1", arrival_time=0.0, burst_time=5.0)]
        actions = [
            {
                "device_id": "disk_0",
                "failure_probability": 0.85,
                "device_status": "degrading",
                "backup_device_id": "disk_backup_0",
                "threshold": 0.7,
            }
        ]
        step = manager.step(processes=procs, device_actions=actions)

        self.assertEqual(len(step.device_results), 1)
        dev_res = step.device_results[0]
        self.assertEqual(dev_res["device_id"], "disk_0")
        self.assertTrue(dev_res["backup_result"]["triggered"])
        self.assertTrue(dev_res["backup_result"]["success"])
        self.assertTrue(dev_res["recovery_result"]["recovered"])
        self.assertTrue(dev_res["recovery_result"]["rerouted"])


        self.assertEqual(len(step.system_snapshot["devices"]["recent_actions"]), 1)
        self.assertEqual(
            step.system_snapshot["devices"]["recent_actions"][0]["device_id"],
            "disk_0",
        )

    def test_low_risk_device_action(self) -> None:
        """Low failure-probability device action does not trigger backup."""
        manager = IntegratedExecutionManager(cpu_action_loop=self.cpu_action_loop)
        procs = [Process(pid="P1", arrival_time=0.0, burst_time=5.0)]
        actions = [
            {
                "device_id": "disk_0",
                "failure_probability": 0.2,
                "device_status": "healthy",
                "threshold": 0.7,
            }
        ]
        step = manager.step(processes=procs, device_actions=actions)

        self.assertEqual(len(step.device_results), 1)
        dev_res = step.device_results[0]
        self.assertFalse(dev_res["backup_result"]["triggered"])
        self.assertFalse(dev_res["recovery_result"]["recovered"])

    def test_compare_with_baseline(self) -> None:
        """compare_with_baseline=True populates CPU baseline_result and action_delta."""
        manager = IntegratedExecutionManager(cpu_action_loop=self.cpu_action_loop)
        procs = [Process(pid="P1", arrival_time=0.0, burst_time=12.0)]
        step = manager.step(processes=procs, compare_with_baseline=True)

        self.assertIsNotNone(step.cpu_result["baseline_result"])
        self.assertEqual(step.cpu_result["baseline_result"]["algorithm_name"], "FCFS")
        self.assertIsNotNone(step.cpu_result["action_delta"])
        self.assertIn("waiting_time_delta", step.cpu_result["action_delta"])

    def test_multiple_execution_steps_and_history(self) -> None:
        """Consecutive steps advance step_index, accumulate history, and maintain telemetry."""
        manager = IntegratedExecutionManager(cpu_action_loop=self.cpu_action_loop)
        p1 = [Process(pid="P1", arrival_time=0.0, burst_time=5.0)]
        p2 = [Process(pid="P2", arrival_time=0.0, burst_time=10.0)]

        s0 = manager.step(p1)
        s1 = manager.step(p2)

        self.assertEqual(s0.step_index, 0)
        self.assertEqual(s1.step_index, 1)

        history = manager.get_history()
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0].step_index, 0)
        self.assertEqual(history[1].step_index, 1)


        history.clear()
        self.assertEqual(len(manager.get_history()), 2)

    def test_reset_behavior(self) -> None:
        """reset clears history and resets CPU and SharedMonitor states exactly once."""
        manager = IntegratedExecutionManager(cpu_action_loop=self.cpu_action_loop)
        procs = [Process(pid="P1", arrival_time=0.0, burst_time=5.0)]
        actions = [
            {
                "device_id": "disk_0",
                "failure_probability": 0.8,
                "device_status": "degrading",
            }
        ]
        manager.step(processes=procs, device_actions=actions)

        self.assertEqual(len(manager.get_history()), 1)
        self.assertEqual(self.cpu_action_loop.get_summary()["total_steps"], 1)
        self.assertEqual(len(manager.shared_monitor._recent_device_actions), 1)

        manager.reset()
        self.assertEqual(len(manager.get_history()), 0)
        self.assertEqual(self.cpu_action_loop.get_summary()["total_steps"], 0)
        self.assertEqual(len(manager.shared_monitor._recent_device_actions), 0)

    def test_empty_or_invalid_processes(self) -> None:
        """Empty process list or non-Process items raise ValueError / TypeError."""
        manager = IntegratedExecutionManager(cpu_action_loop=self.cpu_action_loop)
        with self.assertRaises(ValueError):
            manager.step(processes=[])

        with self.assertRaises(TypeError):
            manager.step(processes="not_a_list")

        with self.assertRaises(TypeError):
            manager.step(processes=[{"pid": "P1", "burst_time": 5.0}])

        with self.assertRaises(ValueError):
            manager.step(processes=[Process(pid="P1", arrival_time=0.0, burst_time=-5.0)])

    def test_invalid_device_actions(self) -> None:
        """Malformed device_actions items raise ValueError or TypeError."""
        manager = IntegratedExecutionManager(cpu_action_loop=self.cpu_action_loop)
        procs = [Process(pid="P1", arrival_time=0.0, burst_time=5.0)]


        with self.assertRaises(TypeError):
            manager.step(processes=procs, device_actions="not_a_list")


        with self.assertRaises(ValueError):
            manager.step(
                processes=procs,
                device_actions=[{"device_id": "disk_0", "failure_probability": 0.5}],
            )


        with self.assertRaises(ValueError):
            manager.step(
                processes=procs,
                device_actions=[
                    {"device_id": "disk_0", "failure_probability": 1.5, "device_status": "ok"}
                ],
            )


        with self.assertRaises(ValueError):
            manager.step(
                processes=procs,
                device_actions=[
                    {"device_id": "   ", "failure_probability": 0.5, "device_status": "ok"}
                ],
            )

    def test_invalid_pid(self) -> None:
        """Invalid PID types or negative numbers raise ValueError; non-existent PID raises NoSuchProcess."""
        manager = IntegratedExecutionManager(cpu_action_loop=self.cpu_action_loop)
        procs = [Process(pid="P1", arrival_time=0.0, burst_time=5.0)]

        with self.assertRaises(ValueError):
            manager.step(processes=procs, pid=-1)

        with self.assertRaises(ValueError):
            manager.step(processes=procs, pid="invalid")

        with self.assertRaises(psutil.NoSuchProcess):
            manager.step(processes=procs, pid=99999999)

    def test_unexpected_exception_propagation(self) -> None:
        """Unexpected internal exceptions propagate cleanly for route-level 500 handling."""
        manager = IntegratedExecutionManager(cpu_action_loop=self.cpu_action_loop)
        procs = [Process(pid="P1", arrival_time=0.0, burst_time=5.0)]
        with patch.object(
            self.cpu_action_loop, "step", side_effect=RuntimeError("Internal scheduler bus fault")
        ):
            with self.assertRaises(RuntimeError):
                manager.step(processes=procs)


if __name__ == "__main__":
    unittest.main()
