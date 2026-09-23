"""Unit test suite for CP5.2 SharedMonitor."""

from __future__ import annotations

import os
import unittest

import psutil

from ml.cpu.burst_predictor import CPUBurstPredictor
from ml.cpu.workload_classifier import CPUWorkloadClassifier
from modules.cpu.action_loop import CPUActionLoop
from modules.cpu.meta_scheduler import CPUMetaScheduler
from modules.cpu.process import Process
from modules.device.backup import BackupResult
from modules.device.device import Device, DeviceType
from modules.device.predictive_action import PredictiveActionResult
from modules.device.recovery import RecoveryResult
from modules.memory.memory_monitor import MemoryMonitor
from modules.monitoring.shared_monitor import SharedMonitor


class TestSharedMonitor(unittest.TestCase):
    """Test suite validating SharedMonitor initialization, telemetry aggregation, and lifecycle."""

    @classmethod
    def setUpClass(cls) -> None:
        """Set up a CPUActionLoop for tests requiring CPU simulation."""
        predictor = CPUBurstPredictor(random_state=42)
        predictor.train_on_synthetic_workload(num_processes=50, seed=42)
        classifier = CPUWorkloadClassifier(random_state=42)
        classifier.train_on_synthetic_dataset(seed=42, target_per_class=10)
        meta = CPUMetaScheduler(classifier=classifier, burst_predictor=predictor)
        cls.cpu_action_loop = CPUActionLoop(meta_scheduler=meta)

    def setUp(self) -> None:
        self.cpu_action_loop.reset()

    def test_init_defaults(self) -> None:
        """SharedMonitor initializes with reasonable defaults when no components injected."""
        monitor = SharedMonitor()
        self.assertIsNone(monitor.cpu_action_loop)
        self.assertIsInstance(monitor.memory_monitor, MemoryMonitor)
        self.assertEqual(len(monitor.fleet), 4)
        self.assertEqual(len(monitor._recent_device_actions), 0)

    def test_init_custom(self) -> None:
        """SharedMonitor accepts custom injected components."""
        custom_fleet = [Device(device_id="custom_disk", device_type=DeviceType.DISK)]
        mem = MemoryMonitor()
        monitor = SharedMonitor(
            cpu_action_loop=self.cpu_action_loop,
            memory_monitor=mem,
            fleet=custom_fleet,
            max_recent_actions=10,
        )
        self.assertIs(monitor.cpu_action_loop, self.cpu_action_loop)
        self.assertIs(monitor.memory_monitor, mem)
        self.assertEqual(len(monitor.fleet), 1)
        self.assertEqual(monitor.max_recent_actions, 10)

    def test_get_system_snapshot_structure(self) -> None:
        """get_system_snapshot returns well-formed dictionary with all required subsystems."""
        monitor = SharedMonitor()
        snap = monitor.get_system_snapshot()

        self.assertIn("timestamp", snap)
        self.assertIn("status", snap)
        self.assertIn("cpu", snap)
        self.assertIn("memory", snap)
        self.assertIn("devices", snap)


        self.assertEqual(snap["status"], "ok")


        self.assertIn("ram_total_bytes", snap["memory"])
        self.assertIn("ram_percent", snap["memory"])
        self.assertIn("swap_percent", snap["memory"])


        self.assertEqual(snap["devices"]["device_count"], 4)
        self.assertEqual(len(snap["devices"]["devices"]), 4)
        self.assertEqual(snap["devices"]["recent_actions"], [])

    def test_get_system_snapshot_with_cpu_loop(self) -> None:
        """Snapshot accurately reflects CPUActionLoop execution and respects include_history."""
        monitor = SharedMonitor(cpu_action_loop=self.cpu_action_loop)


        snap1 = monitor.get_system_snapshot(include_history=False)
        self.assertEqual(snap1["cpu"]["total_steps"], 0)
        self.assertNotIn("history", snap1["cpu"])


        proc = Process(pid="P1", arrival_time=0.0, burst_time=10.0)
        self.cpu_action_loop.step([proc])


        snap2 = monitor.get_system_snapshot(include_history=False)
        self.assertEqual(snap2["cpu"]["total_steps"], 1)
        self.assertNotIn("history", snap2["cpu"])


        snap3 = monitor.get_system_snapshot(include_history=True)
        self.assertEqual(snap3["cpu"]["total_steps"], 1)
        self.assertIn("history", snap3["cpu"])
        self.assertEqual(len(snap3["cpu"]["history"]), 1)
        self.assertEqual(snap3["cpu"]["history"][0]["process_count"], 1)

    def test_get_system_snapshot_with_pid(self) -> None:
        """Per-process memory metrics populated when valid PID supplied."""
        monitor = SharedMonitor()
        my_pid = os.getpid()
        snap = monitor.get_system_snapshot(pid=my_pid)
        self.assertEqual(snap["memory"]["proc_pid"], my_pid)
        self.assertIsNotNone(snap["memory"]["proc_rss_bytes"])

    def test_get_system_snapshot_invalid_pid(self) -> None:
        """Non-existent PID raises NoSuchProcess."""
        monitor = SharedMonitor()
        with self.assertRaises(psutil.NoSuchProcess):
            monitor.get_system_snapshot(pid=99999999)

    def test_device_telemetry_in_snapshot(self) -> None:
        """Each device entry contains base device state and generated telemetry."""
        monitor = SharedMonitor()
        snap = monitor.get_system_snapshot()
        devices = snap["devices"]["devices"]

        for d in devices:
            self.assertIn("device_id", d)
            self.assertIn("device_type", d)
            self.assertIn("status", d)
            self.assertIn("telemetry", d)
            telem = d["telemetry"]
            self.assertEqual(telem["device_id"], d["device_id"])
            self.assertIn("temperature", telem)
            self.assertIn("error_rate", telem)
            self.assertIn("response_latency", telem)
            self.assertIn("queue_length", telem)
            self.assertIn("io_activity", telem)

    def test_record_device_action(self) -> None:
        """record_device_action stores recent predictive action results up to max_recent_actions."""
        monitor = SharedMonitor(max_recent_actions=2)


        res1 = PredictiveActionResult(
            device_id="disk_0",
            failure_probability=0.85,
            backup_result=BackupResult(
                device_id="disk_0",
                triggered=True,
                success=True,
                backup_location="s3://backup",
                message="Backup created",
            ),
            recovery_result=RecoveryResult(
                device_id="disk_0",
                recovered=True,
                io_blocked=True,
                rerouted=True,
                backup_device_id="disk_backup",
                message="I/O rerouted",
            ),
        )
        monitor.record_device_action(res1)
        snap1 = monitor.get_system_snapshot()
        self.assertEqual(len(snap1["devices"]["recent_actions"]), 1)
        self.assertEqual(snap1["devices"]["recent_actions"][0]["device_id"], "disk_0")


        res2 = {"device_id": "disk_1", "failure_probability": 0.9}
        monitor.record_device_action(res2)
        snap2 = monitor.get_system_snapshot()
        self.assertEqual(len(snap2["devices"]["recent_actions"]), 2)


        res3 = {"device_id": "disk_2", "failure_probability": 0.95}
        monitor.record_device_action(res3)
        snap3 = monitor.get_system_snapshot()
        self.assertEqual(len(snap3["devices"]["recent_actions"]), 2)
        self.assertEqual(snap3["devices"]["recent_actions"][0]["device_id"], "disk_1")
        self.assertEqual(snap3["devices"]["recent_actions"][1]["device_id"], "disk_2")

    def test_record_device_action_invalid_type(self) -> None:
        """record_device_action rejects invalid types."""
        monitor = SharedMonitor()
        with self.assertRaises(TypeError):
            monitor.record_device_action("invalid_string_not_dict_or_dataclass")

    def test_overall_status_neutral(self) -> None:
        """Overall status reflects neutral 'ok' unless a device is explicitly in an unhealthy state."""

        m1 = SharedMonitor()
        self.assertEqual(m1.get_system_snapshot()["status"], "ok")


        degraded_fleet = [
            Device(device_id="disk_0", device_type=DeviceType.DISK, status="degrading"),
            Device(device_id="NIC_0", device_type=DeviceType.NIC, status="ok"),
        ]
        m2 = SharedMonitor(fleet=degraded_fleet)
        self.assertEqual(m2.get_system_snapshot()["status"], "degraded")


        failed_fleet = [
            Device(device_id="disk_0", device_type=DeviceType.DISK, status="failed"),
        ]
        m3 = SharedMonitor(fleet=failed_fleet)
        self.assertEqual(m3.get_system_snapshot()["status"], "degraded")

    def test_reset(self) -> None:
        """reset clears both CPU action history and recorded device actions."""
        monitor = SharedMonitor(cpu_action_loop=self.cpu_action_loop)
        self.cpu_action_loop.step([Process(pid="P1", arrival_time=0.0, burst_time=5.0)])
        monitor.record_device_action({"device_id": "disk_0", "test": True})

        self.assertEqual(len(monitor._recent_device_actions), 1)
        self.assertEqual(self.cpu_action_loop.get_summary()["total_steps"], 1)

        monitor.reset()
        self.assertEqual(len(monitor._recent_device_actions), 0)
        self.assertEqual(self.cpu_action_loop.get_summary()["total_steps"], 0)


if __name__ == "__main__":
    unittest.main()
