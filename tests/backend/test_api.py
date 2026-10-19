"""Unit test suite for CP5.1 Common Backend Interface."""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from backend.app import create_app
from ml.cpu.burst_predictor import CPUBurstPredictor
from ml.cpu.workload_classifier import CPUWorkloadClassifier
from modules.cpu.action_loop import CPUActionLoop
from modules.cpu.meta_scheduler import CPUMetaScheduler


class TestBackendAPI(unittest.TestCase):
    """Test suite validating all backend REST endpoints and error handlers."""

    @classmethod
    def setUpClass(cls) -> None:
        """Set up a single trained CPUActionLoop instance for fast and deterministic test runs."""
        predictor = CPUBurstPredictor(random_state=42)
        predictor.train_on_synthetic_workload(num_processes=50, seed=42)

        classifier = CPUWorkloadClassifier(random_state=42)
        classifier.train_on_synthetic_dataset(seed=42, target_per_class=10)

        meta = CPUMetaScheduler(classifier=classifier, burst_predictor=predictor)
        cls.cpu_action_loop = CPUActionLoop(meta_scheduler=meta)

        cls.app = create_app(
            cpu_action_loop=cls.cpu_action_loop,
            test_config={"TESTING": True},
        )
        cls.client = cls.app.test_client()

    def setUp(self) -> None:
        self.cpu_action_loop.reset()
        mgr = self.app.config.get("INTEGRATED_EXECUTION_MANAGER")
        if mgr is not None:
            mgr.reset()
        else:
            mon = self.app.config.get("SHARED_MONITOR")
            if mon is not None:
                mon.reset()





    def test_health(self) -> None:
        """GET /health returns HTTP 200 with status ok."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.is_json)
        data = response.get_json()
        self.assertEqual(data, {"status": "ok"})




    def test_cpu_action_valid_workload(self) -> None:
        """POST /api/v1/cpu/action successfully runs a valid workload."""
        payload = {
            "processes": [
                {
                    "pid": "P1",
                    "arrival_time": 0.0,
                    "burst_time": 12.0,
                    "priority": 1,
                    "metadata": {
                        "instruction_count": 12000.0,
                        "cpu_intensity": 0.85,
                        "io_wait_ratio": 0.1,
                        "process_complexity": 1.5,
                    },
                },
                {
                    "pid": "P2",
                    "arrival_time": 1.0,
                    "burst_time": 8.0,
                    "priority": 2,
                    "metadata": {
                        "instruction_count": 8000.0,
                        "cpu_intensity": 0.80,
                        "io_wait_ratio": 0.15,
                        "process_complexity": 1.2,
                    },
                },
            ],
            "compare_with_baseline": False,
        }
        response = self.client.post("/api/v1/cpu/action", json=payload)
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.is_json)
        data = response.get_json()

        self.assertEqual(data["step_index"], 0)
        self.assertEqual(data["process_count"], 2)
        self.assertIn("selected_scheduler", data)
        self.assertIn("workload_class", data)
        self.assertIn("meta_result", data)
        self.assertIsNone(data["baseline_result"])
        self.assertIsNone(data["action_delta"])

    def test_cpu_action_with_baseline_comparison(self) -> None:
        """POST /api/v1/cpu/action populates baseline and delta when enabled."""
        payload = {
            "processes": [
                {
                    "pid": 1,
                    "arrival_time": 0.0,
                    "burst_time": 15.0,
                    "metadata": {
                        "instruction_count": 15000.0,
                        "cpu_intensity": 0.9,
                        "io_wait_ratio": 0.05,
                        "process_complexity": 1.8,
                    },
                }
            ],
            "compare_with_baseline": True,
        }
        response = self.client.post("/api/v1/cpu/action", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.get_json()

        self.assertIsNotNone(data["baseline_result"])
        self.assertEqual(data["baseline_result"]["algorithm_name"], "FCFS")
        self.assertIsNotNone(data["action_delta"])
        self.assertIn("waiting_time_delta", data["action_delta"])
        self.assertIn("turnaround_time_delta", data["action_delta"])
        self.assertIn("response_time_delta", data["action_delta"])




    def test_cpu_action_invalid_requests(self) -> None:
        """POST /api/v1/cpu/action rejects malformed workloads with HTTP 400."""

        r1 = self.client.post("/api/v1/cpu/action", json={"compare_with_baseline": False})
        self.assertEqual(r1.status_code, 400)
        self.assertIn("error", r1.get_json())


        r2 = self.client.post("/api/v1/cpu/action", json={"processes": []})
        self.assertEqual(r2.status_code, 400)
        self.assertIn("error", r2.get_json())


        r3 = self.client.post("/api/v1/cpu/action", json={"processes": "not_a_list"})
        self.assertEqual(r3.status_code, 400)
        self.assertIn("error", r3.get_json())


        r4 = self.client.post("/api/v1/cpu/action", json={"processes": [{"pid": "P1", "arrival_time": 0.0}]})
        self.assertEqual(r4.status_code, 400)
        self.assertIn("error", r4.get_json())


        r5 = self.client.post(
            "/api/v1/cpu/action",
            json={"processes": [{"pid": "P1", "arrival_time": 0.0, "burst_time": -5.0}]},
        )
        self.assertEqual(r5.status_code, 400)
        self.assertIn("error", r5.get_json())


        r6 = self.client.post(
            "/api/v1/cpu/action",
            json={"processes": [{"pid": "P1", "arrival_time": 0.0, "burst_time": 5.0}], "compare_with_baseline": "yes"},
        )
        self.assertEqual(r6.status_code, 400)
        self.assertIn("error", r6.get_json())




    def test_memory_page_replacement_fifo(self) -> None:
        """POST /api/v1/memory/page-replacement evaluates FIFO correctly."""
        payload = {
            "algorithm": "fifo",
            "num_frames": 3,
            "reference_string": [1, 2, 3, 4, 1, 2, 5, 1, 2, 3, 4, 5],
        }
        response = self.client.post("/api/v1/memory/page-replacement", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["algorithm"], "FIFO")
        self.assertEqual(data["num_frames"], 3)
        self.assertEqual(data["page_faults"], 9)
        self.assertEqual(data["page_hits"], 3)
        self.assertIn("frame_snapshots", data)
        self.assertIn("hit_ratio", data)
        self.assertIn("miss_ratio", data)

    def test_memory_page_replacement_lru(self) -> None:
        """POST /api/v1/memory/page-replacement evaluates LRU correctly."""
        payload = {
            "algorithm": "lru",
            "num_frames": 3,
            "reference_string": [1, 2, 3, 4, 1, 2, 5, 1, 2, 3, 4, 5],
        }
        response = self.client.post("/api/v1/memory/page-replacement", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["algorithm"], "LRU")
        self.assertEqual(data["page_faults"], 10)
        self.assertEqual(data["page_hits"], 2)

    def test_memory_page_replacement_optimal(self) -> None:
        """POST /api/v1/memory/page-replacement evaluates Optimal correctly."""
        payload = {
            "algorithm": "optimal",
            "num_frames": 3,
            "reference_string": [1, 2, 3, 4, 1, 2, 5, 1, 2, 3, 4, 5],
        }
        response = self.client.post("/api/v1/memory/page-replacement", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["algorithm"], "Optimal")
        self.assertEqual(data["page_faults"], 7)
        self.assertEqual(data["page_hits"], 5)




    def test_memory_page_replacement_invalid(self) -> None:
        """POST /api/v1/memory/page-replacement rejects invalid algorithm and parameters."""

        r1 = self.client.post(
            "/api/v1/memory/page-replacement",
            json={"algorithm": "clock", "num_frames": 3, "reference_string": [1, 2, 3]},
        )
        self.assertEqual(r1.status_code, 400)
        self.assertIn("error", r1.get_json())
        self.assertIn("Invalid memory algorithm", r1.get_json()["error"])


        r2 = self.client.post(
            "/api/v1/memory/page-replacement",
            json={"algorithm": "fifo", "num_frames": 0, "reference_string": [1, 2, 3]},
        )
        self.assertEqual(r2.status_code, 400)
        self.assertIn("error", r2.get_json())


        r3 = self.client.post(
            "/api/v1/memory/page-replacement",
            json={"algorithm": "fifo", "num_frames": 3, "reference_string": [1, "two", 3]},
        )
        self.assertEqual(r3.status_code, 400)
        self.assertIn("error", r3.get_json())




    def test_memory_monitor_snapshot(self) -> None:
        """POST /api/v1/memory/monitor returns system memory snapshot."""
        response = self.client.post("/api/v1/memory/monitor", json={})
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn("ram_total_bytes", data)
        self.assertIn("ram_percent", data)
        self.assertIn("swap_total_bytes", data)
        self.assertIn("timestamp", data)
        self.assertIsNone(data["proc_pid"])

    def test_memory_monitor_snapshot_with_pid(self) -> None:
        """POST /api/v1/memory/monitor includes per-process metrics when PID given."""
        my_pid = os.getpid()
        response = self.client.post("/api/v1/memory/monitor", json={"pid": my_pid})
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["proc_pid"], my_pid)
        self.assertIsNotNone(data["proc_rss_bytes"])

    def test_memory_monitor_collection(self) -> None:
        """POST /api/v1/memory/monitor collects multiple samples."""
        response = self.client.post(
            "/api/v1/memory/monitor",
            json={"count": 2, "interval_s": 0.0},
        )
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["count"], 2)
        self.assertEqual(len(data["snapshots"]), 2)

    def test_memory_monitor_invalid(self) -> None:
        """POST /api/v1/memory/monitor validates parameters and non-existent PIDs."""

        r1 = self.client.post("/api/v1/memory/monitor", json={"pid": 99999999})
        self.assertEqual(r1.status_code, 400)
        self.assertIn("error", r1.get_json())


        r2 = self.client.post("/api/v1/memory/monitor", json={"interval_s": 0.5})
        self.assertEqual(r2.status_code, 400)
        self.assertIn("error", r2.get_json())


        r3 = self.client.post("/api/v1/memory/monitor", json={"count": 0, "interval_s": 0.0})
        self.assertEqual(r3.status_code, 400)
        self.assertIn("error", r3.get_json())




    def test_device_predictive_action_healthy(self) -> None:
        """POST /api/v1/device/predictive-action succeeds for healthy status."""
        payload = {
            "device_id": "disk_01",
            "failure_probability": 0.25,
            "device_status": "healthy",
        }
        response = self.client.post("/api/v1/device/predictive-action", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["device_id"], "disk_01")
        self.assertEqual(data["failure_probability"], 0.25)
        self.assertFalse(data["backup_result"]["triggered"])
        self.assertFalse(data["recovery_result"]["recovered"])

    def test_device_predictive_action_high_risk_degraded(self) -> None:
        """POST /api/v1/device/predictive-action triggers backup and reroute on risk."""
        payload = {
            "device_id": "disk_01",
            "failure_probability": 0.85,
            "device_status": "degrading",
            "backup_device_id": "disk_backup_01",
            "threshold": 0.7,
        }
        response = self.client.post("/api/v1/device/predictive-action", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data["backup_result"]["triggered"])
        self.assertTrue(data["backup_result"]["success"])
        self.assertTrue(data["recovery_result"]["recovered"])
        self.assertTrue(data["recovery_result"]["rerouted"])
        self.assertEqual(data["recovery_result"]["backup_device_id"], "disk_backup_01")

    def test_device_predictive_action_invalid(self) -> None:
        """POST /api/v1/device/predictive-action rejects invalid device requests."""

        r1 = self.client.post(
            "/api/v1/device/predictive-action",
            json={"device_id": "disk_01", "failure_probability": 0.5},
        )
        self.assertEqual(r1.status_code, 400)
        self.assertIn("error", r1.get_json())


        r2 = self.client.post(
            "/api/v1/device/predictive-action",
            json={"device_id": "disk_01", "failure_probability": 1.5, "device_status": "healthy"},
        )
        self.assertEqual(r2.status_code, 400)
        self.assertIn("error", r2.get_json())


        r3 = self.client.post(
            "/api/v1/device/predictive-action",
            json={"device_id": "   ", "failure_probability": 0.5, "device_status": "healthy"},
        )
        self.assertEqual(r3.status_code, 400)
        self.assertIn("error", r3.get_json())




    def test_malformed_missing_json(self) -> None:
        """Non-JSON or malformed bodies return HTTP 400 JSON."""

        r1 = self.client.post("/api/v1/cpu/action", data="not json", content_type="text/plain")
        self.assertEqual(r1.status_code, 400)
        self.assertTrue(r1.is_json)
        self.assertIn("error", r1.get_json())


        r2 = self.client.post("/api/v1/cpu/action", data="{'broken': json}", content_type="application/json")
        self.assertEqual(r2.status_code, 400)
        self.assertTrue(r2.is_json)
        self.assertIn("error", r2.get_json())

    def test_unknown_route_returns_json_404(self) -> None:
        """Unknown route returns HTTP 404 with JSON error and no traceback."""
        response = self.client.get("/api/v1/unknown-route")
        self.assertEqual(response.status_code, 404)
        self.assertTrue(response.is_json)
        self.assertEqual(response.get_json(), {"error": "Resource not found"})

    def test_method_not_allowed_returns_json_405(self) -> None:
        """Invalid HTTP method returns HTTP 405 with JSON error."""
        response = self.client.post("/health")
        self.assertEqual(response.status_code, 405)
        self.assertTrue(response.is_json)
        self.assertEqual(response.get_json(), {"error": "Method not allowed"})

    def test_unexpected_exception_returns_generic_500(self) -> None:
        """Unexpected internal exceptions return generic HTTP 500 JSON without exposing details."""

        with patch.object(
            self.cpu_action_loop, "step", side_effect=RuntimeError("Unforeseen scheduler hardware failure")
        ):
            payload = {
                "processes": [
                    {
                        "pid": "P1",
                        "arrival_time": 0.0,
                        "burst_time": 10.0,
                    }
                ]
            }
            response = self.client.post("/api/v1/cpu/action", json=payload)
            self.assertEqual(response.status_code, 500)
            self.assertTrue(response.is_json)
            self.assertEqual(response.get_json(), {"error": "Internal server error"})
            self.assertNotIn("Unforeseen scheduler hardware failure", response.get_data(as_text=True))


        with patch(
            "backend.routes.handle_predictive_action",
            side_effect=RuntimeError("Unexpected hardware connection dropped"),
        ):
            device_payload = {
                "device_id": "disk_01",
                "failure_probability": 0.8,
                "device_status": "degrading",
            }
            response = self.client.post("/api/v1/device/predictive-action", json=device_payload)
            self.assertEqual(response.status_code, 500)
            self.assertTrue(response.is_json)
            self.assertEqual(response.get_json(), {"error": "Internal server error"})
            self.assertNotIn("Unexpected hardware connection dropped", response.get_data(as_text=True))




    def test_monitor_overview_valid(self) -> None:
        """GET /api/v1/monitor/overview returns HTTP 200 with consolidated telemetry."""
        response = self.client.get("/api/v1/monitor/overview")
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.is_json)
        data = response.get_json()

        self.assertIn("timestamp", data)
        self.assertEqual(data["status"], "ok")
        self.assertIn("cpu", data)
        self.assertIn("memory", data)
        self.assertIn("devices", data)


        self.assertEqual(data["cpu"]["total_steps"], 0)
        self.assertNotIn("history", data["cpu"])


        self.assertIn("ram_percent", data["memory"])
        self.assertIn("swap_percent", data["memory"])


        self.assertEqual(data["devices"]["device_count"], 4)
        self.assertEqual(len(data["devices"]["devices"]), 4)

    def test_monitor_overview_with_history(self) -> None:
        """GET /api/v1/monitor/overview?include_history=true populates CPU history."""

        payload = {
            "processes": [
                {
                    "pid": "P1",
                    "arrival_time": 0.0,
                    "burst_time": 6.0,
                }
            ]
        }
        post_resp = self.client.post("/api/v1/cpu/action", json=payload)
        self.assertEqual(post_resp.status_code, 200)


        response = self.client.get("/api/v1/monitor/overview?include_history=true")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["cpu"]["total_steps"], 1)
        self.assertIn("history", data["cpu"])
        self.assertEqual(len(data["cpu"]["history"]), 1)
        self.assertEqual(data["cpu"]["history"][0]["process_count"], 1)

    def test_monitor_overview_with_pid(self) -> None:
        """GET /api/v1/monitor/overview?pid=<pid> returns process-level memory stats."""
        my_pid = os.getpid()
        response = self.client.get(f"/api/v1/monitor/overview?pid={my_pid}")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["memory"]["proc_pid"], my_pid)
        self.assertIsNotNone(data["memory"]["proc_rss_bytes"])

    def test_monitor_overview_invalid_query_params(self) -> None:
        """GET /api/v1/monitor/overview validates query parameters."""

        r1 = self.client.get("/api/v1/monitor/overview?include_history=maybe")
        self.assertEqual(r1.status_code, 400)
        self.assertIn("error", r1.get_json())


        r2 = self.client.get("/api/v1/monitor/overview?pid=not_a_number")
        self.assertEqual(r2.status_code, 400)
        self.assertIn("error", r2.get_json())


        r3 = self.client.get("/api/v1/monitor/overview?pid=-5")
        self.assertEqual(r3.status_code, 400)
        self.assertIn("error", r3.get_json())


        r4 = self.client.get("/api/v1/monitor/overview?pid=99999999")
        self.assertEqual(r4.status_code, 400)
        self.assertIn("error", r4.get_json())

    def test_monitor_overview_unexpected_500(self) -> None:
        """Unexpected internal exception in monitor returns generic HTTP 500 JSON."""
        with patch(
            "modules.monitoring.shared_monitor.SharedMonitor.get_system_snapshot",
            side_effect=RuntimeError("Hardware sensor monitoring bus failed"),
        ):
            response = self.client.get("/api/v1/monitor/overview")
            self.assertEqual(response.status_code, 500)
            self.assertTrue(response.is_json)
            self.assertEqual(response.get_json(), {"error": "Internal server error"})
            self.assertNotIn("Hardware sensor monitoring bus failed", response.get_data(as_text=True))




    def test_system_step_valid(self) -> None:
        """POST /api/v1/system/step executes coordinated multi-resource step."""
        payload = {
            "processes": [
                {
                    "pid": "P1",
                    "arrival_time": 0.0,
                    "burst_time": 10.0,
                    "priority": 1,
                }
            ],
            "compare_with_baseline": False,
            "device_actions": [
                {
                    "device_id": "disk_0",
                    "failure_probability": 0.8,
                    "device_status": "degrading",
                    "backup_device_id": "disk_backup",
                }
            ],
        }
        response = self.client.post("/api/v1/system/step", json=payload)
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.is_json)
        data = response.get_json()

        self.assertEqual(data["step_index"], 0)
        self.assertIn("timestamp", data)
        self.assertIn("cpu", data)
        self.assertIn("memory", data)
        self.assertIn("devices", data)
        self.assertIn("system_snapshot", data)


        self.assertEqual(data["cpu"]["process_count"], 1)


        self.assertIn("ram_percent", data["memory"])


        self.assertEqual(len(data["devices"]), 1)
        self.assertEqual(data["devices"][0]["device_id"], "disk_0")
        self.assertTrue(data["devices"][0]["backup_result"]["triggered"])


        self.assertEqual(data["system_snapshot"]["cpu"]["total_steps"], 1)
        self.assertEqual(len(data["system_snapshot"]["devices"]["recent_actions"]), 1)

    def test_system_step_with_baseline(self) -> None:
        """POST /api/v1/system/step populates baseline and delta when enabled."""
        payload = {
            "processes": [
                {
                    "pid": 1,
                    "arrival_time": 0.0,
                    "burst_time": 12.0,
                }
            ],
            "compare_with_baseline": True,
        }
        response = self.client.post("/api/v1/system/step", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIsNotNone(data["cpu"]["baseline_result"])
        self.assertIsNotNone(data["cpu"]["action_delta"])

    def test_system_step_with_pid(self) -> None:
        """POST /api/v1/system/step returns process memory metrics for valid PID."""
        my_pid = os.getpid()
        payload = {
            "processes": [{"pid": "P1", "arrival_time": 0.0, "burst_time": 5.0}],
            "pid": my_pid,
        }
        response = self.client.post("/api/v1/system/step", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["memory"]["proc_pid"], my_pid)

    def test_system_step_validation_failures(self) -> None:
        """POST /api/v1/system/step rejects malformed requests with HTTP 400."""

        r1 = self.client.post("/api/v1/system/step", json={})
        self.assertEqual(r1.status_code, 400)
        self.assertIn("error", r1.get_json())


        r2 = self.client.post("/api/v1/system/step", json={"processes": []})
        self.assertEqual(r2.status_code, 400)


        r3 = self.client.post(
            "/api/v1/system/step",
            json={"processes": [{"pid": "P1", "arrival_time": 0.0, "burst_time": -1.0}]},
        )
        self.assertEqual(r3.status_code, 400)


        r4 = self.client.post(
            "/api/v1/system/step",
            json={
                "processes": [{"pid": "P1", "arrival_time": 0.0, "burst_time": 5.0}],
                "compare_with_baseline": "invalid",
            },
        )
        self.assertEqual(r4.status_code, 400)


        r5 = self.client.post(
            "/api/v1/system/step",
            json={
                "processes": [{"pid": "P1", "arrival_time": 0.0, "burst_time": 5.0}],
                "device_actions": [
                    {"device_id": "disk_0", "failure_probability": 2.0, "device_status": "ok"}
                ],
            },
        )
        self.assertEqual(r5.status_code, 400)


        r6 = self.client.post(
            "/api/v1/system/step",
            json={
                "processes": [{"pid": "P1", "arrival_time": 0.0, "burst_time": 5.0}],
                "pid": 99999999,
            },
        )
        self.assertEqual(r6.status_code, 400)
        self.assertIn("Process not found", r6.get_json()["error"])

    def test_system_step_unexpected_500(self) -> None:
        """Unexpected exception in system_step returns generic HTTP 500 JSON."""
        with patch(
            "modules.integration.integrated_execution.IntegratedExecutionManager.step",
            side_effect=RuntimeError("Bus arbiter synchronization failed"),
        ):
            payload = {
                "processes": [{"pid": "P1", "arrival_time": 0.0, "burst_time": 5.0}],
            }
            response = self.client.post("/api/v1/system/step", json=payload)
            self.assertEqual(response.status_code, 500)
            self.assertTrue(response.is_json)
            self.assertEqual(response.get_json(), {"error": "Internal server error"})
            self.assertNotIn("Bus arbiter synchronization failed", response.get_data(as_text=True))

    def test_system_step_generated_profiles(self) -> None:
        """POST /api/v1/system/step with workload_profile generates a matching workload."""
        expected = {
            "cpu_bound": "CPU_BOUND",
            "io_bound": "IO_BOUND",
            "priority_critical": "PRIORITY_CRITICAL",
            "balanced": "BALANCED",
        }
        for profile, label in expected.items():
            response = self.client.post(
                "/api/v1/system/step",
                json={"workload_profile": profile, "num_processes": 6, "seed": 7},
            )
            self.assertEqual(response.status_code, 200, profile)
            data = response.get_json()
            self.assertEqual(data["generated_workload"]["profile"], profile)
            self.assertEqual(data["generated_workload"]["rule_label"], label)
            self.assertEqual(data["cpu"]["process_count"], 6)
            processes = data["cpu"]["meta_result"]["schedule_result"]["processes"]
            for p in processes:
                self.assertEqual(p["workload_type"], profile)
                for feature in ("cpu_intensity", "io_wait_ratio", "instruction_count", "process_complexity"):
                    self.assertIn(feature, p["metadata"])
            predicted = {p["predicted_burst_time"] for p in processes}
            self.assertGreater(len(predicted), 1, f"{profile}: predicted bursts should vary per process")

    def test_system_step_generated_profile_is_reproducible(self) -> None:
        """Same workload_profile and seed produce the same processes."""
        payload = {"workload_profile": "random", "num_processes": 5, "seed": 123}
        first = self.client.post("/api/v1/system/step", json=payload).get_json()
        second = self.client.post("/api/v1/system/step", json=payload).get_json()
        self.assertEqual(first["generated_workload"]["profile"], second["generated_workload"]["profile"])
        procs = lambda d: [(p["pid"], p["burst_time"]) for p in d["cpu"]["meta_result"]["schedule_result"]["processes"]]
        self.assertEqual(sorted(procs(first)), sorted(procs(second)))

    def test_system_step_generated_profile_validation(self) -> None:
        """Invalid workload_profile, num_processes or seed return HTTP 400."""
        bad_payloads = [
            {"workload_profile": "gpu_bound"},
            {"workload_profile": 3},
            {"workload_profile": "balanced", "num_processes": 0},
            {"workload_profile": "balanced", "num_processes": 51},
            {"workload_profile": "balanced", "num_processes": True},
            {"workload_profile": "balanced", "seed": "x"},
        ]
        for payload in bad_payloads:
            response = self.client.post("/api/v1/system/step", json=payload)
            self.assertEqual(response.status_code, 400, payload)
            self.assertIn("error", response.get_json())

    def test_system_history_empty(self) -> None:
        """GET /api/v1/system/history returns an empty list before any step."""
        response = self.client.get("/api/v1/system/history")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json(), {"total_steps": 0, "steps": []})

    def test_system_history_newest_first_and_limit(self) -> None:
        """GET /api/v1/system/history returns recorded steps newest first, capped by limit."""
        payload = {"processes": [{"pid": "P1", "arrival_time": 0.0, "burst_time": 4.0}]}
        for _ in range(3):
            self.assertEqual(self.client.post("/api/v1/system/step", json=payload).status_code, 200)

        data = self.client.get("/api/v1/system/history?limit=2").get_json()
        self.assertEqual(data["total_steps"], 3)
        self.assertEqual([s["step_index"] for s in data["steps"]], [2, 1])
        self.assertIn("cpu", data["steps"][0])

    def test_system_history_invalid_limit(self) -> None:
        """GET /api/v1/system/history rejects a non-integer or out-of-range limit."""
        for value in ("abc", "0", "201"):
            response = self.client.get("/api/v1/system/history?limit=" + value)
            self.assertEqual(response.status_code, 400)
            self.assertIn("limit", response.get_json()["error"])


if __name__ == "__main__":
    unittest.main()
