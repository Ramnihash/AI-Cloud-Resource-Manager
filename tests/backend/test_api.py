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

    # --------------------------------------------------------------------------
    # 1. Health Endpoint
    # --------------------------------------------------------------------------
    def test_health(self) -> None:
        """GET /health returns HTTP 200 with status ok."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.is_json)
        data = response.get_json()
        self.assertEqual(data, {"status": "ok"})

    # --------------------------------------------------------------------------
    # 2. CPU Action Endpoint (Valid Requests)
    # --------------------------------------------------------------------------
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

    # --------------------------------------------------------------------------
    # 3. CPU Action Endpoint (Invalid Requests)
    # --------------------------------------------------------------------------
    def test_cpu_action_invalid_requests(self) -> None:
        """POST /api/v1/cpu/action rejects malformed workloads with HTTP 400."""
        # Missing processes field
        r1 = self.client.post("/api/v1/cpu/action", json={"compare_with_baseline": False})
        self.assertEqual(r1.status_code, 400)
        self.assertIn("error", r1.get_json())

        # Empty processes list
        r2 = self.client.post("/api/v1/cpu/action", json={"processes": []})
        self.assertEqual(r2.status_code, 400)
        self.assertIn("error", r2.get_json())

        # Processes not a list
        r3 = self.client.post("/api/v1/cpu/action", json={"processes": "not_a_list"})
        self.assertEqual(r3.status_code, 400)
        self.assertIn("error", r3.get_json())

        # Missing required process fields (e.g. burst_time)
        r4 = self.client.post("/api/v1/cpu/action", json={"processes": [{"pid": "P1", "arrival_time": 0.0}]})
        self.assertEqual(r4.status_code, 400)
        self.assertIn("error", r4.get_json())

        # Negative burst time rejected
        r5 = self.client.post(
            "/api/v1/cpu/action",
            json={"processes": [{"pid": "P1", "arrival_time": 0.0, "burst_time": -5.0}]},
        )
        self.assertEqual(r5.status_code, 400)
        self.assertIn("error", r5.get_json())

        # Non-boolean compare_with_baseline
        r6 = self.client.post(
            "/api/v1/cpu/action",
            json={"processes": [{"pid": "P1", "arrival_time": 0.0, "burst_time": 5.0}], "compare_with_baseline": "yes"},
        )
        self.assertEqual(r6.status_code, 400)
        self.assertIn("error", r6.get_json())

    # --------------------------------------------------------------------------
    # 4. Memory Page-Replacement Endpoint (FIFO, LRU, Optimal)
    # --------------------------------------------------------------------------
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

    # --------------------------------------------------------------------------
    # 5. Memory Page-Replacement Endpoint (Invalid Requests)
    # --------------------------------------------------------------------------
    def test_memory_page_replacement_invalid(self) -> None:
        """POST /api/v1/memory/page-replacement rejects invalid algorithm and parameters."""
        # Invalid algorithm name
        r1 = self.client.post(
            "/api/v1/memory/page-replacement",
            json={"algorithm": "clock", "num_frames": 3, "reference_string": [1, 2, 3]},
        )
        self.assertEqual(r1.status_code, 400)
        self.assertIn("error", r1.get_json())
        self.assertIn("Invalid memory algorithm", r1.get_json()["error"])

        # Invalid num_frames (< 1)
        r2 = self.client.post(
            "/api/v1/memory/page-replacement",
            json={"algorithm": "fifo", "num_frames": 0, "reference_string": [1, 2, 3]},
        )
        self.assertEqual(r2.status_code, 400)
        self.assertIn("error", r2.get_json())

        # Invalid reference string (non-integer element)
        r3 = self.client.post(
            "/api/v1/memory/page-replacement",
            json={"algorithm": "fifo", "num_frames": 3, "reference_string": [1, "two", 3]},
        )
        self.assertEqual(r3.status_code, 400)
        self.assertIn("error", r3.get_json())

    # --------------------------------------------------------------------------
    # 6. Memory Monitor Endpoint (Snapshot & Collection)
    # --------------------------------------------------------------------------
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
        # Non-existent PID
        r1 = self.client.post("/api/v1/memory/monitor", json={"pid": 99999999})
        self.assertEqual(r1.status_code, 400)
        self.assertIn("error", r1.get_json())

        # Collection with missing count
        r2 = self.client.post("/api/v1/memory/monitor", json={"interval_s": 0.5})
        self.assertEqual(r2.status_code, 400)
        self.assertIn("error", r2.get_json())

        # Collection with invalid count (< 1)
        r3 = self.client.post("/api/v1/memory/monitor", json={"count": 0, "interval_s": 0.0})
        self.assertEqual(r3.status_code, 400)
        self.assertIn("error", r3.get_json())

    # --------------------------------------------------------------------------
    # 7. Device Predictive-Action Endpoint
    # --------------------------------------------------------------------------
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
        # Missing required device_status
        r1 = self.client.post(
            "/api/v1/device/predictive-action",
            json={"device_id": "disk_01", "failure_probability": 0.5},
        )
        self.assertEqual(r1.status_code, 400)
        self.assertIn("error", r1.get_json())

        # Probability out of bounds
        r2 = self.client.post(
            "/api/v1/device/predictive-action",
            json={"device_id": "disk_01", "failure_probability": 1.5, "device_status": "healthy"},
        )
        self.assertEqual(r2.status_code, 400)
        self.assertIn("error", r2.get_json())

        # Empty device ID
        r3 = self.client.post(
            "/api/v1/device/predictive-action",
            json={"device_id": "   ", "failure_probability": 0.5, "device_status": "healthy"},
        )
        self.assertEqual(r3.status_code, 400)
        self.assertIn("error", r3.get_json())

    # --------------------------------------------------------------------------
    # 8. Malformed / Missing Request Validation & Global Error Handlers
    # --------------------------------------------------------------------------
    def test_malformed_missing_json(self) -> None:
        """Non-JSON or malformed bodies return HTTP 400 JSON."""
        # Content-type not application/json
        r1 = self.client.post("/api/v1/cpu/action", data="not json", content_type="text/plain")
        self.assertEqual(r1.status_code, 400)
        self.assertTrue(r1.is_json)
        self.assertIn("error", r1.get_json())

        # Malformed JSON syntax
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
        # 1. Force unexpected exception in CPUActionLoop.step
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

        # 2. Force unexpected exception in device predictive action handler
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


if __name__ == "__main__":
    unittest.main()
