"""
Unit tests for CP7.5 Memory Stress Testing.
"""
import csv
import json
import unittest
from pathlib import Path

from ml.memory.memory_leak_detector import MemoryLeakDetector
from ml.memory.memory_pressure_forecaster import MemoryPressureForecaster
from modules.memory.memory_health_action import MemoryHealthActionLayer
from modules.memory.page_replacement import (
    FIFOPageReplacement,
    LRUPageReplacement,
    OptimalPageReplacement,
)
from scripts.stress_test_memory import (
    generate_looping_trace,
    generate_random_trace,
    generate_sequential_trace,
    run_health_pressure_stress_tests,
    run_paging_stress_tests,
)

ROOT = Path(__file__).resolve().parents[2]
CP75_DIR = ROOT / "data" / "memory" / "evaluation_cp75"


class TestMemoryStressTesting(unittest.TestCase):
    def test_trace_generators_deterministic(self):
        """Verify synthetic trace generators produce expected lengths and deterministic sequences."""
        # Sequential
        seq1 = generate_sequential_trace(num_pages=8, length=50)
        seq2 = generate_sequential_trace(num_pages=8, length=50)
        self.assertEqual(len(seq1), 50)
        self.assertEqual(seq1, seq2)
        self.assertEqual(seq1[:8], list(range(8)))

        # Random (with fixed seed)
        rand1 = generate_random_trace(num_pages=16, length=100, seed=42)
        rand2 = generate_random_trace(num_pages=16, length=100, seed=42)
        rand_diff = generate_random_trace(num_pages=16, length=100, seed=999)
        self.assertEqual(len(rand1), 100)
        self.assertEqual(rand1, rand2)
        self.assertNotEqual(rand1, rand_diff)

        # Looping
        loop1 = generate_looping_trace(list(range(6)), length=60, seed=2026)
        loop2 = generate_looping_trace(list(range(6)), length=60, seed=2026)
        self.assertEqual(len(loop1), 60)
        self.assertEqual(loop1, loop2)

    def test_paging_stress_invariants(self):
        """Verify fundamental paging invariants across multiple stress configurations."""
        trace = generate_looping_trace(list(range(6)), length=200, seed=42)

        for frames in (3, 4, 8):
            fifo = FIFOPageReplacement(frames).run(trace)
            lru = LRUPageReplacement(frames).run(trace)
            opt = OptimalPageReplacement(frames).run(trace)

            # Invariant 1: Optimal must never have more faults than FIFO or LRU
            self.assertLessEqual(opt["page_faults"], fifo["page_faults"])
            self.assertLessEqual(opt["page_faults"], lru["page_faults"])

            # Invariant 2: Hit ratio + Miss ratio == 1.0
            for res in (fifo, lru, opt):
                self.assertAlmostEqual(res["hit_ratio"] + res["miss_ratio"], 1.0, places=6)
                self.assertEqual(res["page_hits"] + res["page_faults"], len(trace))

        # Invariant 3: Allocating more frames to a looping working set must reduce faults
        opt_3 = OptimalPageReplacement(3).run(trace)
        opt_8 = OptimalPageReplacement(8).run(trace)
        self.assertLess(opt_8["page_faults"], opt_3["page_faults"])

    def test_health_action_layer_priorities(self):
        """Verify strict priority: Leak -> High Pressure -> Nominal."""
        action_layer = MemoryHealthActionLayer(pressure_threshold=90.0)

        # 1. Nominal
        d_nominal = action_layer.evaluate_and_act(
            leak_detection={"is_leak_detected": False, "proc_pid": 1001},
            pressure_prediction={"predicted_ram_percent": 55.0}
        )
        self.assertEqual(d_nominal["action"], "NO_ACTION")

        # 2. High Pressure Alert (RAM >= 90%)
        d_pressure = action_layer.evaluate_and_act(
            leak_detection={"is_leak_detected": False, "proc_pid": 1001},
            pressure_prediction={"predicted_ram_percent": 93.5}
        )
        self.assertEqual(d_pressure["action"], "ALERT_HIGH_PRESSURE")

        # 3. Memory Leak Detected (Priority 1)
        d_leak = action_layer.evaluate_and_act(
            leak_detection={"is_leak_detected": True, "proc_pid": 2045},
            pressure_prediction={"predicted_ram_percent": 75.0}
        )
        self.assertEqual(d_leak["action"], "RESTART_PROCESS")
        self.assertEqual(d_leak["pid"], 2045)

        # 4. Compound (Both Leak AND High Pressure) -> Leak wins
        d_compound = action_layer.evaluate_and_act(
            leak_detection={"is_leak_detected": True, "proc_pid": 3050},
            pressure_prediction={"predicted_ram_percent": 96.0}
        )
        self.assertEqual(d_compound["action"], "RESTART_PROCESS")

    def test_negative_cases_and_graceful_handling(self):
        """Verify that invalid inputs raise descriptive errors and are captured."""
        # Forecaster with short window
        forecaster = MemoryPressureForecaster(window_size=5)
        with self.assertRaises(ValueError):
            forecaster.train([{"ram_percent": 50.0}])

        # Leak detector with short window
        leak_detector = MemoryLeakDetector(window_size=10)
        with self.assertRaises(ValueError):
            leak_detector.detect([{"proc_pid": 1, "proc_rss_bytes": 1000}])

        # Leak detector with mixed PIDs
        with self.assertRaises(ValueError):
            leak_detector.detect([
                {"proc_pid": 1 if i < 5 else 2, "proc_rss_bytes": 1000}
                for i in range(12)
            ])

        # Missing key
        with self.assertRaises(ValueError):
            forecaster.train([{"wrong_key": 50.0} for _ in range(8)])

    def test_simulated_recovery(self):
        """Verify simulated telemetry reset transitions back to nominal NO_ACTION."""
        action_layer = MemoryHealthActionLayer(pressure_threshold=90.0)

        # Active leak
        act1 = action_layer.evaluate_and_act(
            leak_detection={"is_leak_detected": True, "proc_pid": 4010},
            pressure_prediction={"predicted_ram_percent": 80.0}
        )
        self.assertEqual(act1["action"], "RESTART_PROCESS")

        # Post-restart simulated recovery (simulated telemetry reset)
        act2 = action_layer.evaluate_and_act(
            leak_detection={"is_leak_detected": False, "proc_pid": 4010},
            pressure_prediction={"predicted_ram_percent": 52.0}
        )
        self.assertEqual(act2["action"], "NO_ACTION")

    def test_cp75_artifacts_exist_and_valid(self):
        """Verify all 6 CP7.5 artifacts exist and contain valid data."""
        required_files = [
            "stress_results.csv",
            "stress_summary.json",
            "stress_report.md",
            "stress_page_faults.png",
            "stress_actions.png",
            "issue_log.csv",
        ]
        for fname in required_files:
            fpath = CP75_DIR / fname
            self.assertTrue(fpath.exists(), f"Missing artifact: {fpath}")
            self.assertGreater(fpath.stat().st_size, 0, f"Empty artifact: {fpath}")

        # Check JSON summary
        summary = json.loads((CP75_DIR / "stress_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["checkpoint"], "CP7.5")
        self.assertEqual(summary["status"], "ALL_STRESS_TESTS_PASSED")
        self.assertEqual(summary["total_paging_simulations"], 81)
        self.assertEqual(len(summary["health_scenarios"]), 6)
        for sc in summary["health_scenarios"]:
            self.assertTrue(sc["passed"], f"Failed scenario: {sc['scenario_name']}")

        # Check issue log CSV
        with (CP75_DIR / "issue_log.csv").open(newline="", encoding="utf-8") as f:
            reader = list(csv.DictReader(f))
            self.assertGreaterEqual(len(reader), 5)
            for row in reader:
                self.assertEqual(row["handled_gracefully"], "True")


if __name__ == "__main__":
    unittest.main()
