"""
Unit tests for CP7.4 Memory Ablation and Fallback checks.
"""
import json
import unittest
from pathlib import Path

from scripts.evaluate_memory_ablation import (
    compute_ablation_metrics,
    execute_fallback_checks,
    load_cp34_results,
)

ROOT = Path(__file__).resolve().parents[2]
CP74_DIR = ROOT / "data" / "memory" / "evaluation_cp74"


class TestMemoryAblationEvaluation(unittest.TestCase):
    def test_load_cp34_results(self):
        """Verify CP3.4 evaluation data loads and contains expected keys."""
        data = load_cp34_results()
        self.assertIn("markov", data)
        self.assertIn("transformer", data)
        self.assertIn("split_rows", data)
        self.assertEqual(data["split_rows"]["test"], 7936)

    def test_compute_ablation_metrics(self):
        """Verify ablation computation calculates deltas and non-sequential averages."""
        cp34_data = load_cp34_results()
        ablation = compute_ablation_metrics(cp34_data)

        self.assertIn("by_pattern", ablation)
        self.assertIn("non_sequential_averages", ablation)
        self.assertIn("inference_timing", ablation)

        by_pattern = ablation["by_pattern"]
        for pattern in ("looping", "mixed", "sequential", "uniform", "overall"):
            self.assertIn(pattern, by_pattern)
            stat = by_pattern[pattern]
            self.assertIn("markov_top1", stat)
            self.assertIn("transformer_top1", stat)
            self.assertIn("top1_delta", stat)
            self.assertAlmostEqual(
                stat["top1_delta"],
                stat["transformer_top1"] - stat["markov_top1"],
                places=6,
            )

        # Looping pattern must show Transformer advantage
        self.assertGreater(by_pattern["looping"]["transformer_top1"], by_pattern["looping"]["markov_top1"])
        self.assertGreater(by_pattern["looping"]["top1_relative_lift_percent"], 100.0)

        # Sequential pattern must be 1.0 (100%) on both
        self.assertEqual(by_pattern["sequential"]["markov_top1"], 1.0)
        self.assertEqual(by_pattern["sequential"]["transformer_top1"], 1.0)

        # Non-sequential average must be calculated without sequential
        nonseq = ablation["non_sequential_averages"]
        self.assertEqual(nonseq["patterns_included"], ["looping", "mixed", "uniform"])
        self.assertLess(nonseq["markov_top1_avg"], 0.20)
        self.assertLess(nonseq["transformer_top1_avg"], 0.20)

    def test_execute_fallback_checks(self):
        """Verify all live fallback scenarios pass."""
        summary, checks = execute_fallback_checks()
        self.assertEqual(summary["status"], "ALL_CHECKS_PASSED")
        self.assertEqual(summary["passed_checks"], summary["total_checks"])
        self.assertGreaterEqual(summary["total_checks"], 5)

        for check in checks:
            self.assertTrue(check["passed"], f"Check failed: {check['scenario_name']}")
            self.assertTrue(check["fallback_triggered"])
            self.assertEqual(check["active_model"], "markov")
            self.assertTrue(len(check["predictions"]) > 0)

    def test_cp74_artifacts_exist_and_valid(self):
        """Verify all 5 CP7.4 artifacts exist on disk and have valid content."""
        required_files = [
            "ablation_summary.json",
            "ablation_comparison.csv",
            "ablation_report.md",
            "ablation_accuracy.png",
            "fallback_test_results.json",
        ]
        for fname in required_files:
            fpath = CP74_DIR / fname
            self.assertTrue(fpath.exists(), f"Missing artifact: {fpath}")
            self.assertGreater(fpath.stat().st_size, 0, f"Empty artifact: {fpath}")

        # Check JSON validity
        summary = json.loads((CP74_DIR / "ablation_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["checkpoint"], "CP7.4")
        self.assertEqual(summary["fallback_verification"]["status"], "ALL_CHECKS_PASSED")

        fb_results = json.loads((CP74_DIR / "fallback_test_results.json").read_text(encoding="utf-8"))
        self.assertEqual(fb_results["status"], "ALL_CHECKS_PASSED")


if __name__ == "__main__":
    unittest.main()
