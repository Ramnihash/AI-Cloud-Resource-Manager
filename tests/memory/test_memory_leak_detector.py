import unittest
from ml.memory.memory_leak_detector import MemoryLeakDetector

class TestMemoryLeakDetector(unittest.TestCase):
    def setUp(self):
        self.detector = MemoryLeakDetector(
            window_size=10,
            min_growth_bytes_per_step=100.0,
            min_r2=0.75
        )

    def test_construction_and_validation(self):
        with self.assertRaises(ValueError):
            MemoryLeakDetector(window_size=1)
        with self.assertRaises(ValueError):
            MemoryLeakDetector(window_size=0)
        with self.assertRaises(ValueError):
            MemoryLeakDetector(window_size=-5)

        det = MemoryLeakDetector(window_size=5)
        self.assertEqual(det.window_size, 5)

    def test_missing_and_mixed_pids(self):

        snaps = [{"proc_rss_bytes": 1000} for _ in range(10)]
        with self.assertRaises(ValueError):
            self.detector.detect(snaps)


        snaps = [{"proc_pid": None, "proc_rss_bytes": 1000} for _ in range(10)]
        with self.assertRaises(ValueError):
            self.detector.detect(snaps)


        snaps = [{"proc_pid": 100 if i < 5 else 200, "proc_rss_bytes": 1000} for i in range(10)]
        with self.assertRaises(ValueError):
            self.detector.detect(snaps)

    def test_insufficient_history(self):
        snaps = [{"proc_pid": 100, "proc_rss_bytes": 1000} for _ in range(5)]
        with self.assertRaises(ValueError):
            self.detector.detect(snaps)

    def test_stable_process(self):

        snaps = [{"proc_pid": 100, "proc_rss_bytes": 5000.0} for _ in range(10)]
        result = self.detector.detect(snaps)
        self.assertFalse(result["is_leak_detected"])
        self.assertEqual(result["proc_pid"], 100)
        self.assertAlmostEqual(result["growth_rate_bytes_per_step"], 0.0)

    def test_synthetic_leak(self):

        snaps = [{"proc_pid": 100, "proc_rss_bytes": 1000.0 + i * 200.0} for i in range(10)]
        result = self.detector.detect(snaps)
        self.assertTrue(result["is_leak_detected"])
        self.assertAlmostEqual(result["growth_rate_bytes_per_step"], 200.0)
        self.assertAlmostEqual(result["trend_r2_score"], 1.0)

    def test_temporary_spike(self):



        vals = [1000, 1000, 5000, 5000, 5000, 1000, 1000, 1000, 1000, 1000]
        snaps = [{"proc_pid": 100, "proc_rss_bytes": float(v)} for v in vals]
        result = self.detector.detect(snaps)


        self.assertFalse(result["is_leak_detected"])

        self.assertTrue(result["growth_rate_bytes_per_step"] < self.detector.min_growth_bytes_per_step)

        self.assertTrue(result["trend_r2_score"] < self.detector.min_r2)

    def test_evaluate(self):
        labeled_seqs = [
            {
                "snapshots": [{"proc_pid": 1, "proc_rss_bytes": 1000.0} for _ in range(10)],
                "expected_is_leak": False
            },
            {
                "snapshots": [{"proc_pid": 1, "proc_rss_bytes": 1000.0 + i * 200.0} for i in range(10)],
                "expected_is_leak": True
            }
        ]
        result = self.detector.evaluate(labeled_seqs)
        self.assertEqual(result["total_sequences"], 2)
        self.assertEqual(result["correct_predictions"], 2)
        self.assertAlmostEqual(result["accuracy"], 1.0)
