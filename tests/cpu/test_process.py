"""Unit tests for Process model."""

import unittest
from modules.cpu.process import Process


class TestProcess(unittest.TestCase):
    def test_process_initialization(self):
        p = Process(pid="P1", arrival_time=0, burst_time=5, priority=2)
        self.assertEqual(p.pid, "P1")
        self.assertEqual(p.arrival_time, 0.0)
        self.assertEqual(p.burst_time, 5.0)
        self.assertEqual(p.remaining_time, 5.0)
        self.assertEqual(p.priority, 2)
        self.assertIsNone(p.start_time)
        self.assertIsNone(p.completion_time)
        self.assertEqual(p.waiting_time, 0.0)
        self.assertEqual(p.turnaround_time, 0.0)
        self.assertEqual(p.response_time, 0.0)
        self.assertFalse(p.is_completed)

    def test_process_negative_values_raise(self):
        with self.assertRaises(ValueError):
            Process(pid="P1", arrival_time=-1, burst_time=5)

        with self.assertRaises(ValueError):
            Process(pid="P2", arrival_time=0, burst_time=-5)

    def test_process_copy_independence(self):
        original = Process(pid="P1", arrival_time=2, burst_time=10, priority=1)
        clone = original.copy()

        clone.remaining_time = 4.0
        clone.completion_time = 12.0

        self.assertEqual(original.remaining_time, 10.0)
        self.assertIsNone(original.completion_time)

    def test_process_to_dict_and_reset(self):
        p = Process(pid="P1", arrival_time=1.0, burst_time=6.0, priority=3)
        p.start_time = 1.0
        p.completion_time = 7.0
        p.remaining_time = 0.0
        p.turnaround_time = 6.0
        p.waiting_time = 0.0
        p.response_time = 0.0

        d = p.to_dict()
        self.assertEqual(d["pid"], "P1")
        self.assertEqual(d["burst_time"], 6.0)
        self.assertEqual(d["completion_time"], 7.0)

        p.reset_execution_state()
        self.assertEqual(p.remaining_time, 6.0)
        self.assertIsNone(p.completion_time)
        self.assertIsNone(p.start_time)


if __name__ == "__main__":
    unittest.main()
