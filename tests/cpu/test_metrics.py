"""Unit tests for CPU scheduling metrics and Gantt representations."""

import unittest
from modules.cpu.fcfs import FCFSScheduler
from modules.cpu.metrics import GanttSlice, calculate_metrics
from modules.cpu.process import Process


class TestMetrics(unittest.TestCase):
    def test_gantt_slice_properties(self):
        slice_normal = GanttSlice(start_time=2.0, end_time=5.5, pid="P1")
        self.assertEqual(slice_normal.duration, 3.5)
        self.assertFalse(slice_normal.is_idle)
        self.assertEqual(slice_normal.to_dict()["pid"], "P1")

        slice_idle = GanttSlice(start_time=5.5, end_time=7.0, pid=None)
        self.assertEqual(slice_idle.duration, 1.5)
        self.assertTrue(slice_idle.is_idle)
        self.assertIsNone(slice_idle.to_dict()["pid"])

        with self.assertRaises(ValueError):
            GanttSlice(start_time=5.0, end_time=3.0, pid="P1")

    def test_calculate_metrics_math(self):
        p1 = Process(pid="P1", arrival_time=0.0, burst_time=4.0)
        p1.start_time = 0.0
        p1.completion_time = 4.0

        p2 = Process(pid="P2", arrival_time=1.0, burst_time=2.0)
        p2.start_time = 4.0
        p2.completion_time = 6.0

        gantt = [
            GanttSlice(0.0, 4.0, pid="P1"),
            GanttSlice(4.0, 6.0, pid="P2"),
        ]

        metrics = calculate_metrics([p1, p2], gantt, overhead_ms=0.15)



        self.assertEqual(p1.turnaround_time, 4.0)
        self.assertEqual(p1.waiting_time, 0.0)
        self.assertEqual(p1.response_time, 0.0)

        self.assertEqual(p2.turnaround_time, 5.0)
        self.assertEqual(p2.waiting_time, 3.0)
        self.assertEqual(p2.response_time, 3.0)

        self.assertEqual(metrics.avg_turnaround_time, 4.5)
        self.assertEqual(metrics.avg_waiting_time, 1.5)
        self.assertEqual(metrics.avg_response_time, 1.5)
        self.assertEqual(metrics.cpu_utilization, 100.0)
        self.assertEqual(metrics.total_time, 6.0)
        self.assertEqual(metrics.total_busy_time, 6.0)
        self.assertEqual(metrics.total_idle_time, 0.0)
        self.assertEqual(metrics.completed_process_count, 2)
        self.assertEqual(metrics.scheduling_overhead_ms, 0.15)

    def test_metrics_empty_processes(self):
        metrics = calculate_metrics([], [], overhead_ms=0.0)
        self.assertEqual(metrics.avg_waiting_time, 0.0)
        self.assertEqual(metrics.avg_turnaround_time, 0.0)
        self.assertEqual(metrics.avg_response_time, 0.0)
        self.assertEqual(metrics.cpu_utilization, 0.0)
        self.assertEqual(metrics.completed_process_count, 0)

    def test_schedule_result_to_dict(self):
        workload = [Process(pid="P1", arrival_time=0, burst_time=2)]
        scheduler = FCFSScheduler()
        result = scheduler.schedule(workload)

        res_dict = result.to_dict()
        self.assertEqual(res_dict["algorithm_name"], "FCFS")
        self.assertEqual(len(res_dict["processes"]), 1)
        self.assertEqual(len(res_dict["gantt_chart"]), 1)
        self.assertIn("avg_waiting_time", res_dict["metrics"])
        self.assertIn("cpu_utilization", res_dict["metrics"])
        self.assertEqual(res_dict["metrics"]["cpu_utilization"], 100.0)


if __name__ == "__main__":
    unittest.main()
