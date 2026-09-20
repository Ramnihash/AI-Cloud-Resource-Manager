"""Unit tests for FCFS scheduler."""

import unittest
from modules.cpu.fcfs import FCFSScheduler
from modules.cpu.process import Process


class TestFCFS(unittest.TestCase):
    def test_fcfs_basic_simultaneous_arrival(self):
        workload = [
            Process(pid="P1", arrival_time=0, burst_time=4),
            Process(pid="P2", arrival_time=0, burst_time=3),
            Process(pid="P3", arrival_time=0, burst_time=1),
        ]

        scheduler = FCFSScheduler()
        result = scheduler.schedule(workload)

        # Verify order of completion
        completed = {p.pid: p for p in result.processes}
        self.assertEqual(completed["P1"].completion_time, 4.0)
        self.assertEqual(completed["P1"].waiting_time, 0.0)
        self.assertEqual(completed["P1"].turnaround_time, 4.0)
        self.assertEqual(completed["P1"].response_time, 0.0)

        self.assertEqual(completed["P2"].completion_time, 7.0)
        self.assertEqual(completed["P2"].waiting_time, 4.0)
        self.assertEqual(completed["P2"].turnaround_time, 7.0)
        self.assertEqual(completed["P2"].response_time, 4.0)

        self.assertEqual(completed["P3"].completion_time, 8.0)
        self.assertEqual(completed["P3"].waiting_time, 7.0)
        self.assertEqual(completed["P3"].turnaround_time, 8.0)
        self.assertEqual(completed["P3"].response_time, 7.0)

        self.assertAlmostEqual(result.metrics.avg_waiting_time, 11.0 / 3.0)
        self.assertAlmostEqual(result.metrics.avg_turnaround_time, 19.0 / 3.0)
        self.assertAlmostEqual(result.metrics.avg_response_time, 11.0 / 3.0)
        self.assertEqual(result.metrics.cpu_utilization, 100.0)

    def test_fcfs_with_idle_time(self):
        workload = [
            Process(pid="P1", arrival_time=0, burst_time=3),
            Process(pid="P2", arrival_time=5, burst_time=4),
        ]

        scheduler = FCFSScheduler()
        result = scheduler.schedule(workload)

        # Check Gantt chart contains the idle block
        gantt = result.gantt_chart
        self.assertEqual(len(gantt), 3)
        self.assertEqual(gantt[0].start_time, 0.0)
        self.assertEqual(gantt[0].end_time, 3.0)
        self.assertEqual(gantt[0].pid, "P1")

        self.assertEqual(gantt[1].start_time, 3.0)
        self.assertEqual(gantt[1].end_time, 5.0)
        self.assertTrue(gantt[1].is_idle)

        self.assertEqual(gantt[2].start_time, 5.0)
        self.assertEqual(gantt[2].end_time, 9.0)
        self.assertEqual(gantt[2].pid, "P2")

        self.assertEqual(result.metrics.total_time, 9.0)
        self.assertEqual(result.metrics.total_busy_time, 7.0)
        self.assertEqual(result.metrics.total_idle_time, 2.0)
        self.assertAlmostEqual(result.metrics.cpu_utilization, 7.0 / 9.0 * 100.0)

    def test_fcfs_input_immutability(self):
        p1 = Process(pid="P1", arrival_time=0, burst_time=5)
        p2 = Process(pid="P2", arrival_time=1, burst_time=3)
        original_workload = [p1, p2]

        scheduler = FCFSScheduler()
        result = scheduler.schedule(original_workload)

        # Check that original process instances are completely unmodified
        self.assertIsNone(p1.completion_time)
        self.assertEqual(p1.remaining_time, 5.0)
        self.assertEqual(p1.waiting_time, 0.0)
        self.assertIsNone(p2.completion_time)
        self.assertEqual(p2.remaining_time, 3.0)

        # But scheduled result has completed state
        self.assertIsNotNone(result.processes[0].completion_time)

    def test_fcfs_empty_workload(self):
        scheduler = FCFSScheduler()
        result = scheduler.schedule([])
        self.assertEqual(len(result.processes), 0)
        self.assertEqual(len(result.gantt_chart), 0)
        self.assertEqual(result.metrics.cpu_utilization, 0.0)


if __name__ == "__main__":
    unittest.main()
