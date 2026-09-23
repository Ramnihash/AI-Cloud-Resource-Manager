"""Unit tests for non-preemptive SJF scheduler."""

import unittest
from modules.cpu.process import Process
from modules.cpu.sjf import SJFScheduler


class TestSJF(unittest.TestCase):
    def test_sjf_simultaneous_arrival(self):
        workload = [
            Process(pid="P1", arrival_time=0, burst_time=6),
            Process(pid="P2", arrival_time=0, burst_time=2),
            Process(pid="P3", arrival_time=0, burst_time=8),
            Process(pid="P4", arrival_time=0, burst_time=3),
        ]

        scheduler = SJFScheduler()
        result = scheduler.schedule(workload)


        execution_order = [s.pid for s in result.gantt_chart]
        self.assertEqual(execution_order, ["P2", "P4", "P1", "P3"])

        completed = {p.pid: p for p in result.processes}
        self.assertEqual(completed["P2"].completion_time, 2.0)
        self.assertEqual(completed["P2"].waiting_time, 0.0)

        self.assertEqual(completed["P4"].completion_time, 5.0)
        self.assertEqual(completed["P4"].waiting_time, 2.0)

        self.assertEqual(completed["P1"].completion_time, 11.0)
        self.assertEqual(completed["P1"].waiting_time, 5.0)

        self.assertEqual(completed["P3"].completion_time, 19.0)
        self.assertEqual(completed["P3"].waiting_time, 11.0)

        self.assertEqual(result.metrics.avg_waiting_time, 4.5)
        self.assertEqual(result.metrics.avg_turnaround_time, 9.25)

    def test_sjf_non_preemption_with_staggered_arrival(self):
        workload = [
            Process(pid="P1", arrival_time=0, burst_time=7),
            Process(pid="P2", arrival_time=2, burst_time=4),
            Process(pid="P3", arrival_time=4, burst_time=1),
        ]

        scheduler = SJFScheduler()
        result = scheduler.schedule(workload)


        execution_order = [s.pid for s in result.gantt_chart]
        self.assertEqual(execution_order, ["P1", "P3", "P2"])

        completed = {p.pid: p for p in result.processes}
        self.assertEqual(completed["P1"].completion_time, 7.0)
        self.assertEqual(completed["P3"].completion_time, 8.0)
        self.assertEqual(completed["P2"].completion_time, 12.0)

    def test_sjf_with_idle_time(self):
        workload = [
            Process(pid="P1", arrival_time=2, burst_time=3),
            Process(pid="P2", arrival_time=7, burst_time=2),
        ]

        scheduler = SJFScheduler()
        result = scheduler.schedule(workload)

        gantt = result.gantt_chart
        self.assertEqual(len(gantt), 4)
        self.assertTrue(gantt[0].is_idle)
        self.assertEqual(gantt[0].start_time, 0.0)
        self.assertEqual(gantt[0].end_time, 2.0)

        self.assertEqual(gantt[1].pid, "P1")
        self.assertEqual(gantt[1].start_time, 2.0)
        self.assertEqual(gantt[1].end_time, 5.0)

        self.assertTrue(gantt[2].is_idle)
        self.assertEqual(gantt[2].start_time, 5.0)
        self.assertEqual(gantt[2].end_time, 7.0)

        self.assertEqual(gantt[3].pid, "P2")
        self.assertEqual(gantt[3].start_time, 7.0)
        self.assertEqual(gantt[3].end_time, 9.0)

    def test_sjf_input_immutability(self):
        p1 = Process(pid="P1", arrival_time=0, burst_time=4)
        workload = [p1]
        scheduler = SJFScheduler()
        scheduler.schedule(workload)

        self.assertIsNone(p1.completion_time)
        self.assertEqual(p1.remaining_time, 4.0)


if __name__ == "__main__":
    unittest.main()
