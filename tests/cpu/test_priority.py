"""Unit tests for non-preemptive Priority scheduler."""

import unittest
from modules.cpu.priority import PriorityScheduler
from modules.cpu.process import Process


class TestPriority(unittest.TestCase):
    def test_priority_simultaneous_arrival(self):

        workload = [
            Process(pid="P1", arrival_time=0, burst_time=10, priority=3),
            Process(pid="P2", arrival_time=0, burst_time=1, priority=1),
            Process(pid="P3", arrival_time=0, burst_time=2, priority=4),
            Process(pid="P4", arrival_time=0, burst_time=5, priority=2),
        ]

        scheduler = PriorityScheduler()
        result = scheduler.schedule(workload)


        execution_order = [s.pid for s in result.gantt_chart]
        self.assertEqual(execution_order, ["P2", "P4", "P1", "P3"])

        completed = {p.pid: p for p in result.processes}
        self.assertEqual(completed["P2"].completion_time, 1.0)
        self.assertEqual(completed["P4"].completion_time, 6.0)
        self.assertEqual(completed["P1"].completion_time, 16.0)
        self.assertEqual(completed["P3"].completion_time, 18.0)

    def test_priority_non_preemption_staggered(self):


        workload = [
            Process(pid="P1", arrival_time=0, burst_time=4, priority=3),
            Process(pid="P2", arrival_time=1, burst_time=2, priority=2),
            Process(pid="P3", arrival_time=2, burst_time=3, priority=1),
        ]

        scheduler = PriorityScheduler()
        result = scheduler.schedule(workload)


        execution_order = [s.pid for s in result.gantt_chart]
        self.assertEqual(execution_order, ["P1", "P3", "P2"])

        completed = {p.pid: p for p in result.processes}
        self.assertEqual(completed["P1"].completion_time, 4.0)
        self.assertEqual(completed["P3"].completion_time, 7.0)
        self.assertEqual(completed["P2"].completion_time, 9.0)

    def test_priority_tie_breaking_by_arrival_time(self):
        workload = [
            Process(pid="P1", arrival_time=2, burst_time=3, priority=1),
            Process(pid="P2", arrival_time=0, burst_time=3, priority=1),
        ]

        scheduler = PriorityScheduler()
        result = scheduler.schedule(workload)


        self.assertEqual(result.gantt_chart[0].pid, "P2")
        self.assertEqual(result.gantt_chart[1].pid, "P1")

    def test_priority_with_idle_time(self):
        workload = [
            Process(pid="P1", arrival_time=3, burst_time=2, priority=1),
        ]
        scheduler = PriorityScheduler()
        result = scheduler.schedule(workload)

        self.assertEqual(len(result.gantt_chart), 2)
        self.assertTrue(result.gantt_chart[0].is_idle)
        self.assertEqual(result.gantt_chart[0].start_time, 0.0)
        self.assertEqual(result.gantt_chart[0].end_time, 3.0)
        self.assertEqual(result.gantt_chart[1].pid, "P1")

    def test_priority_input_immutability(self):
        p1 = Process(pid="P1", arrival_time=0, burst_time=5, priority=1)
        workload = [p1]
        scheduler = PriorityScheduler()
        scheduler.schedule(workload)

        self.assertIsNone(p1.completion_time)
        self.assertEqual(p1.remaining_time, 5.0)


if __name__ == "__main__":
    unittest.main()
