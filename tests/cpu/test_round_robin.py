"""Unit tests for Round Robin (RR) scheduler."""

import unittest
from modules.cpu.process import Process
from modules.cpu.round_robin import RoundRobinScheduler


class TestRoundRobin(unittest.TestCase):
    def test_round_robin_basic(self):
        workload = [
            Process(pid="P1", arrival_time=0, burst_time=5),
            Process(pid="P2", arrival_time=0, burst_time=3),
            Process(pid="P3", arrival_time=0, burst_time=1),
        ]


        scheduler = RoundRobinScheduler(time_quantum=2.0)
        result = scheduler.schedule(workload)



        expected_gantt = [
            ("P1", 0.0, 2.0),
            ("P2", 2.0, 4.0),
            ("P3", 4.0, 5.0),
            ("P1", 5.0, 7.0),
            ("P2", 7.0, 8.0),
            ("P1", 8.0, 9.0),
        ]

        actual_gantt = [(s.pid, s.start_time, s.end_time) for s in result.gantt_chart]
        self.assertEqual(actual_gantt, expected_gantt)

        completed = {p.pid: p for p in result.processes}
        self.assertEqual(completed["P3"].completion_time, 5.0)
        self.assertEqual(completed["P2"].completion_time, 8.0)
        self.assertEqual(completed["P1"].completion_time, 9.0)


        self.assertEqual(completed["P1"].response_time, 0.0)
        self.assertEqual(completed["P2"].response_time, 2.0)
        self.assertEqual(completed["P3"].response_time, 4.0)

        self.assertAlmostEqual(result.metrics.avg_waiting_time, 13.0 / 3.0)
        self.assertAlmostEqual(result.metrics.avg_turnaround_time, 22.0 / 3.0)
        self.assertAlmostEqual(result.metrics.avg_response_time, 6.0 / 3.0)

    def test_round_robin_configurable_quantum_per_run(self):
        workload = [
            Process(pid="P1", arrival_time=0, burst_time=6),
            Process(pid="P2", arrival_time=0, burst_time=3),
        ]

        scheduler = RoundRobinScheduler(time_quantum=2.0)


        res_q2 = scheduler.schedule(workload)
        self.assertEqual(len(res_q2.gantt_chart), 5)


        res_q10 = scheduler.schedule(workload, time_quantum=10.0)
        self.assertEqual(len(res_q10.gantt_chart), 2)
        self.assertEqual(res_q10.gantt_chart[0].pid, "P1")
        self.assertEqual(res_q10.gantt_chart[0].end_time, 6.0)
        self.assertEqual(res_q10.gantt_chart[1].pid, "P2")
        self.assertEqual(res_q10.gantt_chart[1].end_time, 9.0)

    def test_round_robin_arrival_during_execution_slice(self):


        workload = [
            Process(pid="P1", arrival_time=0, burst_time=4),
            Process(pid="P2", arrival_time=1, burst_time=2),
        ]

        scheduler = RoundRobinScheduler(time_quantum=2.0)
        result = scheduler.schedule(workload)


        order = [s.pid for s in result.gantt_chart]
        self.assertEqual(order, ["P1", "P2", "P1"])

        completed = {p.pid: p for p in result.processes}
        self.assertEqual(completed["P2"].completion_time, 4.0)
        self.assertEqual(completed["P1"].completion_time, 6.0)

    def test_round_robin_with_idle_gap(self):
        workload = [
            Process(pid="P1", arrival_time=0, burst_time=2),
            Process(pid="P2", arrival_time=5, burst_time=2),
        ]

        scheduler = RoundRobinScheduler(time_quantum=2.0)
        result = scheduler.schedule(workload)

        self.assertEqual(len(result.gantt_chart), 3)
        self.assertEqual(result.gantt_chart[0].pid, "P1")
        self.assertEqual(result.gantt_chart[0].end_time, 2.0)

        self.assertTrue(result.gantt_chart[1].is_idle)
        self.assertEqual(result.gantt_chart[1].start_time, 2.0)
        self.assertEqual(result.gantt_chart[1].end_time, 5.0)

        self.assertEqual(result.gantt_chart[2].pid, "P2")
        self.assertEqual(result.gantt_chart[2].end_time, 7.0)

    def test_round_robin_invalid_quantum_raises(self):
        with self.assertRaises(ValueError):
            RoundRobinScheduler(time_quantum=0)

        with self.assertRaises(ValueError):
            RoundRobinScheduler(time_quantum=-2)

        scheduler = RoundRobinScheduler(time_quantum=2)
        with self.assertRaises(ValueError):
            scheduler.schedule([Process(pid="P1", arrival_time=0, burst_time=1)], time_quantum=0)

    def test_round_robin_input_immutability(self):
        p1 = Process(pid="P1", arrival_time=0, burst_time=6)
        p2 = Process(pid="P2", arrival_time=0, burst_time=3)
        workload = [p1, p2]

        scheduler = RoundRobinScheduler(time_quantum=2.0)
        scheduler.schedule(workload)

        self.assertIsNone(p1.completion_time)
        self.assertEqual(p1.remaining_time, 6.0)
        self.assertIsNone(p2.completion_time)
        self.assertEqual(p2.remaining_time, 3.0)


if __name__ == "__main__":
    unittest.main()
