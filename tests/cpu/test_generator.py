"""Unit tests for CPU workload and process generator."""

import tempfile
import unittest
from pathlib import Path

from data.cpu.generator import CPUWorkloadGenerator, WorkloadConfig
from modules.cpu.fcfs import FCFSScheduler
from modules.cpu.priority import PriorityScheduler
from modules.cpu.round_robin import RoundRobinScheduler
from modules.cpu.sjf import SJFScheduler


class TestCPUWorkloadGenerator(unittest.TestCase):
    def test_reproducibility_with_same_seed(self):
        gen1 = CPUWorkloadGenerator(seed=42)
        gen2 = CPUWorkloadGenerator(seed=42)

        w1 = gen1.generate()
        w2 = gen2.generate()

        self.assertEqual(len(w1), len(w2))
        for p1, p2 in zip(w1, w2):
            self.assertEqual(p1.pid, p2.pid)
            self.assertEqual(p1.arrival_time, p2.arrival_time)
            self.assertEqual(p1.burst_time, p2.burst_time)
            self.assertEqual(p1.priority, p2.priority)

    def test_different_results_with_different_seeds(self):
        gen = CPUWorkloadGenerator()
        w1 = gen.generate(num_processes=10, seed=42)
        w2 = gen.generate(num_processes=10, seed=99)

        diff_count = sum(
            1 for p1, p2 in zip(w1, w2) if (p1.burst_time != p2.burst_time or p1.arrival_time != p2.arrival_time)
        )
        self.assertGreater(diff_count, 0)

    def test_requested_process_count(self):
        gen = CPUWorkloadGenerator()
        for count in [1, 5, 25]:
            workload = gen.generate(num_processes=count, seed=123)
            self.assertEqual(len(workload), count)

    def test_valid_ranges_and_constraints(self):
        config = WorkloadConfig(
            num_processes=30,
            arrival_range=(2.0, 8.0),
            burst_range=(3.0, 15.0),
            priority_range=(1, 4),
            seed=10,
        )
        gen = CPUWorkloadGenerator(config=config)
        workload = gen.generate()

        for p in workload:
            self.assertGreaterEqual(p.arrival_time, 2.0)
            self.assertLessEqual(p.arrival_time, 8.0)
            self.assertGreaterEqual(p.burst_time, 3.0)
            self.assertLessEqual(p.burst_time, 15.0)
            self.assertGreaterEqual(p.priority, 1)
            self.assertLessEqual(p.priority, 4)

    def test_zero_arrival_distribution(self):
        config = WorkloadConfig(
            num_processes=10,
            arrival_distribution="zero",
            seed=7,
        )
        gen = CPUWorkloadGenerator(config=config)
        workload = gen.generate()

        for p in workload:
            self.assertEqual(p.arrival_time, 0.0)

    def test_exponential_distributions(self):
        config = WorkloadConfig(
            num_processes=15,
            arrival_distribution="exponential",
            burst_distribution="exponential",
            burst_range=(1.0, 10.0),
            seed=99,
        )
        gen = CPUWorkloadGenerator(config=config)
        workload = gen.generate()

        self.assertEqual(len(workload), 15)
        for p in workload:
            self.assertGreaterEqual(p.arrival_time, 0.0)
            self.assertGreater(p.burst_time, 0.0)

    def test_compatibility_with_all_schedulers(self):
        gen = CPUWorkloadGenerator(seed=123)
        workload = gen.generate(num_processes=8)

        schedulers = [
            FCFSScheduler(),
            SJFScheduler(),
            PriorityScheduler(),
            RoundRobinScheduler(time_quantum=2.5),
        ]

        for s in schedulers:
            res = s.schedule(workload)
            self.assertEqual(len(res.processes), 8)
            self.assertEqual(res.metrics.completed_process_count, 8)
            self.assertGreater(res.metrics.total_time, 0.0)
            self.assertGreater(res.metrics.cpu_utilization, 0.0)
            self.assertGreater(len(res.gantt_chart), 0)

    def test_to_dataframe(self):
        gen = CPUWorkloadGenerator(seed=42)
        workload = gen.generate(num_processes=5)
        df = gen.to_dataframe(workload)

        self.assertEqual(len(df), 5)
        self.assertListEqual(list(df.columns), ["pid", "arrival_time", "burst_time", "priority"])
        self.assertEqual(set(df["pid"]), {"P1", "P2", "P3", "P4", "P5"})

    def test_csv_export_and_load(self):
        gen = CPUWorkloadGenerator(seed=42)
        original_workload = gen.generate(num_processes=6)

        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "test_workload.csv"
            CPUWorkloadGenerator.export_to_csv(original_workload, csv_path)
            self.assertTrue(csv_path.exists())

            loaded_workload = CPUWorkloadGenerator.load_from_csv(csv_path)
            self.assertEqual(len(loaded_workload), len(original_workload))

            for orig, loaded in zip(original_workload, loaded_workload):
                self.assertEqual(orig.pid, loaded.pid)
                self.assertEqual(orig.arrival_time, loaded.arrival_time)
                self.assertEqual(orig.burst_time, loaded.burst_time)
                self.assertEqual(orig.priority, loaded.priority)

    def test_invalid_configurations_rejected(self):
        with self.assertRaises(ValueError):
            WorkloadConfig(num_processes=0).validate()

        with self.assertRaises(ValueError):
            WorkloadConfig(arrival_range=(-1.0, 5.0)).validate()

        with self.assertRaises(ValueError):
            WorkloadConfig(arrival_range=(10.0, 5.0)).validate()

        with self.assertRaises(ValueError):
            WorkloadConfig(burst_range=(0.0, 5.0)).validate()

        with self.assertRaises(ValueError):
            WorkloadConfig(burst_range=(10.0, 5.0)).validate()

        with self.assertRaises(ValueError):
            WorkloadConfig(priority_range=(-1, 3)).validate()

        with self.assertRaises(ValueError):
            WorkloadConfig(priority_range=(5, 2)).validate()

        with self.assertRaises(ValueError):
            WorkloadConfig(arrival_distribution="invalid_dist").validate()

        with self.assertRaises(ValueError):
            WorkloadConfig(burst_distribution="invalid_dist").validate()

    def test_load_csv_missing_file_or_invalid_header(self):
        with self.assertRaises(FileNotFoundError):
            CPUWorkloadGenerator.load_from_csv("non_existent_file.csv")

        with tempfile.TemporaryDirectory() as tmp_dir:
            bad_csv = Path(tmp_dir) / "bad.csv"
            bad_csv.write_text("pid,burst_time\nP1,5.0", encoding="utf-8")
            with self.assertRaises(ValueError):
                CPUWorkloadGenerator.load_from_csv(bad_csv)


if __name__ == "__main__":
    unittest.main()
