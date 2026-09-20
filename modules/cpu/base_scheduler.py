"""Base class and common result structure for CPU scheduling algorithms."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from modules.cpu.metrics import GanttSlice, SchedulingMetrics, calculate_metrics
from modules.cpu.process import Process


@dataclass
class ScheduleResult:
    """Encapsulates the complete result of a CPU scheduling simulation run."""

    algorithm_name: str
    processes: list[Process]
    gantt_chart: list[GanttSlice]
    metrics: SchedulingMetrics

    def to_dict(self) -> dict[str, Any]:
        """Serialize complete schedule result to dictionary."""
        return {
            "algorithm_name": self.algorithm_name,
            "processes": [p.to_dict() for p in self.processes],
            "gantt_chart": [s.to_dict() for s in self.gantt_chart],
            "metrics": self.metrics.to_dict(),
        }


class BaseScheduler(ABC):
    """Abstract base class for all CPU schedulers."""

    def __init__(self, name: str) -> None:
        self.name = name

    def schedule(self, processes: list[Process], **kwargs: Any) -> ScheduleResult:
        """Execute the scheduling algorithm on a process workload.

        Input safety guarantee: The caller's process list and individual process
        objects are cloned to prevent permanent mutation.
        """
        # Protect caller workload by deep-copying process objects
        workload = [p.copy() for p in processes]

        start_time = time.perf_counter()
        completed_processes, gantt_chart = self._run_schedule(workload, **kwargs)
        overhead_ms = (time.perf_counter() - start_time) * 1000.0

        # Sort completed processes by PID or original order for stable inspection
        metrics = calculate_metrics(completed_processes, gantt_chart, overhead_ms=overhead_ms)

        return ScheduleResult(
            algorithm_name=self.name,
            processes=completed_processes,
            gantt_chart=gantt_chart,
            metrics=metrics,
        )

    @abstractmethod
    def _run_schedule(
        self, processes: list[Process], **kwargs: Any
    ) -> tuple[list[Process], list[GanttSlice]]:
        """Core scheduling implementation to be overridden by concrete algorithms.

        Args:
            processes: Pristine copies of input processes.
            **kwargs: Algorithm-specific parameters (e.g., time_quantum).

        Returns:
            Tuple of (completed processes with timing updated, Gantt chart slices).
        """
        pass

    @staticmethod
    def merge_consecutive_slices(slices: list[GanttSlice]) -> list[GanttSlice]:
        """Utility to merge contiguous Gantt slices belonging to the same PID."""
        if not slices:
            return []

        merged: list[GanttSlice] = []
        current = GanttSlice(slices[0].start_time, slices[0].end_time, slices[0].pid)

        for s in slices[1:]:
            if s.pid == current.pid and s.start_time == current.end_time:
                current.end_time = s.end_time
            else:
                merged.append(current)
                current = GanttSlice(s.start_time, s.end_time, s.pid)

        merged.append(current)
        return merged
