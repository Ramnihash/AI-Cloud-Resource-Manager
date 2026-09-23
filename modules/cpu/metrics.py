"""Scheduling metrics calculation and Gantt chart representation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from modules.cpu.process import Process


@dataclass
class GanttSlice:
    """Represents an execution interval on the CPU."""

    start_time: float
    end_time: float
    pid: Optional[int | str] = None

    def __post_init__(self) -> None:
        self.start_time = float(self.start_time)
        self.end_time = float(self.end_time)
        if self.end_time < self.start_time:
            raise ValueError(
                f"Invalid GanttSlice: end_time ({self.end_time}) cannot be less than start_time ({self.start_time})"
            )

    @property
    def duration(self) -> float:
        """Duration of the CPU slice."""
        return self.end_time - self.start_time

    @property
    def is_idle(self) -> bool:
        """Returns True if this slice represents CPU idle time."""
        return self.pid is None or str(self.pid).upper() == "IDLE"

    def to_dict(self) -> dict[str, Any]:
        """Serialize slice to dictionary."""
        return {
            "start_time": self.start_time,
            "end_time": self.end_time,
            "pid": None if self.is_idle else self.pid,
            "duration": self.duration,
            "is_idle": self.is_idle,
        }


@dataclass
class SchedulingMetrics:
    """Aggregated system metrics for a CPU scheduling run."""

    avg_waiting_time: float
    avg_turnaround_time: float
    avg_response_time: float
    cpu_utilization: float
    total_time: float
    total_busy_time: float
    total_idle_time: float
    completed_process_count: int
    scheduling_overhead_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize metrics to dictionary."""
        return {
            "avg_waiting_time": self.avg_waiting_time,
            "avg_turnaround_time": self.avg_turnaround_time,
            "avg_response_time": self.avg_response_time,
            "cpu_utilization": self.cpu_utilization,
            "total_time": self.total_time,
            "total_busy_time": self.total_busy_time,
            "total_idle_time": self.total_idle_time,
            "completed_process_count": self.completed_process_count,
            "scheduling_overhead_ms": self.scheduling_overhead_ms,
        }


def calculate_metrics(
    processes: list[Process],
    gantt_chart: list[GanttSlice],
    overhead_ms: float = 0.0,
) -> SchedulingMetrics:
    """Calculate per-process and aggregate scheduling metrics.

    Ensures per-process waiting, turnaround, and response times are accurately
    calculated and aggregate system statistics (averages and CPU utilization) are returned.
    """
    total_busy_time = 0.0
    total_idle_time = 0.0

    for s in gantt_chart:
        if s.is_idle:
            total_idle_time += s.duration
        else:
            total_busy_time += s.duration

    total_time = total_busy_time + total_idle_time
    cpu_utilization = (total_busy_time / total_time * 100.0) if total_time > 0 else 0.0

    count = len(processes)
    if count == 0:
        return SchedulingMetrics(
            avg_waiting_time=0.0,
            avg_turnaround_time=0.0,
            avg_response_time=0.0,
            cpu_utilization=0.0,
            total_time=total_time,
            total_busy_time=total_busy_time,
            total_idle_time=total_idle_time,
            completed_process_count=0,
            scheduling_overhead_ms=overhead_ms,
        )

    for p in processes:
        if p.completion_time is not None:
            p.turnaround_time = p.completion_time - p.arrival_time
            p.waiting_time = max(0.0, p.turnaround_time - p.burst_time)
            if p.start_time is not None:
                p.response_time = max(0.0, p.start_time - p.arrival_time)

    avg_waiting = sum(p.waiting_time for p in processes) / count
    avg_turnaround = sum(p.turnaround_time for p in processes) / count
    avg_response = sum(p.response_time for p in processes) / count

    return SchedulingMetrics(
        avg_waiting_time=avg_waiting,
        avg_turnaround_time=avg_turnaround,
        avg_response_time=avg_response,
        cpu_utilization=cpu_utilization,
        total_time=total_time,
        total_busy_time=total_busy_time,
        total_idle_time=total_idle_time,
        completed_process_count=count,
        scheduling_overhead_ms=overhead_ms,
    )
