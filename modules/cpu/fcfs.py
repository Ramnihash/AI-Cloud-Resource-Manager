"""First-Come, First-Served (FCFS) CPU scheduling algorithm."""

from __future__ import annotations

from typing import Any

from modules.cpu.base_scheduler import BaseScheduler
from modules.cpu.metrics import GanttSlice
from modules.cpu.process import Process


class FCFSScheduler(BaseScheduler):
    """First-Come, First-Served (FCFS) non-preemptive scheduler.

    Processes are scheduled strictly according to their arrival time.
    Ties in arrival time are resolved by PID order.
    """

    def __init__(self) -> None:
        super().__init__(name="FCFS")

    def _run_schedule(
        self, processes: list[Process], **kwargs: Any
    ) -> tuple[list[Process], list[GanttSlice]]:
        if not processes:
            return [], []

        # Sort by arrival time, then by PID for deterministic tie-breaking
        sorted_processes = sorted(processes, key=lambda p: (p.arrival_time, str(p.pid)))
        gantt_chart: list[GanttSlice] = []
        completed_processes: list[Process] = []
        current_time = 0.0

        for p in sorted_processes:
            # Handle CPU idle time if next process arrives after current_time
            if current_time < p.arrival_time:
                gantt_chart.append(
                    GanttSlice(start_time=current_time, end_time=p.arrival_time, pid=None)
                )
                current_time = p.arrival_time

            p.start_time = current_time
            p.response_time = p.start_time - p.arrival_time
            finish_time = current_time + p.burst_time

            if p.burst_time > 0:
                gantt_chart.append(
                    GanttSlice(start_time=current_time, end_time=finish_time, pid=p.pid)
                )

            current_time = finish_time
            p.remaining_time = 0.0
            p.completion_time = current_time
            p.turnaround_time = p.completion_time - p.arrival_time
            p.waiting_time = p.turnaround_time - p.burst_time
            completed_processes.append(p)

        return completed_processes, self.merge_consecutive_slices(gantt_chart)
