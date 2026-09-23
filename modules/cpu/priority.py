"""Non-preemptive Priority CPU scheduling algorithm."""

from __future__ import annotations

from typing import Any

from modules.cpu.base_scheduler import BaseScheduler
from modules.cpu.metrics import GanttSlice
from modules.cpu.process import Process


class PriorityScheduler(BaseScheduler):
    """Non-preemptive Priority scheduler.

    At each scheduling point, the available process with the highest priority
    (defined as the lowest numerical integer value) is chosen and executed to completion.
    Ties in priority are broken by arrival time (FCFS), then PID.
    """

    def __init__(self) -> None:
        super().__init__(name="Priority")

    def _run_schedule(
        self, processes: list[Process], **kwargs: Any
    ) -> tuple[list[Process], list[GanttSlice]]:
        if not processes:
            return [], []

        uncompleted = list(processes)
        completed_processes: list[Process] = []
        gantt_chart: list[GanttSlice] = []
        current_time = 0.0

        while uncompleted:

            available = [p for p in uncompleted if p.arrival_time <= current_time]

            if not available:

                next_arrival = min(p.arrival_time for p in uncompleted)
                gantt_chart.append(
                    GanttSlice(start_time=current_time, end_time=next_arrival, pid=None)
                )
                current_time = next_arrival
                available = [p for p in uncompleted if p.arrival_time <= current_time]



            selected = min(
                available,
                key=lambda p: (p.priority, p.arrival_time, str(p.pid)),
            )

            uncompleted.remove(selected)

            selected.start_time = current_time
            selected.response_time = selected.start_time - selected.arrival_time
            finish_time = current_time + selected.burst_time

            if selected.burst_time > 0:
                gantt_chart.append(
                    GanttSlice(start_time=current_time, end_time=finish_time, pid=selected.pid)
                )

            current_time = finish_time
            selected.remaining_time = 0.0
            selected.completion_time = current_time
            selected.turnaround_time = selected.completion_time - selected.arrival_time
            selected.waiting_time = selected.turnaround_time - selected.burst_time
            completed_processes.append(selected)

        return completed_processes, self.merge_consecutive_slices(gantt_chart)
