"""Non-preemptive Shortest Job First (SJF) CPU scheduling algorithm."""

from __future__ import annotations

from typing import Any

from modules.cpu.base_scheduler import BaseScheduler
from modules.cpu.metrics import GanttSlice
from modules.cpu.process import Process


class SJFScheduler(BaseScheduler):
    """Non-preemptive Shortest Job First (SJF) scheduler.

    At each scheduling point, the available process with the shortest CPU burst
    time is selected and executed to completion.
    Ties in burst time are broken by arrival time, then PID.
    """

    def __init__(self, use_predicted_burst: bool = False) -> None:
        super().__init__(name="SJF")
        self.use_predicted_burst = use_predicted_burst

    def _run_schedule(
        self, processes: list[Process], **kwargs: Any
    ) -> tuple[list[Process], list[GanttSlice]]:
        if not processes:
            return [], []

        use_predicted = kwargs.get("use_predicted_burst", self.use_predicted_burst)

        def get_decision_burst(proc: Process) -> float:
            if use_predicted and proc.predicted_burst_time is not None:
                return proc.predicted_burst_time
            return proc.burst_time

        uncompleted = list(processes)
        completed_processes: list[Process] = []
        gantt_chart: list[GanttSlice] = []
        current_time = 0.0

        while uncompleted:
            # Find all processes that have arrived by current_time
            available = [p for p in uncompleted if p.arrival_time <= current_time]

            if not available:
                # No process has arrived yet; advance time to next arriving process
                next_arrival = min(p.arrival_time for p in uncompleted)
                gantt_chart.append(
                    GanttSlice(start_time=current_time, end_time=next_arrival, pid=None)
                )
                current_time = next_arrival
                available = [p for p in uncompleted if p.arrival_time <= current_time]

            # Select process with minimum burst time, breaking ties by arrival time then PID
            selected = min(
                available,
                key=lambda p: (get_decision_burst(p), p.arrival_time, str(p.pid)),
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
