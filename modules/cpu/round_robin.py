"""Round Robin (RR) preemptive CPU scheduling algorithm."""

from __future__ import annotations

from collections import deque
from typing import Any

from modules.cpu.base_scheduler import BaseScheduler
from modules.cpu.metrics import GanttSlice
from modules.cpu.process import Process


class RoundRobinScheduler(BaseScheduler):
    """Round Robin (RR) preemptive scheduler with configurable time quantum.

    Allocates the CPU to ready processes in first-in, first-out cyclic order
    for up to a configurable time quantum.
    """

    def __init__(self, time_quantum: float = 2.0, merge_consecutive: bool = False) -> None:
        super().__init__(name="Round Robin")
        if time_quantum <= 0:
            raise ValueError(f"time_quantum must be positive, got {time_quantum}")
        self.time_quantum = float(time_quantum)
        self.merge_consecutive = merge_consecutive

    def _run_schedule(
        self, processes: list[Process], **kwargs: Any
    ) -> tuple[list[Process], list[GanttSlice]]:
        if not processes:
            return [], []

        quantum = float(kwargs.get("time_quantum", self.time_quantum))
        if quantum <= 0:
            raise ValueError(f"time_quantum must be positive, got {quantum}")

        merge_slices = kwargs.get("merge_consecutive", self.merge_consecutive)

        # Sort input processes by arrival time, then PID
        unarrived = deque(sorted(processes, key=lambda p: (p.arrival_time, str(p.pid))))
        ready_queue: deque[Process] = deque()
        completed_processes: list[Process] = []
        gantt_chart: list[GanttSlice] = []
        current_time = 0.0

        # Prime the ready queue with initial arrivals
        while unarrived and unarrived[0].arrival_time <= current_time:
            ready_queue.append(unarrived.popleft())

        while ready_queue or unarrived:
            if not ready_queue:
                # CPU is idle until next process arrives
                next_arrival = unarrived[0].arrival_time
                if next_arrival > current_time:
                    gantt_chart.append(
                        GanttSlice(start_time=current_time, end_time=next_arrival, pid=None)
                    )
                    current_time = next_arrival

                # Enqueue all processes arriving at this new current_time
                while unarrived and unarrived[0].arrival_time <= current_time:
                    ready_queue.append(unarrived.popleft())

            # Dequeue next process from ready queue
            current_process = ready_queue.popleft()

            # Record initial response time if not yet started
            if current_process.start_time is None:
                current_process.start_time = current_time
                current_process.response_time = current_process.start_time - current_process.arrival_time

            # Compute slice duration
            slice_duration = min(quantum, current_process.remaining_time)
            finish_slice_time = current_time + slice_duration

            if slice_duration > 0:
                gantt_chart.append(
                    GanttSlice(
                        start_time=current_time,
                        end_time=finish_slice_time,
                        pid=current_process.pid,
                    )
                )

            current_process.remaining_time -= slice_duration

            # Enqueue any processes arriving during this execution slice BEFORE re-enqueuing current_process
            while unarrived and unarrived[0].arrival_time <= finish_slice_time:
                ready_queue.append(unarrived.popleft())

            current_time = finish_slice_time

            if current_process.remaining_time > 0:
                # Process not yet finished; place at back of ready queue
                ready_queue.append(current_process)
            else:
                # Process completed
                current_process.completion_time = current_time
                current_process.turnaround_time = current_process.completion_time - current_process.arrival_time
                current_process.waiting_time = current_process.turnaround_time - current_process.burst_time
                completed_processes.append(current_process)

        if merge_slices:
            return completed_processes, self.merge_consecutive_slices(gantt_chart)
        return completed_processes, gantt_chart
