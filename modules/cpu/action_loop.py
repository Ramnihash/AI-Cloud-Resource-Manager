"""CPU AI Action Loop Simulation and Orchestration Engine (CP4.1).

Coordinates the PREDICT → CLASSIFY → DECIDE → ACT → MEASURE simulation cycle
across successive workload arrivals.

CRITICAL ARCHITECTURAL BOUNDARY:
    This class is strictly an in-memory SIMULATION and orchestration layer within
    the cloud resource-manager simulator. It operates entirely on simulated Process
    control blocks and simulated scheduler queues. It does NOT interact with, manage,
    or control the host operating system CPU, host processes, kernel threads, or
    hardware execution units.

Delegates inner prediction, classification, scheduler selection, and execution
to CPUMetaScheduler without duplicating any ML models or classical scheduling
algorithms.
"""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any, Optional

from modules.cpu.base_scheduler import BaseScheduler, ScheduleResult
from modules.cpu.fcfs import FCFSScheduler
from modules.cpu.meta_scheduler import CPUMetaScheduler, MetaScheduleResult
from modules.cpu.metrics import GanttSlice, SchedulingMetrics
from modules.cpu.process import Process


@dataclass(frozen=True)
class ActionLoopStep:
    """Immutable telemetry record for a single CPU action-loop simulation step."""

    step_index: int
    timestamp: float
    process_count: int
    meta_result: MetaScheduleResult
    baseline_result: Optional[ScheduleResult] = None
    action_delta: Optional[dict[str, float]] = None

    @property
    def selected_scheduler(self) -> str:
        """Name of the classical scheduler selected by the AI meta-scheduler."""
        return self.meta_result.selected_scheduler

    @property
    def workload_class(self) -> str:
        """Effective workload classification used for scheduler selection."""
        return self.meta_result.workload_class

    @property
    def predicted_class(self) -> str:
        """Raw classifier prediction before fallback thresholding."""
        return self.meta_result.predicted_class

    @property
    def confidence(self) -> float:
        """Prediction confidence score from the workload classifier."""
        return self.meta_result.confidence

    @property
    def is_fallback(self) -> bool:
        """Whether safe fallback was triggered due to low confidence."""
        return self.meta_result.is_fallback

    @property
    def metrics(self) -> SchedulingMetrics:
        """Scheduling performance metrics from the AI-selected execution."""
        return self.meta_result.metrics

    @property
    def processes(self) -> list[Process]:
        """Completed simulated processes with execution timing populated."""
        return self.meta_result.processes

    @property
    def gantt_chart(self) -> list[GanttSlice]:
        """Execution timeline Gantt chart slices."""
        return self.meta_result.gantt_chart

    def to_dict(self) -> dict[str, Any]:
        """Serialize step telemetry record to a dictionary."""
        return {
            "step_index": self.step_index,
            "timestamp": self.timestamp,
            "process_count": self.process_count,
            "selected_scheduler": self.selected_scheduler,
            "workload_class": self.workload_class,
            "predicted_class": self.predicted_class,
            "confidence": self.confidence,
            "is_fallback": self.is_fallback,
            "meta_result": self.meta_result.to_dict(),
            "baseline_result": self.baseline_result.to_dict() if self.baseline_result else None,
            "action_delta": dict(self.action_delta) if self.action_delta else None,
        }


class CPUActionLoop:
    """In-memory simulation controller orchestrating the CPU AI action loop.

    Operationalizes the PREDICT → CLASSIFY → DECIDE → ACT → MEASURE lifecycle
    for simulated process workloads. Maintains an in-memory step history and
    supports optional comparative baseline evaluation.
    """

    def __init__(
        self,
        meta_scheduler: CPUMetaScheduler,
        baseline_scheduler: Optional[BaseScheduler] = None,
    ) -> None:
        """Initialize CPUActionLoop simulation controller.

        Args:
            meta_scheduler: Configured CPUMetaScheduler instance.
            baseline_scheduler: Optional scheduler for baseline comparison.
                If None, internally defaults to FCFSScheduler().

        Raises:
            ValueError: If meta_scheduler is None.
            TypeError: If meta_scheduler or baseline_scheduler have invalid types.
        """
        if meta_scheduler is None:
            raise ValueError("meta_scheduler cannot be None.")
        if not isinstance(meta_scheduler, CPUMetaScheduler):
            raise TypeError(
                f"meta_scheduler must be a CPUMetaScheduler instance, got {type(meta_scheduler).__name__}"
            )

        if baseline_scheduler is not None:
            if not isinstance(baseline_scheduler, BaseScheduler):
                raise TypeError(
                    f"baseline_scheduler must be a BaseScheduler instance, got {type(baseline_scheduler).__name__}"
                )
            self._baseline_scheduler: BaseScheduler = baseline_scheduler
        else:
            self._baseline_scheduler = FCFSScheduler()

        self.meta_scheduler = meta_scheduler
        self._history: list[ActionLoopStep] = []
        self._step_counter: int = 0

    @property
    def baseline_scheduler(self) -> BaseScheduler:
        """Return the active baseline scheduler instance."""
        return self._baseline_scheduler

    def step(
        self,
        processes: list[Process],
        compare_with_baseline: bool = False,
        **kwargs: Any,
    ) -> ActionLoopStep:
        """Execute a single simulation cycle of the CPU AI action loop.

        Pipeline:
            PREDICT  → burst times predicted (via meta_scheduler.burst_predictor)
            CLASSIFY → workload classified with confidence (via meta_scheduler.classifier)
            DECIDE   → classical scheduler selected (via meta_scheduler.select_scheduler)
            ACT      → simulated dispatch executed (via classical scheduler)
            MEASURE  → metrics computed (via calculate_metrics)
            [OPTIONAL EVALUATION] → baseline scheduler executed on identical workload copy

        Args:
            processes: Input workload batch of simulated Process objects.
            compare_with_baseline: If True, evaluates baseline_scheduler on an
                identical workload copy and computes action delta.
            **kwargs: Scheduler-specific parameters forwarded to meta_scheduler.

        Returns:
            ActionLoopStep recording the complete telemetry for this step.
        """

        if not processes:
            raise ValueError("Cannot schedule an empty process workload.")
        for i, p in enumerate(processes):
            if not isinstance(p, Process):
                raise TypeError(
                    f"All elements in processes must be Process instances, got {type(p).__name__} at index {i}"
                )


        cloned_workload = [p.copy() for p in processes]


        meta_result = self.meta_scheduler.schedule(cloned_workload, **kwargs)


        baseline_result: Optional[ScheduleResult] = None
        action_delta: Optional[dict[str, float]] = None

        if compare_with_baseline:
            cloned_workload_baseline = [p.copy() for p in processes]
            baseline_result = self._baseline_scheduler.schedule(cloned_workload_baseline)




            meta_m = meta_result.metrics
            base_m = baseline_result.metrics
            action_delta = {
                "waiting_time_delta": round(meta_m.avg_waiting_time - base_m.avg_waiting_time, 4),
                "turnaround_time_delta": round(meta_m.avg_turnaround_time - base_m.avg_turnaround_time, 4),
                "response_time_delta": round(meta_m.avg_response_time - base_m.avg_response_time, 4),
            }

        step_record = ActionLoopStep(
            step_index=self._step_counter,
            timestamp=time.time(),
            process_count=len(processes),
            meta_result=meta_result,
            baseline_result=baseline_result,
            action_delta=action_delta,
        )

        self._history.append(step_record)
        self._step_counter += 1

        return step_record

    def run(
        self,
        workload_sequence: list[list[Process]],
        compare_with_baseline: bool = False,
        **kwargs: Any,
    ) -> list[ActionLoopStep]:
        """Execute a sequence of simulated workload batches sequentially.

        Args:
            workload_sequence: Ordered sequence of process batches.
            compare_with_baseline: If True, runs baseline comparison on each step.
            **kwargs: Parameters forwarded to each step.

        Returns:
            List of ActionLoopStep records generated during this run.

        Raises:
            ValueError: If workload_sequence is None.
        """
        if workload_sequence is None:
            raise ValueError("workload_sequence cannot be None.")

        executed_steps: list[ActionLoopStep] = []
        for batch in workload_sequence:
            step_record = self.step(
                batch,
                compare_with_baseline=compare_with_baseline,
                **kwargs,
            )
            executed_steps.append(step_record)

        return executed_steps

    def get_history(self) -> list[ActionLoopStep]:
        """Return a copy of the in-memory history list.

        Returns a new list to prevent external mutation of the internal history.
        """
        return list(self._history)

    def reset(self) -> None:
        """Clear the in-memory history and reset the step counter to 0."""
        self._history.clear()
        self._step_counter = 0

    def get_summary(self) -> dict[str, Any]:
        """Aggregate summary metrics for the current in-memory simulation session.

        Returns:
            Dictionary containing session totals, scheduler selection distribution,
            fallback frequency, and average performance metrics.
        """
        if not self._history:
            return {
                "total_steps": 0,
                "total_processes": 0,
                "scheduler_distribution": {},
                "scheduler_percentages": {},
                "fallback_count": 0,
                "fallback_rate": 0.0,
                "mean_metrics": {},
                "mean_action_delta": None,
            }

        total_steps = len(self._history)
        total_processes = sum(s.process_count for s in self._history)

        scheduler_counts = Counter(s.selected_scheduler for s in self._history)
        scheduler_percentages = {
            sched: round((count / total_steps) * 100.0, 2)
            for sched, count in scheduler_counts.items()
        }

        fallback_count = sum(1 for s in self._history if s.is_fallback)
        fallback_rate = round(fallback_count / total_steps, 4)

        mean_metrics = {
            "avg_waiting_time": round(
                sum(s.metrics.avg_waiting_time for s in self._history) / total_steps, 4
            ),
            "avg_turnaround_time": round(
                sum(s.metrics.avg_turnaround_time for s in self._history) / total_steps, 4
            ),
            "avg_response_time": round(
                sum(s.metrics.avg_response_time for s in self._history) / total_steps, 4
            ),
            "cpu_utilization": round(
                sum(s.metrics.cpu_utilization for s in self._history) / total_steps, 2
            ),
        }

        delta_steps = [s.action_delta for s in self._history if s.action_delta is not None]
        mean_action_delta: Optional[dict[str, float]] = None
        if delta_steps:
            n_delta = len(delta_steps)
            mean_action_delta = {
                "mean_waiting_time_delta": round(
                    sum(d["waiting_time_delta"] for d in delta_steps) / n_delta, 4
                ),
                "mean_turnaround_time_delta": round(
                    sum(d["turnaround_time_delta"] for d in delta_steps) / n_delta, 4
                ),
                "mean_response_time_delta": round(
                    sum(d["response_time_delta"] for d in delta_steps) / n_delta, 4
                ),
            }

        return {
            "total_steps": total_steps,
            "total_processes": total_processes,
            "scheduler_distribution": dict(scheduler_counts),
            "scheduler_percentages": scheduler_percentages,
            "fallback_count": fallback_count,
            "fallback_rate": fallback_rate,
            "mean_metrics": mean_metrics,
            "mean_action_delta": mean_action_delta,
        }
