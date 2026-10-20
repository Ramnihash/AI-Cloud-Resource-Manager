"""AI-Assisted CPU Meta-Scheduler for CP3.2 Step 3.

Operationalizes the PREDICT → CLASSIFY → DECIDE → ACT → MEASURE architecture.
Consumes batch classification from CPUWorkloadClassifier, selects the
appropriate classical scheduler according to the locked mapping:
    - CPU_BOUND         → SJF
    - IO_BOUND          → Round Robin
    - PRIORITY_CRITICAL → Priority
    - BALANCED          → FCFS
    - Fallback          → FCFS

Preserves caller workload immutability and delegates execution to existing
classical schedulers in modules/cpu/.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

from modules.cpu.base_scheduler import BaseScheduler, ScheduleResult
from modules.cpu.fcfs import FCFSScheduler
from modules.cpu.metrics import GanttSlice, SchedulingMetrics
from modules.cpu.priority import PriorityScheduler
from modules.cpu.process import Process
from modules.cpu.round_robin import RoundRobinScheduler
from modules.cpu.sjf import SJFScheduler

if TYPE_CHECKING:
    from ml.cpu.burst_predictor import CPUBurstPredictor
    from ml.cpu.workload_classifier import CPUWorkloadClassifier


logger = logging.getLogger(__name__)

CLASS_CPU_BOUND = "CPU_BOUND"
CLASS_IO_BOUND = "IO_BOUND"
CLASS_PRIORITY_CRITICAL = "PRIORITY_CRITICAL"
CLASS_BALANCED = "BALANCED"


CLASS_TO_SCHEDULER_NAME: dict[str, str] = {
    CLASS_CPU_BOUND: "SJF",
    CLASS_IO_BOUND: "Round Robin",
    CLASS_PRIORITY_CRITICAL: "Priority",
    CLASS_BALANCED: "FCFS",
}

DEFAULT_FALLBACK_SCHEDULER_NAME: str = "FCFS"
DEFAULT_ROUND_ROBIN_QUANTUM: float = 2.0


@dataclass(frozen=True)
class MetaScheduleResult:
    """Encapsulates the decision, execution, and metric results of meta-scheduling."""

    selected_scheduler: str
    workload_class: str
    predicted_class: str
    confidence: float
    is_fallback: bool
    classification_result: Any
    schedule_result: ScheduleResult
    scheduler_kwargs_used: dict[str, Any]
    fallback_reason: Optional[str] = None

    @property
    def algorithm_name(self) -> str:
        """Name of the selected scheduling algorithm."""
        return self.selected_scheduler

    @property
    def processes(self) -> list[Process]:
        """Completed processes with execution timing populated."""
        return self.schedule_result.processes

    @property
    def gantt_chart(self) -> list[GanttSlice]:
        """Gantt chart execution slices."""
        return self.schedule_result.gantt_chart

    @property
    def metrics(self) -> SchedulingMetrics:
        """Performance metrics computed for the schedule."""
        return self.schedule_result.metrics

    def to_dict(self) -> dict[str, Any]:
        """Serialize complete meta-schedule result to dictionary."""
        return {
            "selected_scheduler": self.selected_scheduler,
            "workload_class": self.workload_class,
            "predicted_class": self.predicted_class,
            "confidence": self.confidence,
            "is_fallback": self.is_fallback,
            "classification_result": (
                self.classification_result.to_dict()
                if hasattr(self.classification_result, "to_dict")
                else self.classification_result
            ),
            "schedule_result": self.schedule_result.to_dict(),
            "scheduler_kwargs_used": dict(self.scheduler_kwargs_used),
            "fallback_reason": self.fallback_reason,
        }


class CPUMetaScheduler:
    """AI-assisted CPU Meta-Scheduler orchestrating DECIDE and ACT stages.

    Maps high-level workload classifications to existing classical schedulers,
    safeguarding caller workload immutability and forwarding validated parameters.
    """

    def __init__(
        self,
        classifier: CPUWorkloadClassifier,
        burst_predictor: Optional[CPUBurstPredictor] = None,
        schedulers: Optional[dict[str, BaseScheduler]] = None,
        default_time_quantum: float = DEFAULT_ROUND_ROBIN_QUANTUM,
        sjf_use_predicted_burst: bool = True,
    ) -> None:
        from ml.cpu.burst_predictor import CPUBurstPredictor as BurstPredictorCls
        from ml.cpu.workload_classifier import CPUWorkloadClassifier as ClassifierCls

        if classifier is None:
            raise ValueError("classifier cannot be None.")
        if not isinstance(classifier, ClassifierCls):
            raise TypeError(
                f"classifier must be a CPUWorkloadClassifier instance, got {type(classifier).__name__}"
            )

        if burst_predictor is not None and not isinstance(burst_predictor, BurstPredictorCls):
            raise TypeError(
                f"burst_predictor must be a CPUBurstPredictor instance, got {type(burst_predictor).__name__}"
            )

        self.classifier = classifier
        self.burst_predictor = burst_predictor
        self.default_time_quantum = float(default_time_quantum)
        self.sjf_use_predicted_burst = sjf_use_predicted_burst

        if schedulers is not None:
            self._schedulers: dict[str, BaseScheduler] = dict(schedulers)
        else:
            self._schedulers = {
                "SJF": SJFScheduler(use_predicted_burst=self.sjf_use_predicted_burst),
                "Round Robin": RoundRobinScheduler(time_quantum=self.default_time_quantum),
                "Priority": PriorityScheduler(),
                "FCFS": FCFSScheduler(),
            }

    @property
    def registered_schedulers(self) -> dict[str, BaseScheduler]:
        """Return a shallow copy of the registered scheduler instances."""
        return dict(self._schedulers)

    def select_scheduler(self, classification: Any) -> BaseScheduler:
        """Resolve and return the classical scheduler instance for a classification result."""
        effective_class = classification.effective_class
        scheduler_name = CLASS_TO_SCHEDULER_NAME.get(
            effective_class, DEFAULT_FALLBACK_SCHEDULER_NAME
        )
        if scheduler_name not in self._schedulers:
            raise KeyError(
                f"Required scheduler '{scheduler_name}' for workload class '{effective_class}' "
                f"is missing from the scheduler registry."
            )
        return self._schedulers[scheduler_name]

    def _prepare_scheduler_kwargs(
        self, scheduler_name: str, **kwargs: Any
    ) -> dict[str, Any]:
        """Validate and filter scheduler kwargs so arguments are not blindly misdirected."""
        filtered: dict[str, Any] = {}

        if scheduler_name == "Round Robin":
            if "time_quantum" in kwargs:
                tq = kwargs["time_quantum"]
                if tq is not None:
                    if float(tq) <= 0:
                        raise ValueError(f"time_quantum must be positive, got {tq}")
                    filtered["time_quantum"] = float(tq)
            if "merge_consecutive" in kwargs:
                filtered["merge_consecutive"] = bool(kwargs["merge_consecutive"])
        elif scheduler_name == "SJF":
            if "use_predicted_burst" in kwargs:
                filtered["use_predicted_burst"] = bool(kwargs["use_predicted_burst"])

        return filtered

    def _validate_workload(self, processes: list[Process]) -> None:
        """Validate process list input types and completeness."""
        if not processes:
            raise ValueError("Cannot schedule an empty process workload.")
        for i, p in enumerate(processes):
            if not isinstance(p, Process):
                raise TypeError(
                    f"All elements in processes must be Process instances, got {type(p).__name__} at index {i}"
                )

    def classify_workload(self, processes: list[Process]) -> Any:
        """Classify the incoming workload without executing scheduling simulation."""
        if not self.classifier.is_fitted:
            raise RuntimeError("CPUWorkloadClassifier is not fitted. Cannot classify workload.")
        if self.burst_predictor is not None and not self.burst_predictor.is_fitted:
            raise RuntimeError("Provided CPUBurstPredictor is not fitted.")

        self._validate_workload(processes)


        cloned_processes = [p.copy() for p in processes]


        if any(p.predicted_burst_time is None for p in cloned_processes):
            if self.burst_predictor is None:
                raise ValueError(
                    "Workload contains processes with missing predicted_burst_time, "
                    "but no burst_predictor was provided to CPUMetaScheduler."
                )
            self.burst_predictor.predict_workload(cloned_processes, inplace=True)

        return self.classifier.predict_workload(
            cloned_processes, burst_predictor=self.burst_predictor
        )

    def schedule(self, processes: list[Process], **kwargs: Any) -> MetaScheduleResult:
        """Run the end-to-end AI-assisted meta-scheduling pipeline on a process workload.

        Workflow:
            1. Validate inputs and fitted state.
            2. Clone process objects to guarantee caller immutability.
            3. Populate predicted_burst_time using burst_predictor if missing.
            4. Classify workload using CPUWorkloadClassifier.
            5. Select classical scheduler based on effective_class (or fallback).
            6. Filter/validate scheduler-specific kwargs.
            7. Delegate execution to the selected classical scheduler.
            8. Return a complete MetaScheduleResult.
        """

        if not self.classifier.is_fitted:
            raise RuntimeError("CPUWorkloadClassifier is not fitted. Cannot schedule workload.")
        if self.burst_predictor is not None and not self.burst_predictor.is_fitted:
            raise RuntimeError("Provided CPUBurstPredictor is not fitted.")


        self._validate_workload(processes)


        cloned_processes = [p.copy() for p in processes]


        needs_prediction = any(p.predicted_burst_time is None for p in cloned_processes)
        if needs_prediction and self.burst_predictor is None:
            raise ValueError(
                "Workload contains processes with missing predicted_burst_time, "
                "but no burst_predictor was provided to CPUMetaScheduler."
            )

        try:
            if needs_prediction:
                self.burst_predictor.predict_workload(cloned_processes, inplace=True)
            classification = self.classifier.predict_workload(
                cloned_processes,
                burst_predictor=self.burst_predictor,
            )
        except Exception as exc:
            return self._fallback_schedule(cloned_processes, exc)


        scheduler = self.select_scheduler(classification)
        scheduler_name = scheduler.name


        scheduler_kwargs = self._prepare_scheduler_kwargs(scheduler_name, **kwargs)


        schedule_result = scheduler.schedule(cloned_processes, **scheduler_kwargs)

        return MetaScheduleResult(
            selected_scheduler=scheduler_name,
            workload_class=classification.effective_class,
            predicted_class=classification.predicted_class,
            confidence=classification.confidence,
            is_fallback=classification.is_fallback,
            classification_result=classification,
            schedule_result=schedule_result,
            scheduler_kwargs_used=scheduler_kwargs,
        )

    def _fallback_schedule(self, processes: list[Process], exc: Exception) -> MetaScheduleResult:
        """Run the default scheduler when the prediction or classification model fails."""
        reason = f"{type(exc).__name__}: {exc}"
        logger.warning(
            "Model failure during meta-scheduling (%s); falling back to %s",
            reason,
            DEFAULT_FALLBACK_SCHEDULER_NAME,
        )
        scheduler = self._schedulers[DEFAULT_FALLBACK_SCHEDULER_NAME]
        return MetaScheduleResult(
            selected_scheduler=scheduler.name,
            workload_class=CLASS_BALANCED,
            predicted_class="MODEL_ERROR",
            confidence=0.0,
            is_fallback=True,
            classification_result=None,
            schedule_result=scheduler.schedule(processes),
            scheduler_kwargs_used={},
            fallback_reason=reason,
        )
