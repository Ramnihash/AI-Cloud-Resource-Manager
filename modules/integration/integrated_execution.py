"""Integrated Execution Orchestration Layer (CP5.3).

Coordinates multi-resource execution across:
- CPU: Simulated AI scheduling lifecycle (CPUActionLoop.step)
- Memory: Host memory observation (MemoryMonitor.take_snapshot)
- Device: Failure risk handling & recovery (handle_predictive_action)
- Monitoring: System-wide state aggregation (SharedMonitor.get_system_snapshot)

This component is strictly an in-memory simulation and orchestration coordinator.
It does not perform database transactions or destructive host OS modifications.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import datetime
import threading
from typing import Any, Optional

from modules.cpu.action_loop import CPUActionLoop
from modules.cpu.process import Process
from modules.device.device import Device
from modules.device.predictive_action import (
    PredictiveActionResult,
    handle_predictive_action,
)
from modules.memory.memory_monitor import MemoryMonitor
from modules.monitoring.shared_monitor import SharedMonitor


@dataclass(frozen=True)
class IntegratedExecutionStep:
    """Immutable record for a coordinated multi-resource execution step."""

    step_index: int
    timestamp: str
    cpu_result: dict[str, Any]
    memory_result: dict[str, Any]
    device_results: list[dict[str, Any]]
    system_snapshot: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Serialize integrated execution step to a dictionary."""
        return {
            "step_index": self.step_index,
            "timestamp": self.timestamp,
            "cpu": self.cpu_result,
            "memory": self.memory_result,
            "devices": self.device_results,
            "system_snapshot": self.system_snapshot,
        }


class IntegratedExecutionManager:
    """Orchestrator coordinating execution cycles across CPU, Memory, Device, and Shared Monitoring."""

    def __init__(
        self,
        cpu_action_loop: Optional[CPUActionLoop] = None,
        memory_monitor: Optional[MemoryMonitor] = None,
        shared_monitor: Optional[SharedMonitor] = None,
        fleet: Optional[list[Device]] = None,
        max_history: int = 1000,
    ) -> None:
        """Initialize IntegratedExecutionManager ensuring shared instance consistency.

        The manager and SharedMonitor always share identical CPUActionLoop and
        MemoryMonitor instances to prevent state divergence.

        Args:
            cpu_action_loop: Optional CPUActionLoop instance.
            memory_monitor: Optional MemoryMonitor instance.
            shared_monitor: Optional SharedMonitor instance.
            fleet: Optional list of simulated Device objects.
        """
        if shared_monitor is not None:
            self.shared_monitor = shared_monitor

            if cpu_action_loop is not None:
                self.cpu_action_loop = cpu_action_loop
                self.shared_monitor.cpu_action_loop = cpu_action_loop
            else:
                self.cpu_action_loop = self.shared_monitor.cpu_action_loop

            if memory_monitor is not None:
                self.memory_monitor = memory_monitor
                self.shared_monitor.memory_monitor = memory_monitor
            else:
                self.memory_monitor = self.shared_monitor.memory_monitor
        else:
            self.cpu_action_loop = cpu_action_loop
            self.memory_monitor = (
                memory_monitor if memory_monitor is not None else MemoryMonitor()
            )
            self.shared_monitor = SharedMonitor(
                cpu_action_loop=self.cpu_action_loop,
                memory_monitor=self.memory_monitor,
                fleet=fleet,
            )

        self.max_history: int = max(1, max_history)
        self._history: list[IntegratedExecutionStep] = []
        # gunicorn serves requests on several threads; the counter and history must change together.
        self._record_lock = threading.Lock()
        self._step_counter: int = 0

    def _ensure_cpu_action_loop(self) -> CPUActionLoop:
        """Lazily initialize CPUActionLoop if not supplied at construction."""
        if self.cpu_action_loop is None:
            from ml.cpu.burst_predictor import CPUBurstPredictor
            from ml.cpu.workload_classifier import CPUWorkloadClassifier
            from modules.cpu.meta_scheduler import CPUMetaScheduler

            predictor = CPUBurstPredictor(random_state=42)
            predictor.train_on_synthetic_workload(num_processes=50, seed=42)
            classifier = CPUWorkloadClassifier(random_state=42)
            classifier.train_on_synthetic_dataset(seed=42, target_per_class=10)
            meta = CPUMetaScheduler(classifier=classifier, burst_predictor=predictor)
            self.cpu_action_loop = CPUActionLoop(meta_scheduler=meta)

            self.shared_monitor.cpu_action_loop = self.cpu_action_loop
        return self.cpu_action_loop

    def step(
        self,
        processes: list[Process],
        device_actions: Optional[list[dict[str, Any]]] = None,
        compare_with_baseline: bool = False,
        pid: Optional[int] = None,
    ) -> IntegratedExecutionStep:
        """Execute a coordinated multi-resource execution cycle.

        Args:
            processes: Non-empty list of Process objects for CPU execution.
            device_actions: Optional list of device failure-risk action dictionaries.
            compare_with_baseline: Whether to evaluate baseline CPU scheduler.
            pid: Optional process ID for process memory observation.

        Returns:
            IntegratedExecutionStep containing consolidated execution telemetry.

        Raises:
            ValueError: If inputs are invalid or empty.
            TypeError: If input element types are invalid.
            psutil.NoSuchProcess: If supplied PID does not exist.
        """

        if not processes:
            raise ValueError("Workload must contain at least one process.")
        if not isinstance(processes, list):
            raise TypeError("Field 'processes' must be a list of Process objects.")
        for i, p in enumerate(processes):
            if not isinstance(p, Process):
                raise TypeError(
                    f"Element at index {i} in processes must be a Process instance, got {type(p).__name__}"
                )
            if p.burst_time < 0:
                raise ValueError(f"Process at index {i} has negative burst_time: {p.burst_time}")


        if not isinstance(compare_with_baseline, bool):
            raise TypeError("compare_with_baseline must be a boolean.")


        if device_actions is not None:
            if not isinstance(device_actions, list):
                raise TypeError("device_actions must be a list of dictionaries when provided.")
            for idx, action_req in enumerate(device_actions):
                if not isinstance(action_req, dict):
                    raise TypeError(f"device_action at index {idx} must be a dictionary.")
                for req_key in ("device_id", "failure_probability", "device_status"):
                    if req_key not in action_req:
                        raise ValueError(f"device_action at index {idx} missing required key: {req_key}")

                dev_id = action_req["device_id"]
                if not isinstance(dev_id, str) or not dev_id.strip():
                    raise ValueError(f"device_action at index {idx} has invalid device_id: must be non-empty string")

                prob = action_req["failure_probability"]
                if isinstance(prob, bool) or not isinstance(prob, (int, float)):
                    raise TypeError(f"device_action at index {idx} has invalid failure_probability: must be float")
                if not (0.0 <= float(prob) <= 1.0):
                    raise ValueError(f"device_action at index {idx} failure_probability out of bounds [0.0, 1.0]: {prob}")

                status = action_req["device_status"]
                if not isinstance(status, str) or not status.strip():
                    raise ValueError(f"device_action at index {idx} has invalid device_status: must be non-empty string")


        if pid is not None:
            if isinstance(pid, bool) or not isinstance(pid, int) or pid < 0:
                raise ValueError(f"pid must be a non-negative integer, got {pid}")

        now_timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()


        cpu_loop = self._ensure_cpu_action_loop()
        cpu_step_record = cpu_loop.step(
            processes=processes,
            compare_with_baseline=compare_with_baseline,
        )
        cpu_result = cpu_step_record.to_dict()


        executed_device_results: list[dict[str, Any]] = []
        if device_actions:
            for action_dict in device_actions:
                dev_id = action_dict["device_id"].strip()
                prob = float(action_dict["failure_probability"])
                status = action_dict["device_status"].strip()
                backup_id = action_dict.get("backup_device_id")
                if backup_id is not None:
                    backup_id = str(backup_id).strip() if str(backup_id).strip() else None
                threshold = action_dict.get("threshold", 0.7)
                if threshold is not None:
                    threshold = float(threshold)

                pred_result: PredictiveActionResult = handle_predictive_action(
                    device_id=dev_id,
                    failure_probability=prob,
                    device_status=status,
                    backup_device_id=backup_id,
                    threshold=threshold,
                )
                self.shared_monitor.record_device_action(pred_result)
                executed_device_results.append(asdict(pred_result))


        memory_result = self.memory_monitor.take_snapshot(pid=pid)


        system_snapshot = self.shared_monitor.get_system_snapshot(
            include_history=False,
            pid=pid,
        )


        with self._record_lock:
            step_record = IntegratedExecutionStep(
                step_index=self._step_counter,
                timestamp=now_timestamp,
                cpu_result=cpu_result,
                memory_result=memory_result,
                device_results=executed_device_results,
                system_snapshot=system_snapshot,
            )

            self._history.append(step_record)
            if len(self._history) > self.max_history:
                del self._history[: -self.max_history]
            self._step_counter += 1

        return step_record

    def get_history(self) -> list[IntegratedExecutionStep]:
        """Return a copy of the in-memory integrated execution step history."""
        with self._record_lock:
            return list(self._history)

    def reset(self) -> None:
        """Reset execution history and cascade reset to SharedMonitor and CPUActionLoop exactly once."""
        with self._record_lock:
            self._history.clear()
            self._step_counter = 0
        if self.shared_monitor is not None:

            self.shared_monitor.reset()
        elif self.cpu_action_loop is not None:
            self.cpu_action_loop.reset()
