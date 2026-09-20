"""Process model and PCB representation for CPU scheduling."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class Process:
    """Represents a process control block (PCB) for CPU scheduling.

    Supports both input workload parameters and execution state required for
    computing scheduling performance metrics.
    """

    pid: int | str
    arrival_time: float
    burst_time: float
    priority: int = 0

    # Execution state (populated during/after scheduling)
    remaining_time: float = field(init=False)
    start_time: Optional[float] = field(default=None, init=False)
    completion_time: Optional[float] = field(default=None, init=False)
    waiting_time: float = field(default=0.0, init=False)
    turnaround_time: float = field(default=0.0, init=False)
    response_time: float = field(default=0.0, init=False)

    # Metadata hook for future ML integration (features, predictions, classification)
    predicted_burst_time: Optional[float] = None
    workload_type: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.arrival_time < 0:
            raise ValueError(f"Process {self.pid}: arrival_time cannot be negative ({self.arrival_time})")
        if self.burst_time < 0:
            raise ValueError(f"Process {self.pid}: burst_time cannot be negative ({self.burst_time})")

        self.remaining_time = float(self.burst_time)
        self.arrival_time = float(self.arrival_time)
        self.burst_time = float(self.burst_time)

    @property
    def is_completed(self) -> bool:
        """Returns True if the process has finished execution."""
        return self.remaining_time <= 0 and self.completion_time is not None

    def copy(self) -> Process:
        """Create an independent copy of this process to protect caller workloads."""
        return copy.deepcopy(self)

    def reset_execution_state(self) -> None:
        """Reset execution attributes back to pristine state for re-scheduling."""
        self.remaining_time = float(self.burst_time)
        self.start_time = None
        self.completion_time = None
        self.waiting_time = 0.0
        self.turnaround_time = 0.0
        self.response_time = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize process state to dictionary."""
        return {
            "pid": self.pid,
            "arrival_time": self.arrival_time,
            "burst_time": self.burst_time,
            "priority": self.priority,
            "remaining_time": self.remaining_time,
            "start_time": self.start_time,
            "completion_time": self.completion_time,
            "waiting_time": self.waiting_time,
            "turnaround_time": self.turnaround_time,
            "response_time": self.response_time,
            "predicted_burst_time": self.predicted_burst_time,
            "workload_type": self.workload_type,
            "metadata": copy.deepcopy(self.metadata),
        }
