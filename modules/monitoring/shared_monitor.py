"""Shared Monitoring Service for CP5.2.

Provides an in-memory cross-subsystem telemetry aggregation layer connecting:
- CPU: CPUActionLoop simulation metrics and execution history
- Memory: MemoryMonitor real-time host RAM/swap snapshots
- Devices: Simulated device fleet status and telemetry
- Predictive Actions: Recent device failure-risk action records
"""

from __future__ import annotations

import datetime
from dataclasses import asdict, is_dataclass
import time
from typing import Any, Optional

from modules.cpu.action_loop import CPUActionLoop
from modules.device.device import Device
from modules.device.fleet import create_default_fleet
from modules.device.predictive_action import PredictiveActionResult
from modules.device.telemetry import generate_normal_telemetry
from modules.memory.memory_monitor import MemoryMonitor


class SharedMonitor:
    """In-memory cross-subsystem monitoring service."""

    def __init__(
        self,
        cpu_action_loop: Optional[CPUActionLoop] = None,
        memory_monitor: Optional[MemoryMonitor] = None,
        fleet: Optional[list[Device]] = None,
        max_recent_actions: int = 50,
    ) -> None:
        """Initialize SharedMonitor with injected or default subsystem components.

        Args:
            cpu_action_loop: Optional CPUActionLoop instance for CPU metrics.
            memory_monitor: Optional MemoryMonitor instance for memory metrics.
            fleet: Optional list of Device objects for device telemetry.
            max_recent_actions: Maximum number of recent device actions to retain in-memory.
        """
        self.cpu_action_loop: Optional[CPUActionLoop] = cpu_action_loop
        self.memory_monitor: MemoryMonitor = (
            memory_monitor if memory_monitor is not None else MemoryMonitor()
        )
        self.fleet: list[Device] = list(fleet) if fleet is not None else create_default_fleet()
        self.max_recent_actions: int = max(1, max_recent_actions)
        self._recent_device_actions: list[dict[str, Any]] = []
        self._sample_counter: int = 0

    def record_device_action(
        self,
        action_result: PredictiveActionResult | dict[str, Any],
    ) -> None:
        """Record a device predictive action result in recent history.

        Args:
            action_result: PredictiveActionResult instance or dictionary representation.

        Raises:
            TypeError: If action_result is neither a PredictiveActionResult nor a dictionary.
        """
        if is_dataclass(action_result):
            action_dict = asdict(action_result)
        elif isinstance(action_result, dict):
            action_dict = dict(action_result)
        else:
            raise TypeError(
                f"action_result must be a PredictiveActionResult or dict, got {type(action_result).__name__}"
            )

        self._recent_device_actions.append(action_dict)
        if len(self._recent_device_actions) > self.max_recent_actions:
            self._recent_device_actions = self._recent_device_actions[-self.max_recent_actions:]

    def get_system_snapshot(
        self,
        include_history: bool = False,
        pid: Optional[int] = None,
    ) -> dict[str, Any]:
        """Aggregate cross-subsystem telemetry into a unified snapshot dictionary.

        Args:
            include_history: If True, include historical step records for CPU action loop.
            pid: Optional process ID to query per-process memory metrics.

        Returns:
            Dictionary containing timestamp, overall status, and subsystem metrics for
            CPU, Memory, and Devices.
        """
        now_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
        now_ts = time.time()

        # 1. CPU Telemetry (reusing existing CPUActionLoop APIs)
        if self.cpu_action_loop is not None:
            cpu_summary = self.cpu_action_loop.get_summary()
            cpu_data: dict[str, Any] = dict(cpu_summary)
            if include_history:
                cpu_data["history"] = [
                    s.to_dict() for s in self.cpu_action_loop.get_history()
                ]
        else:
            cpu_data = {
                "total_steps": 0,
                "total_processes": 0,
                "scheduler_distribution": {},
                "scheduler_percentages": {},
                "fallback_count": 0,
                "fallback_rate": 0.0,
                "mean_metrics": {},
                "mean_action_delta": None,
            }
            if include_history:
                cpu_data["history"] = []

        # 2. Memory Telemetry (reusing existing MemoryMonitor API)
        memory_data = self.memory_monitor.take_snapshot(pid=pid)

        # 3. Device Telemetry (reusing existing device fleet and telemetry APIs)
        devices_data: list[dict[str, Any]] = []
        for device in self.fleet:
            telemetry = generate_normal_telemetry(
                device=device,
                timestamp=now_ts,
                sample_index=self._sample_counter,
            )
            device_entry = device.to_dict()
            device_entry["telemetry"] = telemetry.to_dict()
            devices_data.append(device_entry)
        self._sample_counter += 1

        # 4. Status Determination
        # Use neutral status representation based on existing device statuses from modules.device.recovery
        # without inventing new arbitrary metric thresholds.
        any_unhealthy = any(
            getattr(d, "status", "").strip().lower() in {"unhealthy", "degrading", "failed"}
            for d in self.fleet
        )
        overall_status = "degraded" if any_unhealthy else "ok"

        return {
            "timestamp": now_utc,
            "status": overall_status,
            "cpu": cpu_data,
            "memory": memory_data,
            "devices": {
                "device_count": len(self.fleet),
                "devices": devices_data,
                "recent_actions": list(self._recent_device_actions),
            },
        }

    def reset(self) -> None:
        """Reset in-memory history and counters."""
        if self.cpu_action_loop is not None:
            self.cpu_action_loop.reset()
        self._recent_device_actions.clear()
        self._sample_counter = 0
