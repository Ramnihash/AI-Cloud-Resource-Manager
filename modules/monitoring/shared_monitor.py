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

        action_dict.setdefault("recorded_at", datetime.datetime.now(datetime.timezone.utc).isoformat())
        self._recent_device_actions.append(action_dict)
        if len(self._recent_device_actions) > self.max_recent_actions:
            self._recent_device_actions = self._recent_device_actions[-self.max_recent_actions:]

        self._apply_action_to_fleet(action_dict)

    def _apply_action_to_fleet(self, action_dict: dict[str, Any]) -> None:
        """Reflect a recovery outcome in the fleet: blocked I/O marks the device failed.

        A device whose I/O was blocked becomes ``failed`` (traffic may have been
        rerouted to a backup); a device reported healthy goes back to ``ok``.
        """
        recovery = action_dict.get("recovery_result")
        if not isinstance(recovery, dict):
            return
        for device in self.fleet:
            if device.device_id == action_dict.get("device_id"):
                device.status = "failed" if recovery.get("io_blocked") else "ok"

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


        memory_data = self.memory_monitor.take_snapshot(pid=pid)


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
        for device in self.fleet:
            device.status = "ok"
        self._sample_counter = 0
