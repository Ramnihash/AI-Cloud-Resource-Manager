"""Deterministic simulation of device recovery / I/O rerouting (CP4.5)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class RecoveryResult:
    """Result of a simulated recovery/rerouting action.

    Attributes
    ----------
    device_id: str
        Identifier of the device under consideration.
    recovered: bool
        ``True`` when a backup device was provided and the device was recovered.
    io_blocked: bool
        ``True`` when I/O to the device is blocked.
    rerouted: bool
        ``True`` when I/O is rerouted to a backup device.
    backup_device_id: str | None
        Identifier of the backup device used for rerouting, if any.
    message: str
        Human‑readable description of the outcome.
    """

    device_id: str
    recovered: bool
    io_blocked: bool
    rerouted: bool
    backup_device_id: Optional[str]
    message: str


def _validate_nonempty(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non‑empty string")


def recover_device(
    device_id: str,
    device_status: str,
    backup_device_id: str | None = None,
) -> RecoveryResult:
    """Simulate recovery actions based on device status.

    Parameters
    ----------
    device_id: str
        Identifier of the device.
    device_status: str
        Current status – must be one of the recognized healthy or unhealthy
        values (case‑insensitive, whitespace trimmed).
    backup_device_id: str | None, optional
        Identifier of a healthy backup device for rerouting.

    Returns
    -------
    RecoveryResult
        Deterministic result describing what would happen.

    Raises
    ------
    ValueError
        If any argument is invalid or ``device_status`` is unsupported.
    """
    # Validate inputs
    _validate_nonempty(device_id, "device_id")
    _validate_nonempty(device_status, "device_status")
    if backup_device_id is not None:
        _validate_nonempty(backup_device_id, "backup_device_id")

    status = device_status.strip().lower()
    healthy = {"healthy", "available", "normal"}
    unhealthy = {"unhealthy", "degrading", "failed"}

    if status in healthy:
        return RecoveryResult(
            device_id=device_id,
            recovered=False,
            io_blocked=False,
            rerouted=False,
            backup_device_id=None,
            message="Recovery not required: device is healthy.",
        )

    if status in unhealthy:
        # Block I/O regardless of backup availability
        if backup_device_id is not None:
            return RecoveryResult(
                device_id=device_id,
                recovered=True,
                io_blocked=True,
                rerouted=True,
                backup_device_id=backup_device_id,
                message=(
                    f"I/O blocked for device {device_id} and rerouted to backup "
                    f"device {backup_device_id}."
                ),
            )
        else:
            return RecoveryResult(
                device_id=device_id,
                recovered=False,
                io_blocked=True,
                rerouted=False,
                backup_device_id=None,
                message=f"I/O blocked for device {device_id}, but no backup device was provided.",
            )

    raise ValueError(f"Unsupported device_status: {device_status!r}")
