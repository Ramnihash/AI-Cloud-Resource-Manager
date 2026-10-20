"""Orchestration layer for CP4.5 — failure-risk prediction to action.

Connects the SMART disk failure-risk prediction to the simulated backup and
device recovery actions in a single deterministic call.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from modules.device.backup import BackupResult
from modules.device.backup_manager import handle_failure_risk
from modules.device.recovery import RecoveryResult, recover_device


@dataclass(frozen=True)
class PredictiveActionResult:
    """Combined result of a failure-risk prediction and its follow-on actions.

    Attributes
    ----------
    device_id: str
        Identifier of the device.
    failure_probability: float
        Failure probability produced by the upstream classifier.
    backup_result: BackupResult
        Result of the simulated backup action.
    recovery_result: RecoveryResult
        Result of the simulated recovery / I/O rerouting action.
    """

    device_id: str
    failure_probability: float
    backup_result: BackupResult
    recovery_result: RecoveryResult


def handle_predictive_action(
    device_id: str,
    failure_probability: float,
    device_status: str,
    backup_device_id: str | None = None,
    threshold: float = 0.7,
    context: Optional[dict[str, Any]] = None,
) -> PredictiveActionResult:
    """Execute the full prediction-to-action pipeline for one device.

    This function is a thin orchestrator. It delegates all business logic and
    validation to the lower-level modules:

    - :func:`modules.device.backup_manager.handle_failure_risk` for the backup
      decision and simulated cloud-backup action.
    - :func:`modules.device.recovery.recover_device` for the simulated I/O
      blocking and rerouting action.

    Parameters
    ----------
    device_id: str
        Identifier of the device to act on.
    failure_probability: float
        Predicted failure probability (validated by ``handle_failure_risk``).
    device_status: str
        Current device health status (validated by ``recover_device``).
    backup_device_id: str | None, optional
        Identifier of a backup device to reroute I/O to, if available.
    threshold: float, default ``0.7``
        Failure-risk threshold for triggering a backup action.
    context: dict | None, optional
        Extra facts stored in the cloud backup record (telemetry, anomaly score).

    Returns
    -------
    PredictiveActionResult
        Frozen dataclass containing the device ID, the original failure
        probability, and the results of both the backup and recovery actions.

    Raises
    ------
    ValueError
        Propagated from the underlying backup or recovery modules when inputs
        are invalid.
    """

    backup_result = handle_failure_risk(
        device_id=device_id,
        failure_probability=failure_probability,
        threshold=threshold,
        context=context,
    )


    recovery_result = recover_device(
        device_id=device_id,
        device_status=device_status,
        backup_device_id=backup_device_id,
    )

    return PredictiveActionResult(
        device_id=device_id,
        failure_probability=failure_probability,
        backup_result=backup_result,
        recovery_result=recovery_result,
    )
