"""Simulated cloud backup action for device failure risk.

This module provides a deterministic, side‑effect‑free backup simulation
that can be used by the Device Management action layer (CP4.5).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from modules.device import cloud_backup

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BackupResult:
    """Result of a simulated backup operation.

    Attributes
    ----------
    device_id: str
        Identifier of the device for which the backup was considered.
    triggered: bool
        ``True`` if the backup logic decided to attempt a backup.
    success: bool
        ``True`` only when a backup was triggered and the simulated operation
        succeeded. The simulation always succeeds when triggered; with
        ``BACKUP_S3_BUCKET`` set, a real S3 upload is attempted and may fail.
    backup_location: Optional[str]
        Deterministic location string when the backup is successful, otherwise
        ``None``.
    message: str
        Human‑readable description of the outcome.
    """

    device_id: str
    triggered: bool
    success: bool
    backup_location: Optional[str]
    message: str


def _validate_probability(value: float, name: str) -> None:
    """Validate that a probability value is in the range ``[0.0, 1.0]``.

    Parameters
    ----------
    value: float
        The value to validate.
    name: str
        Name used in the error message.
    """
    if not (0.0 <= value <= 1.0):
        raise ValueError(f"{name} must be between 0.0 and 1.0 inclusive, got {value!r}")


def should_backup(failure_probability: float, threshold: float = 0.7) -> bool:
    """Determine whether a backup should be triggered.

    The function validates its inputs and returns ``True`` when the failure
    probability meets or exceeds the supplied ``threshold``.

    Parameters
    ----------
    failure_probability: float
        Predicted failure probability in the range ``[0.0, 1.0]``.
    threshold: float, default ``0.7``
        Decision threshold also in the range ``[0.0, 1.0]``.

    Returns
    -------
    bool
        ``True`` if ``failure_probability >= threshold``.

    Raises
    ------
    ValueError
        If either argument is outside the ``[0.0, 1.0]`` interval.
    """
    _validate_probability(failure_probability, "failure_probability")
    _validate_probability(threshold, "threshold")
    return failure_probability >= threshold


def create_backup(
    device_id: str,
    failure_probability: float,
    threshold: float = 0.7,
) -> BackupResult:
    """Create a deterministic simulated backup result.

    If the ``failure_probability`` is below the ``threshold`` the backup is not
    triggered and the result reflects a no‑op. When the probability meets or
    exceeds the threshold a deterministic backup location string is generated.

    Parameters
    ----------
    device_id: str
        Identifier of the device.
    failure_probability: float
        Predicted failure probability (validated same as :func:`should_backup`).
    threshold: float, default ``0.7``
        Decision threshold.

    Returns
    -------
    BackupResult
        A frozen dataclass instance describing the outcome.
    """
    triggered = should_backup(failure_probability, threshold)
    if not triggered:
        return BackupResult(
            device_id=device_id,
            triggered=False,
            success=False,
            backup_location=None,
            message="Backup not triggered: failure probability below threshold.",
        )


    if cloud_backup.configured_bucket() is not None:
        try:
            s3_location = cloud_backup.upload_backup(device_id, failure_probability, threshold)
        except cloud_backup.CloudBackupError as exc:
            logger.warning("Cloud backup failed for %s: %s", device_id, exc)
            return BackupResult(
                device_id=device_id,
                triggered=True,
                success=False,
                backup_location=None,
                message=f"Backup failed: {exc}",
            )
        return BackupResult(
            device_id=device_id,
            triggered=True,
            success=True,
            backup_location=s3_location,
            message="Backup uploaded to cloud storage.",
        )

    location = f"cloud://backup/{device_id}"
    return BackupResult(
        device_id=device_id,
        triggered=True,
        success=True,
        backup_location=location,
        message="Backup completed successfully.",
    )
