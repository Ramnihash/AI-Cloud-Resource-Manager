"""Orchestration layer for disk failure‑risk → backup action (CP4.5)."""

from __future__ import annotations

from modules.device.backup import BackupResult, create_backup


def handle_failure_risk(
    device_id: str,
    failure_probability: float,
    threshold: float = 0.7,
) -> BackupResult:
    """Handle a disk's failure‑risk prediction.

    This thin wrapper delegates to :func:`modules.device.backup.create_backup`
    and returns the resulting :class:`modules.device.backup.BackupResult`.
    It does **not** duplicate any threshold logic – the underlying function
    performs validation and the decision.

    Parameters
    ----------
    device_id: str
        Identifier of the device for which the risk was predicted.
    failure_probability: float
        Predicted probability (validated by ``create_backup``).
    threshold: float, default ``0.7``
        Decision threshold (validated by ``create_backup``).

    Returns
    -------
    BackupResult
        The result of the simulated backup action.
    """
    return create_backup(device_id, failure_probability, threshold)
