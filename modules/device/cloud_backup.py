"""Optional real cloud backup to Amazon S3 for the device failure-risk action.

Enabled only when the ``BACKUP_S3_BUCKET`` environment variable is set.
Credentials come from the environment's AWS identity (an IAM role on EC2);
nothing secret is read from or stored in the repository. boto3 is imported
lazily so the rest of the backend works without it.
"""

from __future__ import annotations

import base64
import datetime
import hashlib
import json
import os
import socket
from dataclasses import dataclass
from typing import Any, Optional

BUCKET_ENV = "BACKUP_S3_BUCKET"
PREFIX_ENV = "BACKUP_S3_PREFIX"
DEFAULT_PREFIX = "device-backups"
SCHEMA_VERSION = 2
MAX_ATTEMPTS = 3
CONNECT_TIMEOUT_S = 3
READ_TIMEOUT_S = 5


@dataclass(frozen=True)
class UploadReceipt:
    """What was stored: the ``s3://`` location, its size and SHA-256 digest."""

    location: str
    size_bytes: int
    sha256: str


class CloudBackupError(RuntimeError):
    """The S3 upload could not be completed."""


def configured_bucket() -> Optional[str]:
    """Return the S3 bucket name when real backups are enabled, else ``None``."""
    bucket = os.environ.get(BUCKET_ENV, "").strip()
    return bucket or None


def _get_client() -> Any:
    try:
        import boto3
    except ImportError as exc:
        raise CloudBackupError("boto3 is not installed on the backend") from exc
    from botocore.config import Config

    # Short timeouts and a few retries: the backend has one worker, so a stuck
    # upload would block every other request.
    config = Config(
        retries={"max_attempts": MAX_ATTEMPTS, "mode": "standard"},
        connect_timeout=CONNECT_TIMEOUT_S,
        read_timeout=READ_TIMEOUT_S,
    )
    return boto3.client("s3", config=config)


def upload_backup(
    device_id: str,
    failure_probability: float,
    threshold: float,
    context: Optional[dict[str, Any]] = None,
) -> str:
    """Upload a backup record and return its ``s3://`` location (see :func:`upload_backup_receipt`)."""
    return upload_backup_receipt(device_id, failure_probability, threshold, context).location


def upload_backup_receipt(
    device_id: str,
    failure_probability: float,
    threshold: float,
    context: Optional[dict[str, Any]] = None,
) -> UploadReceipt:
    """Upload a backup record for ``device_id`` and return where and what was stored.

    The simulated devices hold no files, so the object is a JSON record of the
    backup event: which device, the risk and threshold that triggered it, why,
    when, from which host, and optional ``context`` supplied by the caller (for
    example the latest telemetry and anomaly score). The record is marked
    ``"simulated": true`` because the device data is simulated. S3 verifies the
    SHA-256 of the body on upload, and the digest is stored as object metadata.

    Raises
    ------
    CloudBackupError
        If S3 is not configured, boto3 is missing, or the upload fails.
    """
    bucket = configured_bucket()
    if bucket is None:
        raise CloudBackupError(f"{BUCKET_ENV} is not set")

    now = datetime.datetime.now(datetime.timezone.utc)
    prefix = os.environ.get(PREFIX_ENV, DEFAULT_PREFIX).strip("/")
    key = f"{prefix}/{device_id}/{now.strftime('%Y%m%dT%H%M%S%fZ')}.json"
    record = {
        "device_id": device_id,
        "failure_probability": failure_probability,
        "threshold": threshold,
        "reason": f"failure_probability {failure_probability:.3f} reached threshold {threshold:.3f}",
        "backed_up_at": now.isoformat(),
        "source_host": socket.gethostname(),
        "simulated": True,
        "schema_version": SCHEMA_VERSION,
        "context": context or {},
    }
    body = json.dumps(record, indent=2, default=str).encode("utf-8")
    digest = hashlib.sha256(body)
    try:
        _get_client().put_object(
            Bucket=bucket,
            Key=key,
            Body=body,
            ContentType="application/json",
            ChecksumSHA256=base64.b64encode(digest.digest()).decode("ascii"),
            Metadata={"sha256": digest.hexdigest(), "device-id": device_id},
        )
    except CloudBackupError:
        raise
    except Exception as exc:
        raise CloudBackupError(f"S3 upload failed: {type(exc).__name__}") from exc
    return UploadReceipt(location=f"s3://{bucket}/{key}", size_bytes=len(body), sha256=digest.hexdigest())
