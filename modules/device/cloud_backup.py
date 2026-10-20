"""Optional real cloud backup to Amazon S3 for the device failure-risk action.

Enabled only when the ``BACKUP_S3_BUCKET`` environment variable is set.
Credentials come from the environment's AWS identity (an IAM role on EC2);
nothing secret is read from or stored in the repository. boto3 is imported
lazily so the rest of the backend works without it.
"""

from __future__ import annotations

import datetime
import json
import os
import socket
from typing import Any, Optional

BUCKET_ENV = "BACKUP_S3_BUCKET"
PREFIX_ENV = "BACKUP_S3_PREFIX"
DEFAULT_PREFIX = "device-backups"


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
    return boto3.client("s3")


def upload_backup(device_id: str, failure_probability: float, threshold: float) -> str:
    """Upload a backup record for ``device_id`` and return its ``s3://`` location.

    The simulated devices hold no files, so the object is a JSON record of the
    backup event: which device, the risk that triggered it, and when.

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
        "backed_up_at": now.isoformat(),
        "source_host": socket.gethostname(),
    }
    try:
        _get_client().put_object(
            Bucket=bucket,
            Key=key,
            Body=json.dumps(record, indent=2).encode("utf-8"),
            ContentType="application/json",
        )
    except CloudBackupError:
        raise
    except Exception as exc:
        raise CloudBackupError(f"S3 upload failed: {type(exc).__name__}") from exc
    return f"s3://{bucket}/{key}"
