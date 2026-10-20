"""Tests for the optional S3 backup path (boto3 is mocked)."""

import base64
import hashlib
import json
import os
import unittest
from unittest.mock import MagicMock, patch

from modules.device import cloud_backup
from modules.device.backup import create_backup


class TestCloudBackup(unittest.TestCase):
    def test_disabled_without_bucket(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(cloud_backup.configured_bucket())
            result = create_backup("disk_0", 0.9)
        self.assertEqual(result.backup_location, "cloud://backup/disk_0")

    def test_upload_writes_json_record(self) -> None:
        client = MagicMock()
        with patch.dict(os.environ, {"BACKUP_S3_BUCKET": "my-bucket"}, clear=True), \
                patch.object(cloud_backup, "_get_client", return_value=client):
            location = cloud_backup.upload_backup("disk_0", 0.9, 0.7)
        kwargs = client.put_object.call_args.kwargs
        self.assertEqual(kwargs["Bucket"], "my-bucket")
        self.assertTrue(kwargs["Key"].startswith("device-backups/disk_0/"))
        self.assertEqual(location, "s3://my-bucket/" + kwargs["Key"])
        record = json.loads(kwargs["Body"].decode("utf-8"))
        self.assertEqual(record["device_id"], "disk_0")
        self.assertEqual(record["failure_probability"], 0.9)

    def test_record_has_reason_context_and_integrity(self) -> None:
        client = MagicMock()
        context = {"anomaly_score": 0.31, "health_state": "degrading", "latest_telemetry": {"temperature": 71.0}}
        with patch.dict(os.environ, {"BACKUP_S3_BUCKET": "my-bucket"}, clear=True), \
                patch.object(cloud_backup, "_get_client", return_value=client):
            receipt = cloud_backup.upload_backup_receipt("disk_0", 0.9, 0.7, context)
        kwargs = client.put_object.call_args.kwargs
        record = json.loads(kwargs["Body"].decode("utf-8"))
        self.assertIn("0.900", record["reason"])
        self.assertIn("0.700", record["reason"])
        self.assertEqual(record["context"], context)
        self.assertTrue(record["simulated"])
        self.assertEqual(record["schema_version"], cloud_backup.SCHEMA_VERSION)
        digest = hashlib.sha256(kwargs["Body"])
        self.assertEqual(receipt.sha256, digest.hexdigest())
        self.assertEqual(receipt.size_bytes, len(kwargs["Body"]))
        self.assertEqual(kwargs["ChecksumSHA256"], base64.b64encode(digest.digest()).decode("ascii"))
        self.assertEqual(kwargs["Metadata"]["sha256"], digest.hexdigest())

    def test_create_backup_returns_digest_and_size(self) -> None:
        client = MagicMock()
        with patch.dict(os.environ, {"BACKUP_S3_BUCKET": "my-bucket"}, clear=True), \
                patch.object(cloud_backup, "_get_client", return_value=client):
            result = create_backup("disk_0", 0.9, context={"note": "x"})
        self.assertEqual(len(result.sha256), 64)
        self.assertGreater(result.size_bytes, 0)

    def test_simulated_backup_has_no_digest(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            result = create_backup("disk_0", 0.9)
        self.assertIsNone(result.sha256)
        self.assertIsNone(result.size_bytes)

    def test_client_uses_timeouts_and_retries(self) -> None:
        try:
            import boto3  # noqa: F401
        except ImportError:
            self.skipTest("boto3 not installed")
        with patch("boto3.client") as make_client:
            cloud_backup._get_client()
        config = make_client.call_args.kwargs["config"]
        self.assertEqual(config.retries["max_attempts"], cloud_backup.MAX_ATTEMPTS)
        self.assertEqual(config.connect_timeout, cloud_backup.CONNECT_TIMEOUT_S)
        self.assertEqual(config.read_timeout, cloud_backup.READ_TIMEOUT_S)

    @unittest.skipUnless(
        os.environ.get("RUN_S3_INTEGRATION") == "1" and os.environ.get("BACKUP_S3_BUCKET"),
        "set RUN_S3_INTEGRATION=1 and BACKUP_S3_BUCKET to test against a real bucket",
    )
    def test_real_bucket_roundtrip(self) -> None:
        receipt = cloud_backup.upload_backup_receipt("integration_test", 0.95, 0.7, {"source": "integration test"})
        bucket, key = receipt.location[len("s3://"):].split("/", 1)
        client = cloud_backup._get_client()
        try:
            obj = client.get_object(Bucket=bucket, Key=key)
            body = obj["Body"].read()
            self.assertEqual(hashlib.sha256(body).hexdigest(), receipt.sha256)
            self.assertEqual(len(body), receipt.size_bytes)
            self.assertEqual(obj["Metadata"]["sha256"], receipt.sha256)
            self.assertEqual(json.loads(body)["context"]["source"], "integration test")
        finally:
            client.delete_object(Bucket=bucket, Key=key)

    def test_create_backup_uses_s3_when_configured(self) -> None:
        client = MagicMock()
        with patch.dict(os.environ, {"BACKUP_S3_BUCKET": "my-bucket", "BACKUP_S3_PREFIX": "x/"}, clear=True), \
                patch.object(cloud_backup, "_get_client", return_value=client):
            result = create_backup("NIC_0", 0.8)
        self.assertTrue(result.triggered and result.success)
        self.assertTrue(result.backup_location.startswith("s3://my-bucket/x/NIC_0/"))

    def test_below_threshold_never_touches_s3(self) -> None:
        client = MagicMock()
        with patch.dict(os.environ, {"BACKUP_S3_BUCKET": "my-bucket"}, clear=True), \
                patch.object(cloud_backup, "_get_client", return_value=client):
            result = create_backup("disk_0", 0.2)
        self.assertFalse(result.triggered)
        client.put_object.assert_not_called()

    def test_s3_failure_is_reported_not_raised(self) -> None:
        client = MagicMock()
        client.put_object.side_effect = RuntimeError("AccessDenied secret-detail")
        with patch.dict(os.environ, {"BACKUP_S3_BUCKET": "my-bucket"}, clear=True), \
                patch.object(cloud_backup, "_get_client", return_value=client):
            result = create_backup("disk_0", 0.9)
        self.assertTrue(result.triggered)
        self.assertFalse(result.success)
        self.assertIsNone(result.backup_location)
        self.assertIn("S3 upload failed", result.message)
        self.assertNotIn("secret-detail", result.message)

    def test_missing_boto3_is_reported(self) -> None:
        with patch.dict(os.environ, {"BACKUP_S3_BUCKET": "my-bucket"}, clear=True), \
                patch.object(cloud_backup, "_get_client", side_effect=cloud_backup.CloudBackupError("boto3 is not installed on the backend")):
            result = create_backup("disk_0", 0.9)
        self.assertFalse(result.success)
        self.assertIn("boto3", result.message)


if __name__ == "__main__":
    unittest.main()
