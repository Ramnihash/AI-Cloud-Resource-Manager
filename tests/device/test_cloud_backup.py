"""Tests for the optional S3 backup path (boto3 is mocked)."""

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
