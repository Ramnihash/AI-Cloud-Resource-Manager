"""Unit tests for the Device class and DeviceType enum in modules.device.device."""

import unittest
from modules.device.device import Device, DeviceType


class TestDevice(unittest.TestCase):
    """Test suite covering the simulated Device object and state."""

    def test_valid_disk_device_and_defaults(self):
        """1. Create a valid disk device and verify ID, type, and all default state fields."""
        dev = Device(device_id="disk_0", device_type="disk")

        self.assertEqual(dev.device_id, "disk_0")
        self.assertEqual(dev.device_type, "disk")
        self.assertEqual(dev.status, "ok")
        self.assertEqual(dev.temperature, 35.0)
        self.assertEqual(dev.error_rate, 0.0)
        self.assertEqual(dev.response_latency, 0.0)
        self.assertEqual(dev.queue_length, 0)

    def test_all_four_allowed_device_types(self):
        """2. Create valid devices for all four allowed device types: disk, printer, NIC, sensor."""
        allowed_types = ["disk", "printer", "NIC", "sensor"]
        for dev_type in allowed_types:
            with self.subTest(device_type=dev_type):
                dev = Device(device_id=f"dev_{dev_type}", device_type=dev_type)
                self.assertEqual(dev.device_type, dev_type)

    def test_invalid_device_type_raises_value_error(self):
        """3. Verify that an invalid device_type raises ValueError."""
        invalid_types = ["gpu", "tpu", "nic", "DISK", "unknown", ""]
        for invalid_type in invalid_types:
            with self.subTest(invalid_type=invalid_type):
                with self.assertRaises(ValueError):
                    Device(device_id="bad_dev", device_type=invalid_type)

    def test_device_created_with_device_type_enum_normalized(self):
        """4. Verify that a Device created using DeviceType.DISK is normalized correctly."""
        dev = Device(device_id="disk_enum", device_type=DeviceType.DISK)
        self.assertEqual(dev.device_type, "disk")
        self.assertIsInstance(dev.device_type, str)

    def test_to_dict_returns_expected_fields_and_values(self):
        """5. Verify that to_dict() returns the expected device state fields and values."""
        dev = Device(
            device_id="sensor_01",
            device_type="sensor",
            status="active",
            temperature=42.5,
            error_rate=0.01,
            response_latency=2.4,
            queue_length=3,
        )
        expected_dict = {
            "device_id": "sensor_01",
            "device_type": "sensor",
            "status": "active",
            "temperature": 42.5,
            "error_rate": 0.01,
            "response_latency": 2.4,
            "queue_length": 3,
        }
        self.assertEqual(dev.to_dict(), expected_dict)


if __name__ == "__main__":
    unittest.main()
