"""Unit tests for the simulated device fleet in modules.device.fleet."""

import unittest
from modules.device.device import Device
from modules.device.fleet import create_default_fleet


class TestFleet(unittest.TestCase):
    """Test suite covering the creation and structure of the default device fleet."""

    def test_fleet_device_count(self):
        """1. Verify that create_default_fleet() returns exactly 4 devices."""
        fleet = create_default_fleet()
        self.assertEqual(len(fleet), 4)

    def test_fleet_device_ids(self):
        """2. Verify that device IDs are exactly ['disk_0', 'printer_0', 'NIC_0', 'sensor_0']."""
        fleet = create_default_fleet()
        expected_ids = ["disk_0", "printer_0", "NIC_0", "sensor_0"]
        actual_ids = [dev.device_id for dev in fleet]
        self.assertEqual(actual_ids, expected_ids)

    def test_fleet_device_types(self):
        """3. Verify that the corresponding device types are ['disk', 'printer', 'NIC', 'sensor']."""
        fleet = create_default_fleet()
        expected_types = ["disk", "printer", "NIC", "sensor"]
        actual_types = [dev.device_type for dev in fleet]
        self.assertEqual(actual_types, expected_types)

    def test_fleet_device_instances(self):
        """4. Verify that every returned object in the fleet is an instance of Device."""
        fleet = create_default_fleet()
        for dev in fleet:
            with self.subTest(device_id=getattr(dev, "device_id", None)):
                self.assertIsInstance(dev, Device)

    def test_fleet_determinism(self):
        """5. Verify that calling create_default_fleet() twice produces the same IDs/types without randomness."""
        fleet_1 = create_default_fleet()
        fleet_2 = create_default_fleet()

        self.assertEqual(
            [d.device_id for d in fleet_1],
            [d.device_id for d in fleet_2],
        )
        self.assertEqual(
            [d.device_type for d in fleet_1],
            [d.device_type for d in fleet_2],
        )
        self.assertEqual(
            [d.to_dict() for d in fleet_1],
            [d.to_dict() for d in fleet_2],
        )


if __name__ == "__main__":
    unittest.main()
