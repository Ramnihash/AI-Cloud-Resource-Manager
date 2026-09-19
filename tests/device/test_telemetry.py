"""Unit tests for normal telemetry generation in modules.device.telemetry."""

import unittest
from modules.device.device import Device, DeviceType
from modules.device.fleet import create_default_fleet
from modules.device.telemetry import DeviceTelemetry, generate_normal_telemetry


class TestTelemetry(unittest.TestCase):
    """Test suite covering normal telemetry generation and DeviceTelemetry dataclass."""

    def setUp(self):
        self.disk = Device(device_id="disk_0", device_type=DeviceType.DISK)
        self.timestamp = 1000.0

    def test_normal_telemetry_generation_for_disk_0(self):
        """1. A normal telemetry sample can be generated for disk_0."""
        sample = generate_normal_telemetry(self.disk, timestamp=self.timestamp)
        self.assertIsNotNone(sample)

    def test_returned_object_is_device_telemetry(self):
        """2. The returned object is DeviceTelemetry."""
        sample = generate_normal_telemetry(self.disk, timestamp=self.timestamp)
        self.assertIsInstance(sample, DeviceTelemetry)

    def test_device_id_is_preserved(self):
        """3. device_id is preserved."""
        sample = generate_normal_telemetry(self.disk, timestamp=self.timestamp)
        self.assertEqual(sample.device_id, "disk_0")

    def test_timestamp_is_preserved(self):
        """4. timestamp is preserved."""
        sample = generate_normal_telemetry(self.disk, timestamp=self.timestamp)
        self.assertEqual(sample.timestamp, self.timestamp)

    def test_all_telemetry_values_are_non_negative(self):
        """5. All telemetry values are non-negative across multiple sample indices."""
        for idx in range(15):
            with self.subTest(sample_index=idx):
                sample = generate_normal_telemetry(self.disk, timestamp=self.timestamp + idx, sample_index=idx)
                self.assertGreaterEqual(sample.temperature, 0.0)
                self.assertGreaterEqual(sample.error_rate, 0.0)
                self.assertGreaterEqual(sample.response_latency, 0.0)
                self.assertGreaterEqual(sample.queue_length, 0)
                self.assertGreaterEqual(sample.io_activity, 0.0)

    def test_error_rate_remains_low(self):
        """6. error_rate remains low for normal telemetry (e.g., < 0.01)."""
        for idx in range(20):
            sample = generate_normal_telemetry(self.disk, timestamp=self.timestamp, sample_index=idx)
            self.assertLess(sample.error_rate, 0.01)

    def test_generator_works_for_all_four_device_types(self):
        """7. The generator works for all four device types: disk, printer, NIC, sensor."""
        fleet = create_default_fleet()
        self.assertEqual(len(fleet), 4)
        for dev in fleet:
            with self.subTest(device_type=dev.device_type):
                sample = generate_normal_telemetry(dev, timestamp=self.timestamp, sample_index=1)
                self.assertIsInstance(sample, DeviceTelemetry)
                self.assertEqual(sample.device_id, dev.device_id)

    def test_determinism_with_identical_inputs(self):
        """8. Calling the generator twice with identical inputs produces identical telemetry."""
        sample_1 = generate_normal_telemetry(self.disk, timestamp=self.timestamp, sample_index=5)
        sample_2 = generate_normal_telemetry(self.disk, timestamp=self.timestamp, sample_index=5)
        self.assertEqual(sample_1, sample_2)
        self.assertEqual(sample_1.to_dict(), sample_2.to_dict())

    def test_different_sample_indices_produce_different_samples(self):
        """9. Different sample_index values can produce different normal samples."""
        sample_a = generate_normal_telemetry(self.disk, timestamp=self.timestamp, sample_index=0)
        sample_b = generate_normal_telemetry(self.disk, timestamp=self.timestamp, sample_index=5)
        self.assertNotEqual(sample_a, sample_b)

    def test_device_object_not_mutated(self):
        """10. The Device object is not mutated by telemetry generation."""
        initial_state = self.disk.to_dict()
        generate_normal_telemetry(self.disk, timestamp=self.timestamp, sample_index=2)
        generate_normal_telemetry(self.disk, timestamp=self.timestamp + 10, sample_index=7)
        self.assertEqual(self.disk.to_dict(), initial_state)


if __name__ == "__main__":
    unittest.main()
