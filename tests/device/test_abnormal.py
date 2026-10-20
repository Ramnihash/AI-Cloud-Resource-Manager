"""Unit tests for abnormal telemetry generation in modules.device.abnormal."""

import unittest
from modules.device.device import Device, DeviceType
from modules.device.fleet import create_default_fleet
from modules.device.telemetry import DeviceTelemetry, generate_normal_telemetry
from modules.device.abnormal import AbnormalPattern, generate_abnormal_telemetry


class TestAbnormalTelemetry(unittest.TestCase):
    """Test suite covering abnormal telemetry generation and AbnormalPattern enum."""

    def setUp(self):
        self.disk = Device(device_id="disk_0", device_type=DeviceType.DISK)
        self.timestamp = 2000.0

    def test_abnormal_telemetry_generation_for_disk_0(self):
        """1. Abnormal telemetry can be generated for disk_0."""
        sample = generate_abnormal_telemetry(
            self.disk, timestamp=self.timestamp, pattern=AbnormalPattern.HIGH_TEMPERATURE
        )
        self.assertIsNotNone(sample)

    def test_returned_object_is_device_telemetry(self):
        """2. Returned object is DeviceTelemetry."""
        sample = generate_abnormal_telemetry(
            self.disk, timestamp=self.timestamp, pattern=AbnormalPattern.HIGH_LATENCY
        )
        self.assertIsInstance(sample, DeviceTelemetry)

    def test_device_id_is_preserved(self):
        """3. device_id is preserved."""
        for pattern in AbnormalPattern:
            with self.subTest(pattern=pattern):
                sample = generate_abnormal_telemetry(self.disk, timestamp=self.timestamp, pattern=pattern)
                self.assertEqual(sample.device_id, "disk_0")

    def test_timestamp_is_preserved(self):
        """4. timestamp is preserved."""
        for pattern in AbnormalPattern:
            with self.subTest(pattern=pattern):
                sample = generate_abnormal_telemetry(self.disk, timestamp=self.timestamp, pattern=pattern)
                self.assertEqual(sample.timestamp, self.timestamp)

    def test_high_temperature_pattern(self):
        """5. HIGH_TEMPERATURE produces temperature higher than the corresponding normal sample."""
        for idx in range(5):
            with self.subTest(sample_index=idx):
                normal = generate_normal_telemetry(self.disk, timestamp=self.timestamp, sample_index=idx)
                abnormal = generate_abnormal_telemetry(
                    self.disk, timestamp=self.timestamp, pattern=AbnormalPattern.HIGH_TEMPERATURE, sample_index=idx
                )
                self.assertGreater(abnormal.temperature, normal.temperature)

    def test_high_error_rate_pattern(self):
        """6. HIGH_ERROR_RATE produces error_rate higher than the corresponding normal sample."""
        for idx in range(5):
            with self.subTest(sample_index=idx):
                normal = generate_normal_telemetry(self.disk, timestamp=self.timestamp, sample_index=idx)
                abnormal = generate_abnormal_telemetry(
                    self.disk, timestamp=self.timestamp, pattern=AbnormalPattern.HIGH_ERROR_RATE, sample_index=idx
                )
                self.assertGreater(abnormal.error_rate, normal.error_rate)

    def test_high_latency_pattern(self):
        """7. HIGH_LATENCY produces response_latency higher than the corresponding normal sample."""
        for idx in range(5):
            with self.subTest(sample_index=idx):
                normal = generate_normal_telemetry(self.disk, timestamp=self.timestamp, sample_index=idx)
                abnormal = generate_abnormal_telemetry(
                    self.disk, timestamp=self.timestamp, pattern=AbnormalPattern.HIGH_LATENCY, sample_index=idx
                )
                self.assertGreater(abnormal.response_latency, normal.response_latency)

    def test_high_queue_pattern(self):
        """8. HIGH_QUEUE produces queue_length higher than the corresponding normal sample."""
        for idx in range(5):
            with self.subTest(sample_index=idx):
                normal = generate_normal_telemetry(self.disk, timestamp=self.timestamp, sample_index=idx)
                abnormal = generate_abnormal_telemetry(
                    self.disk, timestamp=self.timestamp, pattern=AbnormalPattern.HIGH_QUEUE, sample_index=idx
                )
                self.assertGreater(abnormal.queue_length, normal.queue_length)

    def test_abnormal_io_pattern(self):
        """9. ABNORMAL_IO produces io_activity different from the corresponding normal sample."""
        for idx in range(5):
            with self.subTest(sample_index=idx):
                normal = generate_normal_telemetry(self.disk, timestamp=self.timestamp, sample_index=idx)
                abnormal = generate_abnormal_telemetry(
                    self.disk, timestamp=self.timestamp, pattern=AbnormalPattern.ABNORMAL_IO, sample_index=idx
                )
                self.assertNotEqual(abnormal.io_activity, normal.io_activity)

    def test_all_abnormal_values_remain_non_negative(self):
        """10. All abnormal values remain non-negative across all patterns."""
        for pattern in AbnormalPattern:
            for idx in range(5):
                with self.subTest(pattern=pattern, sample_index=idx):
                    sample = generate_abnormal_telemetry(
                        self.disk, timestamp=self.timestamp + idx, pattern=pattern, sample_index=idx
                    )
                    self.assertGreaterEqual(sample.temperature, 0.0)
                    self.assertGreaterEqual(sample.error_rate, 0.0)
                    self.assertGreaterEqual(sample.response_latency, 0.0)
                    self.assertGreaterEqual(sample.queue_length, 0)
                    self.assertGreaterEqual(sample.io_activity, 0.0)

    def test_generator_works_for_all_four_device_types(self):
        """11. The generator works for all four device types: disk, printer, NIC, sensor."""
        fleet = create_default_fleet()
        for dev in fleet:
            for pattern in AbnormalPattern:
                with self.subTest(device_type=dev.device_type, pattern=pattern):
                    sample = generate_abnormal_telemetry(
                        dev, timestamp=self.timestamp, pattern=pattern, sample_index=1
                    )
                    self.assertIsInstance(sample, DeviceTelemetry)
                    self.assertEqual(sample.device_id, dev.device_id)

    def test_determinism_with_identical_inputs(self):
        """12. Identical inputs produce identical abnormal telemetry."""
        sample_1 = generate_abnormal_telemetry(
            self.disk, timestamp=self.timestamp, pattern=AbnormalPattern.HIGH_TEMPERATURE, sample_index=3
        )
        sample_2 = generate_abnormal_telemetry(
            self.disk, timestamp=self.timestamp, pattern=AbnormalPattern.HIGH_TEMPERATURE, sample_index=3
        )
        self.assertEqual(sample_1, sample_2)
        self.assertEqual(sample_1.to_dict(), sample_2.to_dict())

    def test_device_object_not_mutated(self):
        """13. The original Device object is not mutated."""
        initial_state = self.disk.to_dict()
        for pattern in AbnormalPattern:
            generate_abnormal_telemetry(self.disk, timestamp=self.timestamp, pattern=pattern, sample_index=1)
        self.assertEqual(self.disk.to_dict(), initial_state)

    def test_unrelated_telemetry_fields_remain_unchanged(self):
        """14. Unrelated telemetry fields remain unchanged for each pattern, except the intentionally modified field."""
        sample_idx = 4
        normal = generate_normal_telemetry(self.disk, timestamp=self.timestamp, sample_index=sample_idx)


        abn_temp = generate_abnormal_telemetry(
            self.disk, timestamp=self.timestamp, pattern=AbnormalPattern.HIGH_TEMPERATURE, sample_index=sample_idx
        )
        self.assertNotEqual(abn_temp.temperature, normal.temperature)
        self.assertEqual(abn_temp.error_rate, normal.error_rate)
        self.assertEqual(abn_temp.response_latency, normal.response_latency)
        self.assertEqual(abn_temp.queue_length, normal.queue_length)
        self.assertEqual(abn_temp.io_activity, normal.io_activity)


        abn_err = generate_abnormal_telemetry(
            self.disk, timestamp=self.timestamp, pattern=AbnormalPattern.HIGH_ERROR_RATE, sample_index=sample_idx
        )
        self.assertEqual(abn_err.temperature, normal.temperature)
        self.assertNotEqual(abn_err.error_rate, normal.error_rate)
        self.assertEqual(abn_err.response_latency, normal.response_latency)
        self.assertEqual(abn_err.queue_length, normal.queue_length)
        self.assertEqual(abn_err.io_activity, normal.io_activity)


        abn_lat = generate_abnormal_telemetry(
            self.disk, timestamp=self.timestamp, pattern=AbnormalPattern.HIGH_LATENCY, sample_index=sample_idx
        )
        self.assertEqual(abn_lat.temperature, normal.temperature)
        self.assertEqual(abn_lat.error_rate, normal.error_rate)
        self.assertNotEqual(abn_lat.response_latency, normal.response_latency)
        self.assertEqual(abn_lat.queue_length, normal.queue_length)
        self.assertEqual(abn_lat.io_activity, normal.io_activity)


        abn_q = generate_abnormal_telemetry(
            self.disk, timestamp=self.timestamp, pattern=AbnormalPattern.HIGH_QUEUE, sample_index=sample_idx
        )
        self.assertEqual(abn_q.temperature, normal.temperature)
        self.assertEqual(abn_q.error_rate, normal.error_rate)
        self.assertEqual(abn_q.response_latency, normal.response_latency)
        self.assertNotEqual(abn_q.queue_length, normal.queue_length)
        self.assertEqual(abn_q.io_activity, normal.io_activity)


        abn_io = generate_abnormal_telemetry(
            self.disk, timestamp=self.timestamp, pattern=AbnormalPattern.ABNORMAL_IO, sample_index=sample_idx
        )
        self.assertEqual(abn_io.temperature, normal.temperature)
        self.assertEqual(abn_io.error_rate, normal.error_rate)
        self.assertEqual(abn_io.response_latency, normal.response_latency)
        self.assertEqual(abn_io.queue_length, normal.queue_length)
        self.assertNotEqual(abn_io.io_activity, normal.io_activity)


if __name__ == "__main__":
    unittest.main()


def test_severity_scales_the_fault_and_default_is_unchanged():
    import pytest

    from modules.device.abnormal import generate_abnormal_telemetry
    from modules.device.fleet import create_default_fleet
    from modules.device.telemetry import generate_normal_telemetry

    device = create_default_fleet()[0]
    normal = generate_normal_telemetry(device, 5.0, 5)
    full = generate_abnormal_telemetry(device, 5.0, "high_temperature", 5)
    assert full == generate_abnormal_telemetry(device, 5.0, "high_temperature", 5, severity=1.0)
    mild = generate_abnormal_telemetry(device, 5.0, "high_temperature", 5, severity=0.25)
    assert 0 < mild.temperature - normal.temperature < full.temperature - normal.temperature
    for bad in (0, -0.1, 1.5):
        with pytest.raises(ValueError):
            generate_abnormal_telemetry(device, 5.0, "high_temperature", 5, severity=bad)
