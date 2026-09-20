"""Abnormal and degrading telemetry generator for simulated OS devices.

This module provides deterministic injection of abnormal telemetry patterns
(e.g., thermal spikes, elevated error rates, latency degradation, queue congestion,
and abnormal I/O surges) for testing predictive maintenance and anomaly detection.
"""

from dataclasses import replace
from enum import Enum
import math
from typing import Union

from modules.device.device import Device
from modules.device.telemetry import DeviceTelemetry, generate_normal_telemetry


class AbnormalPattern(str, Enum):
    """Supported abnormal telemetry patterns for simulated devices."""
    HIGH_TEMPERATURE = "high_temperature"
    HIGH_ERROR_RATE = "high_error_rate"
    HIGH_LATENCY = "high_latency"
    HIGH_QUEUE = "high_queue"
    ABNORMAL_IO = "abnormal_io"


def generate_abnormal_telemetry(
    device: Device,
    timestamp: float,
    pattern: Union[AbnormalPattern, str],
    sample_index: int = 0,
) -> DeviceTelemetry:
    """Generate deterministic abnormal telemetry by modifying a specific baseline metric.

    Starts from the deterministic normal telemetry baseline and introduces a
    controlled degradation/anomaly corresponding to the chosen AbnormalPattern.
    All other telemetry fields are strictly preserved.

    Args:
        device: The simulated Device object to measure (not mutated).
        timestamp: Time of the telemetry measurement.
        pattern: The abnormal pattern to simulate.
        sample_index: Sequence or tick index for smooth variation.

    Returns:
        DeviceTelemetry: Telemetry with the targeted field elevated/altered.

    Raises:
        ValueError: If an unsupported abnormal pattern is supplied.
    """
    normal = generate_normal_telemetry(device, timestamp, sample_index)

    if isinstance(pattern, str):
        try:
            pattern = AbnormalPattern(pattern)
        except ValueError:
            raise ValueError(f"Unsupported abnormal pattern: {pattern}")

    if pattern == AbnormalPattern.HIGH_TEMPERATURE:
        elevated_temp = round(normal.temperature + 45.0 + 2.0 * math.sin(sample_index * 0.1), 2)
        return replace(normal, temperature=max(0.0, elevated_temp))

    elif pattern == AbnormalPattern.HIGH_ERROR_RATE:
        elevated_err = round(normal.error_rate + 0.25 + 0.05 * math.sin(sample_index * 0.1), 5)
        return replace(normal, error_rate=max(0.0, elevated_err))

    elif pattern == AbnormalPattern.HIGH_LATENCY:
        elevated_latency = round(normal.response_latency + 150.0 + 10.0 * math.sin(sample_index * 0.1), 2)
        return replace(normal, response_latency=max(0.0, elevated_latency))

    elif pattern == AbnormalPattern.HIGH_QUEUE:
        elevated_queue = normal.queue_length + 20 + int(abs(3 * math.sin(sample_index * 0.2)))
        return replace(normal, queue_length=max(0, elevated_queue))

    elif pattern == AbnormalPattern.ABNORMAL_IO:
        elevated_io = round(normal.io_activity + 200.0 + 15.0 * math.sin(sample_index * 0.1), 2)
        return replace(normal, io_activity=max(0.0, elevated_io))

    else:
        raise ValueError(f"Unsupported abnormal pattern: {pattern}")
