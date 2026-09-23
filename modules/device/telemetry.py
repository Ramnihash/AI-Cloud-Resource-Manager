"""Normal telemetry generator for simulated OS devices.

This module provides data structures and deterministic generators for producing
baseline, healthy device telemetry without random noise or external dependencies.
"""

from dataclasses import dataclass, asdict
import math
from typing import Any, Dict

from modules.device.device import Device


@dataclass
class DeviceTelemetry:
    """Telemetry data sample captured from a simulated hardware device.

    Attributes:
        device_id: Unique identifier string matching the originating device.
        timestamp: Time at which the telemetry sample was recorded.
        temperature: Operating temperature in degrees Celsius.
        error_rate: Normalized error rate metric (non-negative, low for normal ops).
        response_latency: Average response latency in milliseconds.
        queue_length: Number of pending I/O requests in the device queue.
        io_activity: Metric representing current I/O throughput / workload.
    """

    device_id: str
    timestamp: float
    temperature: float
    error_rate: float
    response_latency: float
    queue_length: int
    io_activity: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert telemetry sample to a dictionary."""
        return asdict(self)



_NORMAL_PROFILES: Dict[str, Dict[str, float]] = {
    "disk": {"base_latency": 6.0, "base_queue": 1.0, "base_io": 30.0},
    "printer": {"base_latency": 15.0, "base_queue": 1.0, "base_io": 5.0},
    "NIC": {"base_latency": 2.0, "base_queue": 2.0, "base_io": 45.0},
    "sensor": {"base_latency": 0.5, "base_queue": 0.0, "base_io": 2.0},
}

_DEFAULT_PROFILE: Dict[str, float] = {
    "base_latency": 5.0,
    "base_queue": 1.0,
    "base_io": 20.0,
}


def generate_normal_telemetry(
    device: Device,
    timestamp: float,
    sample_index: int = 0,
) -> DeviceTelemetry:
    """Generate a deterministic normal telemetry sample for a simulated device.

    Uses smooth trigonometric variation over sample_index to simulate realistic
    operational dynamics while remaining strictly deterministic without random state.

    Args:
        device: The simulated Device object to measure (not mutated).
        timestamp: Timestamp of the measurement.
        sample_index: Sequence or tick index used for smooth variation.

    Returns:
        DeviceTelemetry: A populated normal telemetry snapshot.
    """
    profile = _NORMAL_PROFILES.get(device.device_type, _DEFAULT_PROFILE)


    base_temp = device.temperature if device.temperature > 0.0 else 35.0
    temperature = round(base_temp + 1.5 * math.sin(sample_index * 0.15), 2)


    error_rate = round(max(0.0, 0.001 + 0.0005 * math.sin(sample_index * 0.1)), 5)


    latency_offset = 0.8 * math.sin(sample_index * 0.2)
    response_latency = round(max(0.1, profile["base_latency"] + latency_offset), 2)


    queue_offset = 0.8 * math.cos(sample_index * 0.25)
    queue_length = max(0, int(round(profile["base_queue"] + queue_offset)))


    io_offset = 4.0 * math.cos(sample_index * 0.1)
    io_activity = round(max(0.0, profile["base_io"] + io_offset), 2)

    return DeviceTelemetry(
        device_id=device.device_id,
        timestamp=float(timestamp),
        temperature=temperature,
        error_rate=error_rate,
        response_latency=response_latency,
        queue_length=queue_length,
        io_activity=io_activity,
    )
