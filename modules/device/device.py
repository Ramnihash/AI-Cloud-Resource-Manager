"""Device state representation for the Device Simulator module.

This module defines the core data structure and state tracking for
simulated physical and virtual devices within the OS.
"""

from dataclasses import dataclass, asdict
from enum import Enum
from typing import Any, Dict, Union


class DeviceType(str, Enum):
    """Enumeration of allowed device types in the baseline fleet."""
    DISK = "disk"
    PRINTER = "printer"
    NIC = "NIC"
    SENSOR = "sensor"


ALLOWED_DEVICE_TYPES = {dt.value for dt in DeviceType}


@dataclass
class Device:
    """Represents a simulated hardware device and its current operating state.

    Attributes:
        device_id: Unique identifier for the device (e.g., 'disk_0').
        device_type: Device category constrained to ('disk', 'printer', 'NIC', 'sensor').
        status: Current operational status (default: 'ok').
        temperature: Operating temperature in degrees Celsius (default: 35.0).
        error_rate: Observed error rate metric (default: 0.0).
        response_latency: Response latency in milliseconds (default: 0.0).
        queue_length: Number of pending I/O requests in queue (default: 0).
    """

    device_id: str
    device_type: Union[DeviceType, str]
    status: str = "ok"
    temperature: float = 35.0
    error_rate: float = 0.0
    response_latency: float = 0.0
    queue_length: int = 0

    def __post_init__(self) -> None:
        """Validate and normalize device fields upon initialization."""
        if isinstance(self.device_type, DeviceType):
            self.device_type = self.device_type.value

        if self.device_type not in ALLOWED_DEVICE_TYPES:
            allowed_list = ", ".join(sorted(ALLOWED_DEVICE_TYPES))
            raise ValueError(
                f"Invalid device_type '{self.device_type}'. Allowed types: {allowed_list}"
            )

    def to_dict(self) -> Dict[str, Any]:
        """Return the device state as a dictionary."""
        return asdict(self)
