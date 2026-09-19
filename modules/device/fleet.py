"""Simulated device fleet management for the Device Simulator module.

This module provides factory functions to construct and manage the
deterministic baseline fleet of simulated devices for the OS.
"""

from modules.device.device import Device, DeviceType


def create_default_fleet() -> list[Device]:
    """Create and return the deterministic baseline simulated device fleet.

    The baseline fleet contains exactly four devices:
    1. disk_0 (disk)
    2. printer_0 (printer)
    3. NIC_0 (NIC)
    4. sensor_0 (sensor)

    All devices are initialized with standard baseline state defaults.

    Returns:
        list[Device]: The ordered list of baseline simulated devices.
    """
    return [
        Device(device_id="disk_0", device_type=DeviceType.DISK),
        Device(device_id="printer_0", device_type=DeviceType.PRINTER),
        Device(device_id="NIC_0", device_type=DeviceType.NIC),
        Device(device_id="sensor_0", device_type=DeviceType.SENSOR),
    ]
