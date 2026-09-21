# Device Telemetry Dataset

## Purpose

This dataset contains simulated device telemetry for the AI-Driven Cloud Resource Manager.

The telemetry is used by the Device Management module for anomaly detection and predictive maintenance.

## Device Types

The simulated device fleet contains:

- `disk`
- `printer`
- `NIC`
- `sensor`

## Telemetry Fields

Each telemetry record contains the following fields:

| Field | Description |
|---|---|
| `device_id` | Unique identifier of the device |
| `timestamp` | Timestamp of the telemetry observation |
| `temperature` | Device temperature |
| `error_rate` | Device error rate |
| `response_latency` | Device response latency |
| `queue_length` | Number of queued operations |
| `io_activity` | Device I/O activity level |

## Dataset Categories

The telemetry dataset supports two main categories:

1. **Normal telemetry**
2. **Abnormal/degrading telemetry**

### Normal Telemetry

Normal telemetry represents expected operating behaviour for each simulated device type.

### Abnormal / Degrading Telemetry

Abnormal telemetry represents conditions that may indicate device degradation or failure.

Supported abnormal patterns include:

- High temperature
- High error rate
- High response latency
- High queue length
- Abnormal I/O activity

## Data Generation Requirements

The dataset generation process must be:

- deterministic when a fixed configuration is used
- reproducible
- compatible with the existing device simulator
- compatible with the existing normal telemetry generator
- compatible with the existing abnormal telemetry generator
- exportable for later machine-learning use

The generated telemetry must preserve the existing telemetry schema.

## Intended Machine-Learning Use

The resulting telemetry dataset will be used by the Device Management module for training and evaluating the device anomaly-detection component.

The dataset-generation layer is responsible only for producing and preparing telemetry data.

It does **not** perform machine-learning prediction or make device-management decisions.

## Future Processing

The telemetry dataset will later support:

- sequence preparation for the Transformer-based anomaly detector
- anomaly detection evaluation
- device-health monitoring
- detection of degrading device behaviour

The prediction, decision, and recovery actions are implemented in later development checkpoints.