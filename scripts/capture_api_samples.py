"""Capture one real request/response pair per API endpoint into docs/api_samples/.

Run from the project root:  python -m scripts.capture_api_samples
The Flask test client is used, so no server needs to be running. The device
anomaly-score sample needs the trained model (python -m scripts.train_device_anomaly_detector)
and PyTorch; without them it records the 503 the API really returns.
"""
from __future__ import annotations

import json
from pathlib import Path

from backend.app import create_app

OUT = Path(__file__).resolve().parent.parent / "docs" / "api_samples"

CPU_PROCESSES = [
    {"pid": i, "arrival_time": float(i), "burst_time": 3.0 + 2 * i, "priority": 1 + i % 3,
     "metadata": {"cpu_intensity": 0.7, "io_wait_ratio": 0.2, "instruction_count": 30000, "process_complexity": 2.0}}
    for i in range(5)
]

REQUESTS = [
    ("health", "GET", "/health", None),
    ("cpu_action", "POST", "/api/v1/cpu/action", {"processes": CPU_PROCESSES}),
    ("memory_page_replacement", "POST", "/api/v1/memory/page-replacement",
     {"algorithm": "lru", "num_frames": 3, "reference_string": [1, 2, 3, 2, 1, 4, 2, 5, 1, 2]}),
    ("memory_monitor", "POST", "/api/v1/memory/monitor", {}),
    ("memory_analysis", "POST", "/api/v1/memory/analysis",
     {"reference_string": [1, 2, 3, 1, 2, 3, 1, 2, 3, 4, 1, 2], "num_frames": 3, "samples": 8, "interval_s": 0.0}),
    ("device_anomaly_score", "POST", "/api/v1/device/anomaly-score", {"device_id": "disk_0", "pattern": "high_temperature"}),
    ("device_predictive_action", "POST", "/api/v1/device/predictive-action",
     {"device_id": "disk_0", "failure_probability": 0.92, "device_status": "unhealthy", "backup_device_id": "disk_1"}),
    ("system_step", "POST", "/api/v1/system/step", {"workload_profile": "balanced", "num_processes": 5}),
    ("system_history", "GET", "/api/v1/system/history?limit=1", None),
    ("monitor_overview", "GET", "/api/v1/monitor/overview", None),
    ("system_reset", "POST", "/api/v1/system/reset", {}),
    ("error_malformed", "POST", "/api/v1/cpu/action", ["not", "an", "object"]),
]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    client = create_app().test_client()
    for name, method, url, body in REQUESTS:
        response = client.open(url, method=method, json=body) if body is not None else client.open(url, method=method)
        sample = {
            "request": {"method": method, "url": url, "body": body},
            "response": {"status": response.status_code, "body": response.get_json(silent=True)},
        }
        (OUT / f"{name}.json").write_text(json.dumps(sample, indent=2, default=str), encoding="utf-8")
        print(f"{response.status_code}  {method:4} {url}")


if __name__ == "__main__":
    main()
