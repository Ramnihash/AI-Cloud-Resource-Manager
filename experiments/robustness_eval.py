"""CP7.4 forced model failures and CP7.5 stress tests, through the real Flask app.

Usage (from the repository root):

    python -m experiments.robustness_eval

Everything is driven through ``create_app().test_client()``, so latency measures the
Flask routes plus the model and simulator code, not network or gunicorn overhead.
The CPU models are trained at the same small size the backend uses by default.

Writes to ``results/``:
  robustness_forced_failures.csv  what happens when a model or storage dependency fails
  stress_latency.csv              scale and repeated-step runs: p50/p95/p99, errors, memory growth
  stress_malformed.csv            one row per malformed request: status and whether the body is JSON
"""

from __future__ import annotations

import logging
import os
import tempfile
import time
from unittest.mock import patch

import numpy as np
import pandas as pd
import psutil

from backend.app import create_app
from ml.cpu.burst_predictor import CPUBurstPredictor
from ml.cpu.workload_classifier import CPUWorkloadClassifier
from ml.device import anomaly_service
from modules.cpu.action_loop import CPUActionLoop
from modules.cpu.meta_scheduler import CPUMetaScheduler

RESULTS_DIR = "results"
REPEAT_STEPS = 2000
SCALE_REPEATS = 20


def build_app():
    predictor = CPUBurstPredictor(random_state=42)
    predictor.train_on_synthetic_workload(num_processes=50, seed=42)
    classifier = CPUWorkloadClassifier(random_state=42)
    classifier.train_on_synthetic_dataset(seed=42, target_per_class=10)
    loop = CPUActionLoop(meta_scheduler=CPUMetaScheduler(classifier=classifier, burst_predictor=predictor))
    app = create_app(cpu_action_loop=loop, test_config={"TESTING": True})
    return app, loop, predictor, classifier


def workload(n: int, with_prediction: bool = False) -> list[dict]:
    rows = []
    for i in range(n):
        row = {
            "pid": i,
            "arrival_time": float(i % 50),
            "burst_time": 2.0 + (i * 7) % 15,
            "priority": 1 + i % 4,
            "metadata": {"cpu_intensity": 0.5, "io_wait_ratio": 0.3, "instruction_count": 20000, "process_complexity": 2.0},
        }
        if with_prediction:
            row["predicted_burst_time"] = row["burst_time"]
        rows.append(row)
    return rows


class _LogCapture(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


def _call(client, method: str, path: str, **kwargs):
    t0 = time.perf_counter()
    response = getattr(client, method)(path, **kwargs)
    return response, (time.perf_counter() - t0) * 1000.0


# ---------------------------------------------------------------- 7.4 forced failures

def forced_failures(client, loop, predictor, classifier) -> pd.DataFrame:
    rows = []

    def record(scenario: str, endpoint: str, fn, expect_status: int, signal) -> None:
        capture = _LogCapture()
        root = logging.getLogger()
        root.addHandler(capture)
        try:
            response, ms = fn()
        finally:
            root.removeHandler(capture)
        body = response.get_json(silent=True) or {}
        rows.append({
            "scenario": scenario,
            "endpoint": endpoint,
            "http_status": response.status_code,
            "expected_status": expect_status,
            "graceful": response.status_code == expect_status,
            "signal_in_response": signal(body),
            "warning_logged": len(capture.records) > 0,
            "latency_ms": round(ms, 2),
        })

    cpu_body = {"processes": workload(10)}
    with patch.object(classifier, "predict_workload", side_effect=RuntimeError("forced")):
        record("CPU workload classifier raises", "/api/v1/cpu/action",
               lambda: _call(client, "post", "/api/v1/cpu/action", json=cpu_body), 200,
               lambda b: f"is_fallback={b.get('is_fallback')}, scheduler={b.get('selected_scheduler')}, reason={b.get('fallback_reason')}")
    with patch.object(predictor, "predict_workload", side_effect=RuntimeError("forced")):
        record("CPU burst predictor raises", "/api/v1/cpu/action",
               lambda: _call(client, "post", "/api/v1/cpu/action", json=cpu_body), 200,
               lambda b: f"is_fallback={b.get('is_fallback')}, scheduler={b.get('selected_scheduler')}, reason={b.get('fallback_reason')}")

    mem_body = {"reference_string": [1, 2, 3, 4, 1, 2, 5, 1, 2, 3, 4, 5], "num_frames": 3, "samples": 8, "interval_s": 0}
    with patch("backend.routes.MarkovPagePredictor.train", side_effect=RuntimeError("forced")):
        record("Markov page predictor raises", "/api/v1/memory/analysis",
               lambda: _call(client, "post", "/api/v1/memory/analysis", json=mem_body), 200,
               lambda b: "page_prediction[0]=" + str((b.get("page_prediction") or [{}])[0]))

    dev_body = {"device_id": "disk_0", "pattern": "normal"}
    missing = os.path.join(tempfile.gettempdir(), "definitely_missing_model.pt")
    original_load = anomaly_service.load_bundle
    with patch.object(anomaly_service, "load_bundle", side_effect=lambda: original_load(missing)):
        record("Anomaly model file removed", "/api/v1/device/anomaly-score",
               lambda: _call(client, "post", "/api/v1/device/anomaly-score", json=dev_body), 503,
               lambda b: str(b.get("error")))
    with tempfile.TemporaryDirectory() as folder:
        corrupt = os.path.join(folder, "corrupt.pt")
        with open(corrupt, "wb") as handle:
            handle.write(b"not a model")
        with patch.object(anomaly_service, "load_bundle", side_effect=lambda: original_load(corrupt)):
            record("Anomaly model file corrupt", "/api/v1/device/anomaly-score",
                   lambda: _call(client, "post", "/api/v1/device/anomaly-score", json=dev_body), 503,
                   lambda b: str(b.get("error")))

    action_body = {"device_id": "disk_0", "failure_probability": 0.9, "device_status": "unhealthy"}
    with patch.object(anomaly_service, "load_bundle", side_effect=lambda: original_load(missing)):
        record("Anomaly model removed, predictive action still used", "/api/v1/device/predictive-action",
               lambda: _call(client, "post", "/api/v1/device/predictive-action", json=action_body), 200,
               lambda b: f"backup={b.get('backup_result', {}).get('success')}, rerouted={b.get('recovery_result', {}).get('rerouted')}")

    def s3_down():
        class Broken:
            def put_object(self, *a, **k):
                raise ConnectionError("S3 unreachable (forced)")

        with patch.dict(os.environ, {"BACKUP_S3_BUCKET": "forced-test-bucket"}):
            with patch("modules.device.cloud_backup._get_client", return_value=Broken()):
                return _call(client, "post", "/api/v1/device/predictive-action", json=action_body)

    record("Cloud storage unreachable (S3 backup)", "/api/v1/device/predictive-action", s3_down, 200,
           lambda b: f"backup success={b.get('backup_result', {}).get('success')}, message={b.get('backup_result', {}).get('message')}")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 7.5 stress

def _stats(label: str, latencies: list[float], statuses: list[int], rss_before: float, rss_after: float, extra=None, rss_mid=None) -> dict:
    arr = np.array(latencies)
    first, last = arr[: max(1, len(arr) // 10)], arr[-max(1, len(arr) // 10):]
    errors = sum(1 for s in statuses if s != 200)
    row = {
        "scenario": label,
        "requests": len(arr),
        "non_200": errors,
        "status_5xx": sum(1 for s in statuses if s >= 500),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "max_ms": float(arr.max()),
        "first_10pct_mean_ms": float(first.mean()),
        "last_10pct_mean_ms": float(last.mean()),
        "rss_growth_mb": (rss_after - rss_before) / 1e6,
        "rss_growth_second_half_mb": ((rss_after - rss_mid) / 1e6) if rss_mid is not None else None,
    }
    row.update(extra or {})
    return row


def _run_repeated(client, path: str, body: dict, n: int):
    proc = psutil.Process()
    rss_before = proc.memory_info().rss
    latencies, statuses = [], []
    rss_mid = rss_before
    for i in range(n):
        response, ms = _call(client, "post", path, json=body)
        latencies.append(ms)
        statuses.append(response.status_code)
        if i == n // 2:
            rss_mid = proc.memory_info().rss
    return latencies, statuses, rss_before, proc.memory_info().rss, rss_mid


def stress(client, loop, app, repeat_steps: int = REPEAT_STEPS, scale_repeats: int = SCALE_REPEATS) -> pd.DataFrame:
    rows = []
    for n in (20, 200, 1000):
        loop.reset()
        lat, st, a, b, mid = _run_repeated(client, "/api/v1/cpu/action", {"processes": workload(n)}, scale_repeats)
        rows.append(_stats(f"cpu/action, {n} processes x{scale_repeats}", lat, st, a, b, {"processes": n}, mid))

    loop.reset()
    lat, st, a, b, mid = _run_repeated(client, "/api/v1/cpu/action", {"processes": workload(20)}, repeat_steps)
    rows.append(_stats(f"cpu/action, {repeat_steps} repeated steps (20 processes)", lat, st, a, b,
                       {"processes": 20, "history_entries": len(loop.get_history())}, mid))

    lat, st, a, b, mid = _run_repeated(client, "/api/v1/system/step", {"workload_profile": "balanced"}, repeat_steps)
    manager = app.config.get("INTEGRATED_EXECUTION_MANAGER")
    rows.append(_stats(f"system/step, {repeat_steps} repeated steps", lat, st, a, b,
                       {"history_entries": len(manager.get_history()) if manager else None}, mid))
    return pd.DataFrame(rows)


def malformed_requests(client) -> pd.DataFrame:
    endpoints = [
        ("post", "/api/v1/cpu/action"),
        ("post", "/api/v1/memory/page-replacement"),
        ("post", "/api/v1/memory/monitor"),
        ("post", "/api/v1/memory/analysis"),
        ("post", "/api/v1/device/predictive-action"),
        ("post", "/api/v1/device/anomaly-score"),
        ("post", "/api/v1/system/step"),
    ]
    json_payloads = {
        "empty object": {},
        "list instead of object": [1, 2, 3],
        "null": None,
        "number": 42,
        "string": "hello",
        "missing fields": {"unrelated": 1},
        "wrong types": {"processes": "x", "reference_string": "x", "device_id": 5, "failure_probability": "high"},
        "nulls": {"processes": None, "reference_string": None, "device_id": None, "failure_probability": None, "device_status": None},
        "negative and huge numbers": {"processes": [{"pid": 1, "arrival_time": -5, "burst_time": 1e308}], "num_frames": -1, "failure_probability": 1e308},
        "nested garbage": {"processes": [[[[]]]], "reference_string": [[1], {"a": 2}], "device_id": {"x": 1}},
        "very long string": {"device_id": "x" * 100_000, "device_status": "y" * 100_000},
        "unicode": {"device_id": "‮\u0000😀", "device_status": "\u0000"},
        "empty lists": {"processes": [], "reference_string": []},
        "huge burst_time": {"processes": [{"pid": 1, "arrival_time": 0, "burst_time": 1e7}]},
        "too many processes": {"processes": [{"pid": i, "arrival_time": 0, "burst_time": 1} for i in range(1500)]},
        "NaN and infinity": {"failure_probability": float("nan"), "num_frames": float("inf"), "processes": [{"pid": 1, "arrival_time": float("nan"), "burst_time": float("inf")}]},
    }
    rows = []

    def add(method, path, label, response, ms):
        body = response.get_json(silent=True)
        rows.append({
            "endpoint": path, "case": label, "status": response.status_code,
            "json_error_body": isinstance(body, dict) and "error" in body,
            "error_body_ok": response.status_code < 400 or (isinstance(body, dict) and "error" in body),
            "is_5xx": response.status_code >= 500,
            "latency_ms": round(ms, 2),
        })

    for method, path in endpoints:
        for label, payload in json_payloads.items():
            response, ms = _call(client, method, path, json=payload)
            add(method, path, label, response, ms)
        response, ms = _call(client, method, path, data="not json at all", content_type="application/json")
        add(method, path, "malformed JSON text", response, ms)
        response, ms = _call(client, method, path, data='{"a": 1}', content_type="text/plain")
        add(method, path, "wrong content type", response, ms)
        response, ms = _call(client, method, path, data=b"\xff\xfe\x00\x01", content_type="application/json")
        add(method, path, "binary body", response, ms)
        response, ms = _call(client, "get", path)
        add(method, path, "wrong method (GET)", response, ms)
    for label, path in {"unknown route": "/api/v1/nothing", "path traversal": "/api/v1/../../etc/passwd"}.items():
        response, ms = _call(client, "get", path)
        add("get", path, label, response, ms)
    response, ms = _call(client, "post", "/api/v1/memory/monitor", json={"pid": 99999999})
    add("post", "/api/v1/memory/monitor", "process id does not exist", response, ms)
    response, ms = _call(client, "post", "/api/v1/device/predictive-action",
                         json={"device_id": "disk_0", "failure_probability": 0.9, "device_status": "on fire"})
    add("post", "/api/v1/device/predictive-action", "unknown device status", response, ms)
    response, ms = _call(client, "post", "/api/v1/device/predictive-action",
                         json={"device_id": "ghost_9", "failure_probability": 0.9, "device_status": "unhealthy"})
    add("post", "/api/v1/device/predictive-action", "unknown device id", response, ms)
    return pd.DataFrame(rows)


def run(out: str = RESULTS_DIR, repeat_steps: int = REPEAT_STEPS, scale_repeats: int = SCALE_REPEATS) -> dict[str, pd.DataFrame]:
    os.makedirs(out, exist_ok=True)
    app, loop, predictor, classifier = build_app()
    client = app.test_client()
    failures = forced_failures(client, loop, predictor, classifier)
    malformed = malformed_requests(client)
    stress_df = stress(client, loop, app, repeat_steps, scale_repeats)
    failures.to_csv(os.path.join(out, "robustness_forced_failures.csv"), index=False)
    malformed.to_csv(os.path.join(out, "stress_malformed.csv"), index=False)
    stress_df.to_csv(os.path.join(out, "stress_latency.csv"), index=False)
    return {"failures": failures, "malformed": malformed, "stress": stress_df}


def main() -> None:
    r = run()
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    pd.set_option("display.max_colwidth", 110)
    print("\n7.4 Forced failures:")
    print(r["failures"].to_string(index=False))
    m = r["malformed"]
    print(f"\n7.5 Malformed requests: {len(m)} sent, {int(m.is_5xx.sum())} returned 5xx, "
          f"{int((~m.error_body_ok).sum())} errors without a JSON error body")
    print(m.groupby("status").size().to_string())
    bad = m[m.is_5xx | ~m.error_body_ok]
    if len(bad):
        print("\nRequests that returned 5xx or an error without a JSON error body:")
        print(bad.to_string(index=False))
    accepted = m[m.status < 400]
    if len(accepted):
        print("\nMalformed requests accepted with 2xx (check these are intended):")
        print(accepted[["endpoint", "case", "status"]].to_string(index=False))
    print("\n7.5 Stress:")
    print(r["stress"].round(2).to_string(index=False))
    print(f"\nCSVs written to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
