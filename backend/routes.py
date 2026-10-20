"""API route handlers for CP5.1 Common Backend Interface.

Exposes REST endpoints for:
    - GET  /health
    - POST /api/v1/cpu/action
    - POST /api/v1/memory/page-replacement
    - POST /api/v1/memory/monitor
    - POST /api/v1/device/predictive-action
"""

from __future__ import annotations

import dataclasses
import logging
import math
import os
import random
from typing import Any, Optional

import psutil
from flask import Blueprint, current_app, jsonify, request

from modules.cpu.action_loop import CPUActionLoop
from modules.cpu.process import Process
from modules.device.predictive_action import handle_predictive_action
from ml.device import anomaly_service
from ml.memory.markov_predictor import MarkovPagePredictor
from ml.memory.memory_leak_detector import MemoryLeakDetector
from ml.memory.memory_pressure_forecaster import MemoryPressureForecaster
from modules.memory.memory_health_action import MemoryHealthActionLayer
from modules.memory.memory_monitor import MemoryMonitor
from modules.memory.page_prediction_action import PagePredictionActionLayer
from modules.memory.page_replacement import (
    FIFOPageReplacement,
    LRUPageReplacement,
    OptimalPageReplacement,
)

api_bp = Blueprint("api", __name__)
logger = logging.getLogger(__name__)


MAX_PROCESSES = 1000
MAX_TIME_VALUE = 10_000.0
MAX_TOTAL_BURST = 50_000.0


def _check_workload_limits(processes: list[Process]) -> str | None:
    """Reject non-finite or oversized workloads that would tie up the single worker.

    Round Robin emits one Gantt slice per quantum, so one huge burst_time can mean
    millions of slices, gigabytes of response and a stuck worker.
    """
    if len(processes) > MAX_PROCESSES:
        return f"Workload must contain at most {MAX_PROCESSES} processes"
    total = 0.0
    for i, p in enumerate(processes):
        values = [("arrival_time", p.arrival_time), ("burst_time", p.burst_time)]
        if p.predicted_burst_time is not None:
            values.append(("predicted_burst_time", p.predicted_burst_time))
        for name, value in values:
            if not math.isfinite(value):
                return f"Process at index {i} has non-finite {name}"
            if value > MAX_TIME_VALUE:
                return f"Process at index {i} has {name} above the limit of {MAX_TIME_VALUE:g}"
        total += p.burst_time
    if total > MAX_TOTAL_BURST:
        return f"Total burst_time must not exceed {MAX_TOTAL_BURST:g}"
    return None


def _extract_json_payload(allow_empty: bool = False) -> tuple[dict[str, Any] | None, str | None]:
    """Validate and extract JSON payload from the request.

    Returns:
        tuple (data, error_message). If invalid, data is None and error_message is string.
    """
    if not request.is_json:
        return None, "Request must be valid JSON with Content-Type: application/json"

    try:
        data = request.get_json(silent=False)
    except Exception:
        return None, "Malformed JSON request"

    if data is None:
        if allow_empty:
            return {}, None
        return None, "Missing JSON payload"

    if not isinstance(data, dict):
        return None, "JSON payload must be a JSON object"

    return data, None


def _get_cpu_action_loop() -> CPUActionLoop:
    """Retrieve injected CPUActionLoop or initialize a default instance."""
    loop = current_app.config.get("CPU_ACTION_LOOP")
    if loop is None:
        from ml.cpu.burst_predictor import CPUBurstPredictor
        from ml.cpu.workload_classifier import CPUWorkloadClassifier
        from modules.cpu.action_loop import CPUActionLoop
        from modules.cpu.meta_scheduler import CPUMetaScheduler

        predictor = CPUBurstPredictor(random_state=42)
        predictor.train_on_synthetic_workload(num_processes=50, seed=42)
        classifier = CPUWorkloadClassifier(random_state=42)
        classifier.train_on_synthetic_dataset(seed=42, target_per_class=10)
        meta = CPUMetaScheduler(classifier=classifier, burst_predictor=predictor)
        loop = CPUActionLoop(meta_scheduler=meta)
        current_app.config["CPU_ACTION_LOOP"] = loop
    return loop


@api_bp.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({"status": "ok"}), 200


@api_bp.route("/api/v1/cpu/action", methods=["POST"])
def cpu_action():
    """Execute a simulated CPU AI prediction-to-action cycle via CPUActionLoop."""
    data, err = _extract_json_payload(allow_empty=False)
    if err:
        return jsonify({"error": err}), 400

    if "processes" not in data:
        return jsonify({"error": "Missing required field: processes"}), 400

    raw_processes = data["processes"]
    if not isinstance(raw_processes, list):
        return jsonify({"error": "Field 'processes' must be a list"}), 400

    if len(raw_processes) == 0:
        return jsonify({"error": "Workload must contain at least one process"}), 400

    compare_with_baseline = data.get("compare_with_baseline", False)
    if not isinstance(compare_with_baseline, bool):
        return jsonify({"error": "Field 'compare_with_baseline' must be a boolean"}), 400

    process_objects: list[Process] = []
    for i, p_dict in enumerate(raw_processes):
        if not isinstance(p_dict, dict):
            return jsonify({"error": f"Process at index {i} must be an object"}), 400

        for req_field in ("pid", "arrival_time", "burst_time"):
            if req_field not in p_dict:
                return jsonify({"error": f"Process at index {i} missing required field '{req_field}'"}), 400

        pid = p_dict["pid"]
        if not isinstance(pid, (int, str)):
            return jsonify({"error": f"Process at index {i} has invalid pid: must be int or str"}), 400

        arr = p_dict["arrival_time"]
        if isinstance(arr, bool) or not isinstance(arr, (int, float)):
            return jsonify({"error": f"Process at index {i} has invalid arrival_time: must be number"}), 400

        burst = p_dict["burst_time"]
        if isinstance(burst, bool) or not isinstance(burst, (int, float)):
            return jsonify({"error": f"Process at index {i} has invalid burst_time: must be number"}), 400

        priority = p_dict.get("priority", 0)
        if isinstance(priority, bool) or not isinstance(priority, int):
            return jsonify({"error": f"Process at index {i} has invalid priority: must be integer"}), 400

        pred_burst = p_dict.get("predicted_burst_time")
        if pred_burst is not None and (isinstance(pred_burst, bool) or not isinstance(pred_burst, (int, float))):
            return jsonify({"error": f"Process at index {i} has invalid predicted_burst_time: must be number or null"}), 400

        meta = p_dict.get("metadata", {})
        if not isinstance(meta, dict):
            return jsonify({"error": f"Process at index {i} has invalid metadata: must be an object"}), 400

        try:
            p = Process(
                pid=pid,
                arrival_time=float(arr),
                burst_time=float(burst),
                priority=int(priority),
                predicted_burst_time=float(pred_burst) if pred_burst is not None else None,
                workload_type=p_dict.get("workload_type"),
                metadata=dict(meta),
            )
            process_objects.append(p)
        except (ValueError, TypeError) as e:
            return jsonify({"error": f"Invalid process at index {i}: {e}"}), 400

    limit_error = _check_workload_limits(process_objects)
    if limit_error:
        return jsonify({"error": limit_error}), 400

    try:
        loop = _get_cpu_action_loop()
        step_result = loop.step(processes=process_objects, compare_with_baseline=compare_with_baseline)
    except (ValueError, TypeError) as e:
        return jsonify({"error": str(e)}), 400
    except Exception:
        return jsonify({"error": "Internal server error"}), 500

    return jsonify(step_result.to_dict()), 200


@api_bp.route("/api/v1/memory/page-replacement", methods=["POST"])
def memory_page_replacement():
    """Simulate page replacement using FIFO, LRU, or Optimal algorithms."""
    data, err = _extract_json_payload(allow_empty=False)
    if err:
        return jsonify({"error": err}), 400

    if "algorithm" not in data:
        return jsonify({"error": "Missing required field: algorithm"}), 400
    if not isinstance(data["algorithm"], str):
        return jsonify({"error": "Field 'algorithm' must be a string"}), 400

    algo_name = data["algorithm"].strip().lower()
    algo_map = {
        "fifo": FIFOPageReplacement,
        "lru": LRUPageReplacement,
        "optimal": OptimalPageReplacement,
    }
    if algo_name not in algo_map:
        return jsonify({"error": f"Invalid memory algorithm '{data['algorithm']}'. Allowed: fifo, lru, optimal"}), 400

    if "num_frames" not in data:
        return jsonify({"error": "Missing required field: num_frames"}), 400
    num_frames = data["num_frames"]
    if isinstance(num_frames, bool) or not isinstance(num_frames, int) or num_frames < 1:
        return jsonify({"error": "Field 'num_frames' must be an integer >= 1"}), 400

    if "reference_string" not in data:
        return jsonify({"error": "Missing required field: reference_string"}), 400
    ref_string = data["reference_string"]
    if not isinstance(ref_string, list):
        return jsonify({"error": "Field 'reference_string' must be a list of integers"}), 400

    for idx, page in enumerate(ref_string):
        if isinstance(page, bool) or not isinstance(page, int):
            return jsonify({"error": f"Element at index {idx} in 'reference_string' must be an integer"}), 400

    algo_cls = algo_map[algo_name]
    try:
        algo_instance = algo_cls(num_frames=num_frames)
        result = algo_instance.run(reference_string=ref_string)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception:
        return jsonify({"error": "Internal server error"}), 500

    return jsonify(result), 200


@api_bp.route("/api/v1/memory/monitor", methods=["POST"])
def memory_monitor():
    """Retrieve memory snapshot or collect time-series memory samples."""
    data, err = _extract_json_payload(allow_empty=True)
    if err:
        return jsonify({"error": err}), 400

    is_collection = "count" in data or "interval_s" in data

    pid = data.get("pid")
    if pid is not None and (isinstance(pid, bool) or not isinstance(pid, int)):
        return jsonify({"error": "Field 'pid' must be an integer or null"}), 400

    monitor = MemoryMonitor()

    if is_collection:
        if "count" not in data:
            return jsonify({"error": "Collection request missing required field: count"}), 400
        count = data["count"]
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            return jsonify({"error": "Field 'count' must be an integer >= 1"}), 400

        if "interval_s" not in data:
            return jsonify({"error": "Collection request missing required field: interval_s"}), 400
        interval_s = data["interval_s"]
        if isinstance(interval_s, bool) or not isinstance(interval_s, (int, float)) or interval_s < 0:
            return jsonify({"error": "Field 'interval_s' must be a number >= 0"}), 400

        try:
            result = monitor.collect(n=count, interval_s=float(interval_s), pid=pid)
        except psutil.NoSuchProcess as e:
            return jsonify({"error": f"Process not found: {e}"}), 400
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        except Exception:
            return jsonify({"error": "Internal server error"}), 500
    else:
        try:
            result = monitor.take_snapshot(pid=pid)
        except psutil.NoSuchProcess as e:
            return jsonify({"error": f"Process not found: {e}"}), 400
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        except Exception:
            return jsonify({"error": "Internal server error"}), 500

    return jsonify(result), 200


@api_bp.route("/api/v1/memory/analysis", methods=["POST"])
def memory_analysis():
    """Run the memory intelligence layer on a reference string and live RAM samples.

    Page predictors are trained on the supplied reference string. The pressure
    forecaster and leak detector use samples of this backend process and the
    host RAM collected by MemoryMonitor during the request.
    """
    data, err = _extract_json_payload(allow_empty=False)
    if err:
        return jsonify({"error": err}), 400

    ref = data.get("reference_string")
    if not isinstance(ref, list) or len(ref) < 3:
        return jsonify({"error": "Field 'reference_string' must be a list of at least 3 integers"}), 400
    for idx, page in enumerate(ref):
        if isinstance(page, bool) or not isinstance(page, int) or page < 0:
            return jsonify({"error": f"Element at index {idx} in 'reference_string' must be a non-negative integer"}), 400

    num_frames = data.get("num_frames", 3)
    if isinstance(num_frames, bool) or not isinstance(num_frames, int) or not 1 <= num_frames <= 20:
        return jsonify({"error": "Field 'num_frames' must be an integer between 1 and 20"}), 400

    samples = data.get("samples", 12)
    if isinstance(samples, bool) or not isinstance(samples, int) or not 8 <= samples <= 30:
        return jsonify({"error": "Field 'samples' must be an integer between 8 and 30"}), 400

    interval_s = data.get("interval_s", 0.2)
    if isinstance(interval_s, bool) or not isinstance(interval_s, (int, float)) or not 0 <= interval_s <= 1:
        return jsonify({"error": "Field 'interval_s' must be a number between 0 and 1"}), 400

    try:
        frames_result = LRUPageReplacement(num_frames=num_frames).run(reference_string=ref)
        snapshots_in = frames_result.get("frame_snapshots") or frames_result.get("snapshots") or []
        memory_state = [p for p in (snapshots_in[-1] if snapshots_in else []) if p is not None]

        current_page = ref[-1]
        prediction_layer = PagePredictionActionLayer()
        pages = []

        try:
            markov = MarkovPagePredictor()
            markov.train(ref)
            candidates = markov.predict(current_page, top_k=3)
            pages.append({
                "predictor": "Markov",
                "available": True,
                "candidates": candidates,
                "action": prediction_layer.evaluate_and_act(candidates, memory_state, "Markov"),
            })
        except Exception as exc:
            logger.warning("Markov page predictor failed: %s: %s", type(exc).__name__, exc)
            pages.append({"predictor": "Markov", "available": False,
                          "reason": f"Predictor failed: {type(exc).__name__}"})

        try:
            from ml.memory.transformer_predictor import TransformerPagePredictor
        except ImportError:
            pages.append({"predictor": "Transformer", "available": False,
                          "reason": "PyTorch is not installed on the backend"})
        else:
            context_len = 2
            if len(ref) <= context_len:
                pages.append({"predictor": "Transformer", "available": False,
                              "reason": "Reference string is too short for the context window"})
            else:
                try:
                    transformer = TransformerPagePredictor(
                        num_pages=max(ref) + 1, context_len=context_len, d_model=16, nhead=2)
                    transformer.train(ref, epochs=20)
                    t_cands = transformer.predict(ref[-context_len:], top_k=3)
                    pages.append({
                        "predictor": "Transformer",
                        "available": True,
                        "candidates": t_cands,
                        "action": prediction_layer.evaluate_and_act(t_cands, memory_state, "Transformer"),
                    })
                except Exception as exc:
                    logger.warning("Transformer page predictor failed: %s: %s", type(exc).__name__, exc)
                    pages.append({"predictor": "Transformer", "available": False,
                                  "reason": f"Predictor failed: {type(exc).__name__}"})

        pid = os.getpid()
        snaps = MemoryMonitor().collect(n=samples, interval_s=float(interval_s), pid=pid)["snapshots"]

        window = min(5, len(snaps) - 1)
        forecaster = MemoryPressureForecaster(window_size=window)
        forecaster.train(snaps)
        pressure = forecaster.predict(snaps[-window:])

        leak_window = min(8, len(snaps))
        leak = MemoryLeakDetector(window_size=leak_window).detect(snaps)

        health = MemoryHealthActionLayer().evaluate_and_act(leak, pressure)
    except (ValueError, TypeError) as e:
        return jsonify({"error": str(e)}), 400
    except Exception:
        return jsonify({"error": "Internal server error"}), 500

    return jsonify({
        "reference_length": len(ref),
        "num_frames": num_frames,
        "frames_policy": "LRU",
        "memory_state": memory_state,
        "current_page": current_page,
        "page_prediction": pages,
        "samples": {
            "count": len(snaps),
            "interval_s": float(interval_s),
            "pid": pid,
            "ram_percent": [s["ram_percent"] for s in snaps],
            "rss_bytes": [s["proc_rss_bytes"] for s in snaps],
        },
        "pressure_forecast": {**pressure, "window_size": window},
        "leak_detection": {**leak, "window_size": leak_window},
        "health_action": health,
    }), 200


@api_bp.route("/api/v1/device/predictive-action", methods=["POST"])
def device_predictive_action():
    """Execute simulated device failure-risk action pipeline."""
    data, err = _extract_json_payload(allow_empty=False)
    if err:
        return jsonify({"error": err}), 400

    for req_field in ("device_id", "failure_probability", "device_status"):
        if req_field not in data:
            return jsonify({"error": f"Missing required field: {req_field}"}), 400

    device_id = data["device_id"]
    if not isinstance(device_id, str) or not device_id.strip():
        return jsonify({"error": "Field 'device_id' must be a non-empty string"}), 400

    prob = data["failure_probability"]
    if isinstance(prob, bool) or not isinstance(prob, (int, float)):
        return jsonify({"error": "Field 'failure_probability' must be a number between 0.0 and 1.0"}), 400
    if not (0.0 <= float(prob) <= 1.0):
        return jsonify({"error": "Field 'failure_probability' must be between 0.0 and 1.0 inclusive"}), 400

    status = data["device_status"]
    if not isinstance(status, str) or not status.strip():
        return jsonify({"error": "Field 'device_status' must be a non-empty string"}), 400

    backup_device_id = data.get("backup_device_id")
    if backup_device_id is not None and (not isinstance(backup_device_id, str) or not backup_device_id.strip()):
        return jsonify({"error": "Field 'backup_device_id' must be a non-empty string or null"}), 400

    threshold = data.get("threshold")
    if threshold is not None:
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
            return jsonify({"error": "Field 'threshold' must be a number between 0.0 and 1.0"}), 400
        if not (0.0 <= float(threshold) <= 1.0):
            return jsonify({"error": "Field 'threshold' must be between 0.0 and 1.0 inclusive"}), 400

    kwargs: dict[str, Any] = {
        "device_id": device_id.strip(),
        "failure_probability": float(prob),
        "device_status": status.strip(),
        "backup_device_id": backup_device_id.strip() if backup_device_id else None,
    }
    if threshold is not None:
        kwargs["threshold"] = float(threshold)

    try:
        result = handle_predictive_action(**kwargs)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception:
        return jsonify({"error": "Internal server error"}), 500

    _get_shared_monitor().record_device_action(result)

    return jsonify(dataclasses.asdict(result)), 200


@api_bp.route("/api/v1/device/anomaly-score", methods=["POST"])
def device_anomaly_score():
    """Score a simulated telemetry window with the trained anomaly detector.

    Returns the reconstruction-error score, the detector threshold, the
    anomaly decision and a failure-risk value for the predictive-action form.
    """
    data, err = _extract_json_payload(allow_empty=False)
    if err:
        return jsonify({"error": err}), 400

    device_id = data.get("device_id")
    if not isinstance(device_id, str) or not device_id.strip():
        return jsonify({"error": "Field 'device_id' must be a non-empty string"}), 400

    pattern = data.get("pattern")
    if pattern in ("", "normal"):
        pattern = None
    if pattern is not None and pattern not in anomaly_service.FAULT_PATTERNS:
        allowed = ", ".join(anomaly_service.FAULT_PATTERNS)
        return jsonify({"error": f"Invalid pattern. Allowed: normal, {allowed}"}), 400

    from modules.device.fleet import create_default_fleet

    device = next((d for d in create_default_fleet() if d.device_id == device_id.strip()), None)
    if device is None:
        return jsonify({"error": f"Unknown device '{device_id}'"}), 400

    try:
        bundle = anomaly_service.load_bundle()
    except anomaly_service.ModelUnavailableError as e:
        logger.warning("Anomaly detector unavailable: %s", e)
        return jsonify({"error": str(e)}), 503
    except Exception:
        return jsonify({"error": "Internal server error"}), 500

    try:
        window, samples = anomaly_service.build_window(
            device, pattern, bundle["window_size"], start_index=random.randint(0, 1000)
        )
        score = anomaly_service.score_window(bundle, window)
    except (ValueError, TypeError) as e:
        return jsonify({"error": str(e)}), 400
    except Exception:
        return jsonify({"error": "Internal server error"}), 500

    threshold = bundle["threshold"]
    return jsonify({
        "device_id": device.device_id,
        "pattern": pattern or "normal",
        "window_size": bundle["window_size"],
        "anomaly_score": score,
        "threshold": threshold,
        "is_anomalous": bool(score > threshold),
        "failure_risk": anomaly_service.risk_from_score(score, threshold),
        "latest_telemetry": samples[-1],
        "model_trained_at": bundle["trained_at"],
    }), 200


def _get_shared_monitor():
    """Retrieve injected SharedMonitor or initialize a default instance."""
    monitor = current_app.config.get("SHARED_MONITOR")
    if monitor is None:
        from modules.monitoring.shared_monitor import SharedMonitor

        loop = _get_cpu_action_loop()
        monitor = SharedMonitor(cpu_action_loop=loop)
        current_app.config["SHARED_MONITOR"] = monitor
    return monitor


@api_bp.route("/api/v1/monitor/overview", methods=["GET"])
def monitor_overview():
    """Consolidated cross-subsystem monitoring overview."""
    raw_history = request.args.get("include_history", "false").strip().lower()
    if raw_history in ("true", "1", "yes"):
        include_history = True
    elif raw_history in ("false", "0", "no", ""):
        include_history = False
    else:
        return jsonify({"error": "Query parameter 'include_history' must be a boolean ('true' or 'false')"}), 400

    raw_pid = request.args.get("pid")
    pid: Optional[int] = None
    if raw_pid is not None and raw_pid.strip():
        try:
            pid = int(raw_pid.strip())
            if pid < 0:
                return jsonify({"error": "Query parameter 'pid' must be a non-negative integer"}), 400
        except ValueError:
            return jsonify({"error": "Query parameter 'pid' must be an integer"}), 400

    try:
        monitor = _get_shared_monitor()
        snapshot = monitor.get_system_snapshot(include_history=include_history, pid=pid)
    except psutil.NoSuchProcess as e:
        return jsonify({"error": f"Process not found: {e}"}), 400
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception:
        return jsonify({"error": "Internal server error"}), 500

    return jsonify(snapshot), 200


def _get_integrated_execution_manager():
    """Retrieve injected IntegratedExecutionManager or initialize a consistent default instance."""
    manager = current_app.config.get("INTEGRATED_EXECUTION_MANAGER")
    if manager is None:
        from modules.integration.integrated_execution import IntegratedExecutionManager

        shared_mon = _get_shared_monitor()
        manager = IntegratedExecutionManager(
            shared_monitor=shared_mon,
            cpu_action_loop=shared_mon.cpu_action_loop,
            memory_monitor=shared_mon.memory_monitor,
        )
        current_app.config["INTEGRATED_EXECUTION_MANAGER"] = manager
    return manager


WORKLOAD_PROFILES = {
    "cpu_bound": "CPU_BOUND",
    "io_bound": "IO_BOUND",
    "priority_critical": "PRIORITY_CRITICAL",
    "balanced": "BALANCED",
}


def _generate_profile_workload(
    data: dict[str, Any],
) -> tuple[list[dict[str, Any]], Optional[dict[str, Any]], Optional[str]]:
    """Generate a process list for a named workload profile using the CPU workload generator."""
    from data.cpu.workload_classifier_data import generate_class_workload

    profile = data.get("workload_profile")
    if not isinstance(profile, str) or profile not in (*WORKLOAD_PROFILES, "random"):
        allowed = ", ".join([*WORKLOAD_PROFILES, "random"])
        return [], None, f"Field 'workload_profile' must be one of: {allowed}"

    num_processes = data.get("num_processes", 8)
    if isinstance(num_processes, bool) or not isinstance(num_processes, int) or not 1 <= num_processes <= 50:
        return [], None, "Field 'num_processes' must be an integer between 1 and 50"

    seed = data.get("seed")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
        return [], None, "Field 'seed' must be an integer or null"

    rng = random.Random(seed)
    if profile == "random":
        profile = rng.choice(list(WORKLOAD_PROFILES))

    processes, label = generate_class_workload(
        WORKLOAD_PROFILES[profile],
        num_processes=num_processes,
        seed=rng.randint(0, 10_000_000),
    )
    raw_processes = []
    for p in processes:
        raw_processes.append(
            {
                "pid": p.pid,
                "arrival_time": p.arrival_time,
                "burst_time": p.burst_time,
                "priority": p.priority,
                "workload_type": profile,
                "metadata": dict(p.metadata),
            }
        )
    info = {"profile": profile, "rule_label": label, "num_processes": len(raw_processes)}
    return raw_processes, info, None


@api_bp.route("/api/v1/system/step", methods=["POST"])
def system_step():
    """Execute a coordinated multi-resource execution step across CPU, Memory, Device, and Monitoring."""
    data, err = _extract_json_payload(allow_empty=False)
    if err:
        return jsonify({"error": err}), 400


    generated_workload: Optional[dict[str, Any]] = None
    if "processes" not in data and "workload_profile" in data:
        raw_processes, generated_workload, gen_err = _generate_profile_workload(data)
        if gen_err:
            return jsonify({"error": gen_err}), 400
    elif "processes" not in data:
        return jsonify({"error": "Missing required field: processes (or workload_profile)"}), 400
    else:
        raw_processes = data["processes"]
    if not isinstance(raw_processes, list):
        return jsonify({"error": "Field 'processes' must be a list"}), 400
    if len(raw_processes) == 0:
        return jsonify({"error": "Workload must contain at least one process"}), 400

    process_objects: list[Process] = []
    for i, p_dict in enumerate(raw_processes):
        if not isinstance(p_dict, dict):
            return jsonify({"error": f"Process at index {i} must be an object"}), 400

        for req_field in ("pid", "arrival_time", "burst_time"):
            if req_field not in p_dict:
                return jsonify({"error": f"Process at index {i} missing required field '{req_field}'"}), 400

        pid = p_dict["pid"]
        if not isinstance(pid, (int, str)):
            return jsonify({"error": f"Process at index {i} has invalid pid: must be int or str"}), 400

        arr = p_dict["arrival_time"]
        if isinstance(arr, bool) or not isinstance(arr, (int, float)):
            return jsonify({"error": f"Process at index {i} has invalid arrival_time: must be number"}), 400

        burst = p_dict["burst_time"]
        if isinstance(burst, bool) or not isinstance(burst, (int, float)):
            return jsonify({"error": f"Process at index {i} has invalid burst_time: must be number"}), 400
        if float(burst) < 0:
            return jsonify({"error": f"Process at index {i} has negative burst_time: {burst}"}), 400

        priority = p_dict.get("priority", 0)
        if isinstance(priority, bool) or not isinstance(priority, int):
            return jsonify({"error": f"Process at index {i} has invalid priority: must be integer"}), 400

        pred_burst = p_dict.get("predicted_burst_time")
        if pred_burst is not None and (isinstance(pred_burst, bool) or not isinstance(pred_burst, (int, float))):
            return jsonify({"error": f"Process at index {i} has invalid predicted_burst_time: must be number or null"}), 400

        meta = p_dict.get("metadata", {})
        if not isinstance(meta, dict):
            return jsonify({"error": f"Process at index {i} has invalid metadata: must be an object"}), 400

        try:
            p = Process(
                pid=pid,
                arrival_time=float(arr),
                burst_time=float(burst),
                priority=int(priority),
                predicted_burst_time=float(pred_burst) if pred_burst is not None else None,
                workload_type=p_dict.get("workload_type"),
                metadata=dict(meta),
            )
            process_objects.append(p)
        except (ValueError, TypeError) as e:
            return jsonify({"error": f"Invalid process at index {i}: {e}"}), 400


    limit_error = _check_workload_limits(process_objects)
    if limit_error:
        return jsonify({"error": limit_error}), 400

    compare_with_baseline = data.get("compare_with_baseline", False)
    if not isinstance(compare_with_baseline, bool):
        return jsonify({"error": "Field 'compare_with_baseline' must be a boolean"}), 400


    raw_device_actions = data.get("device_actions")
    if raw_device_actions is not None:
        if not isinstance(raw_device_actions, list):
            return jsonify({"error": "Field 'device_actions' must be a list when provided"}), 400
        for idx, action_req in enumerate(raw_device_actions):
            if not isinstance(action_req, dict):
                return jsonify({"error": f"Device action at index {idx} must be an object"}), 400
            for req_key in ("device_id", "failure_probability", "device_status"):
                if req_key not in action_req:
                    return jsonify({"error": f"Device action at index {idx} missing required field '{req_key}'"}), 400

            dev_id = action_req["device_id"]
            if not isinstance(dev_id, str) or not dev_id.strip():
                return jsonify({"error": f"Device action at index {idx} has invalid device_id: must be non-empty string"}), 400

            prob = action_req["failure_probability"]
            if isinstance(prob, bool) or not isinstance(prob, (int, float)):
                return jsonify({"error": f"Device action at index {idx} has invalid failure_probability: must be number"}), 400
            if not (0.0 <= float(prob) <= 1.0):
                return jsonify({"error": f"Device action at index {idx} failure_probability out of bounds [0.0, 1.0]"}), 400

            status = action_req["device_status"]
            if not isinstance(status, str) or not status.strip():
                return jsonify({"error": f"Device action at index {idx} has invalid device_status: must be non-empty string"}), 400


    pid = data.get("pid")
    if pid is not None and (isinstance(pid, bool) or not isinstance(pid, int) or pid < 0):
        return jsonify({"error": "Field 'pid' must be a non-negative integer or null"}), 400


    try:
        manager = _get_integrated_execution_manager()
        step_record = manager.step(
            processes=process_objects,
            device_actions=raw_device_actions,
            compare_with_baseline=compare_with_baseline,
            pid=pid,
        )
    except psutil.NoSuchProcess as e:
        return jsonify({"error": f"Process not found: {e}"}), 400
    except (ValueError, TypeError) as e:
        return jsonify({"error": str(e)}), 400
    except Exception:
        return jsonify({"error": "Internal server error"}), 500

    response = step_record.to_dict()
    if generated_workload is not None:
        response["generated_workload"] = generated_workload
    return jsonify(response), 200


@api_bp.route("/api/v1/system/reset", methods=["POST"])
def system_reset():
    """Clear integrated execution history and counters so the next step is step 1."""
    try:
        _get_integrated_execution_manager().reset()
    except Exception:
        return jsonify({"error": "Internal server error"}), 500
    return jsonify({"status": "reset"}), 200


@api_bp.route("/api/v1/system/history", methods=["GET"])
def system_history():
    """Return recorded integrated execution steps, newest first (read-only)."""
    raw_limit = request.args.get("limit", "20").strip()
    try:
        limit = int(raw_limit)
    except ValueError:
        return jsonify({"error": "Query parameter 'limit' must be an integer"}), 400
    if limit < 1 or limit > 200:
        return jsonify({"error": "Query parameter 'limit' must be between 1 and 200"}), 400

    try:
        history = _get_integrated_execution_manager().get_history()
    except Exception:
        return jsonify({"error": "Internal server error"}), 500

    steps = [record.to_dict() for record in reversed(history[-limit:])]
    return jsonify({"total_steps": len(history), "steps": steps}), 200
