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
from typing import Any, Optional

import psutil
from flask import Blueprint, current_app, jsonify, request

from modules.cpu.action_loop import CPUActionLoop
from modules.cpu.process import Process
from modules.device.predictive_action import handle_predictive_action
from modules.memory.memory_monitor import MemoryMonitor
from modules.memory.page_replacement import (
    FIFOPageReplacement,
    LRUPageReplacement,
    OptimalPageReplacement,
)

api_bp = Blueprint("api", __name__)


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


@api_bp.route("/api/v1/system/step", methods=["POST"])
def system_step():
    """Execute a coordinated multi-resource execution step across CPU, Memory, Device, and Monitoring."""
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

    return jsonify(step_record.to_dict()), 200


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
