# API contract

All endpoints are JSON. POST bodies need `Content-Type: application/json`. Errors always have the
shape `{"error": "<message>"}` with status 400 (bad input), 404/405 (unknown route or method),
503 (a model is unavailable) or 500 (unexpected). A real request and response for every endpoint
is in [`docs/api_samples/`](api_samples/); regenerate them with `python -m scripts.capture_api_samples`.

Limits that protect the single worker: at most 1000 processes per request, every time value
at most 10,000, and total burst time at most 50,000.

## Endpoints

| Method | Path | Purpose | Sample |
|--------|------|---------|--------|
| GET | `/health` | Liveness check, returns `{"status": "ok"}` | [health](api_samples/health.json) |
| POST | `/api/v1/cpu/action` | Classify a workload, pick and run a scheduler | [cpu_action](api_samples/cpu_action.json) |
| POST | `/api/v1/memory/page-replacement` | Run FIFO, LRU or Optimal on a reference string | [memory_page_replacement](api_samples/memory_page_replacement.json) |
| POST | `/api/v1/memory/monitor` | RAM snapshot, or a series when `count` and `interval_s` are given | [memory_monitor](api_samples/memory_monitor.json) |
| POST | `/api/v1/memory/analysis` | Page prediction, leak detection and pressure forecast | [memory_analysis](api_samples/memory_analysis.json) |
| POST | `/api/v1/device/anomaly-score` | Score a simulated telemetry window with the Transformer | [device_anomaly_score](api_samples/device_anomaly_score.json) |
| POST | `/api/v1/device/predictive-action` | Back up and reroute a device from a failure probability | [device_predictive_action](api_samples/device_predictive_action.json) |
| POST | `/api/v1/system/step` | One coordinated CPU + memory + device step | [system_step](api_samples/system_step.json) |
| GET | `/api/v1/system/history?limit=N` | Recorded steps, newest first (N 1 to 200, default 20) | [system_history](api_samples/system_history.json) |
| POST | `/api/v1/system/reset` | Clear history so the next step is step 1 | [system_reset](api_samples/system_reset.json) |
| GET | `/api/v1/monitor/overview` | Snapshot used by the Overview page | [monitor_overview](api_samples/monitor_overview.json) |

Dashboard pages (HTML): `/dashboard/`, `/dashboard/cpu`, `/dashboard/memory`, `/dashboard/device`, `/dashboard/activity`.

## Request fields

**`/api/v1/cpu/action`**: `processes` (required, list). Each process: `pid` (int or string),
`arrival_time` (number), `burst_time` (number, not negative) required; `priority` (int),
`predicted_burst_time` (number or null), `workload_type`, `metadata` (object with
`cpu_intensity`, `io_wait_ratio`, `instruction_count`, `process_complexity`) optional.
Also `compare_with_baseline` (boolean) and `pid` (int or null).

**`/api/v1/memory/page-replacement`**: `algorithm` (`fifo`, `lru` or `optimal`), `num_frames`
(int, 1 or more), `reference_string` (list of non-negative ints).

**`/api/v1/memory/monitor`**: none required. `pid` (int or null). With `count` (int, 1 or more)
and `interval_s` (number, 0 or more) it returns a series instead of one snapshot.

**`/api/v1/memory/analysis`**: `reference_string` (list of at least 3 non-negative ints),
`num_frames` (1 to 20, default 3), `samples` (8 to 30, default 12), `interval_s` (0 to 1, default 0.2).

**`/api/v1/device/anomaly-score`**: `device_id` (one of the fleet, for example `disk_0`),
`pattern` (optional: `normal`, `high_temperature`, `high_error_rate`, `high_latency`,
`high_queue`, `abnormal_io`).

**`/api/v1/device/predictive-action`**: `device_id`, `failure_probability` (0 to 1) and
`device_status` required; `backup_device_id` and `threshold` (0 to 1) optional.
Accepted statuses (case-insensitive): `healthy`, `available`, `normal`, `unhealthy`,
`degrading`, `failed`. Any other value returns 400.

**`/api/v1/system/step`**: either `processes` (as above) or `workload_profile`
(`cpu_bound`, `io_bound`, `priority_critical` or `balanced`) with `num_processes`; optional
`device_actions` (list of `device_id`, `failure_probability`, `device_status`),
`compare_with_baseline`, `pid`.

## Response shapes (top-level keys)

- **cpu/action**: `selected_scheduler`, `predicted_class`, `workload_class`, `confidence`,
  `is_fallback`, `fallback_reason`, `process_count`, `step_index`, `timestamp`,
  `meta_result` (includes `schedule_result` with `algorithm_name`, `gantt_chart`, `metrics`,
  `processes`), `baseline_result`, `action_delta`.
  When a model fails: `is_fallback: true`, FCFS used, `predicted_class: "MODEL_ERROR"`, `fallback_reason` set.
- **memory/page-replacement**: `algorithm`, `num_frames`, `page_faults`, `page_hits`,
  `hit_ratio`, `miss_ratio`, `execution_time_ms`, `frame_snapshots`, `reference_string`.
- **memory/monitor**: `ram_*_bytes`, `ram_percent`, `swap_*`, `timestamp`, and `proc_*` fields
  (null unless `pid` is given).
- **memory/analysis**: `page_prediction` (list of predictors with `available`, `predictor`,
  `candidates`, `action`), `leak_detection`, `pressure_forecast`, `health_action`, `samples`,
  `memory_state`, `current_page`, `frames_policy`, `reference_length`.
- **device/anomaly-score**: `device_id`, `pattern`, `window_size`, `anomaly_score`, `threshold`,
  `is_anomalous`, `failure_risk`, `latest_telemetry`, `model_trained_at`.
- **device/predictive-action**: `device_id`, `failure_probability`, `backup_result`
  (`triggered`, `success`, `backup_location`, `message`), `recovery_result` (`recovered`,
  `io_blocked`, `rerouted`, `backup_device_id`, `message`).
- **system/step**: `step_index`, `timestamp`, `cpu`, `memory`, `devices`, `system_snapshot`,
  `generated_workload` (only when `workload_profile` was used).
- **monitor/overview**: `status` (`ok`, `degraded` ...), `timestamp`, `cpu` (step totals,
  scheduler distribution, fallback rate), `memory` (as memory/monitor), `devices` (fleet list
  with `status`, `temperature`, `error_rate`, `response_latency`, `queue_length`, and `recent_actions`).

## Which ML model runs on which call

| Endpoint | Model use |
|----------|-----------|
| `cpu/action`, `system/step` | CPU workload classifier and burst predictor run on every request. A failure falls back to FCFS. |
| `memory/analysis` | Markov predictor is trained on the supplied reference string per request. The Transformer page predictor runs only if PyTorch is installed. The leak detector and pressure forecaster run on live RAM samples. A predictor failure returns `available: false` for that predictor. |
| `device/anomaly-score` | The trained Transformer. Missing or corrupt model file gives 503 with the retrain command. |
| `device/predictive-action` | No model. The failure probability comes from the caller (the Device page gets it from `anomaly-score`). |
| none | The SMART classifier is not used: the project has no SMART dataset. |

## Notes for clients

- Telemetry comes from a simulator, so `latest_telemetry` and device status are simulated.
- The device list and statuses in `monitor/overview` reflect actions taken since the last restart or reset.
- In-memory state (step history, device actions) is lost when the service restarts.
- The sample for `anomaly-score` in `api_samples/` shows 503 unless it was captured on a machine
  with PyTorch and the trained model.
