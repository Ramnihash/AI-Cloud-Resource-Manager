# AI-Cloud-Resource-Manager

AI-driven cloud resource manager (OS project). It simulates a small cloud node and uses
machine learning to make decisions in three areas, with a web dashboard on top:

- **CPU:** a workload classifier and burst predictor pick the scheduling algorithm (FCFS, SJF,
  Priority, Round Robin) for each workload. If a model fails, the system falls back to FCFS.
- **Memory:** page replacement (FIFO, LRU, Optimal), Markov and Transformer page prediction,
  a memory-leak detector and a memory-pressure forecaster.
- **Device:** a Transformer anomaly detector scores simulated device telemetry. When risk is
  high, the action pipeline backs the device up (to S3 when configured) and reroutes I/O.

Everything runs on simulated workloads and devices. See [docs/RESULTS.md](docs/RESULTS.md)
for what was measured and what the numbers do and do not show.

## Install

Python 3.12 or newer. From the project root:

```
python -m venv .venv
.venv\Scripts\activate          # Windows PowerShell; on Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

PyTorch (in `requirements.txt`) is only needed for the device anomaly detector and the
Transformer page predictor. Without it those parts report "unavailable" and the rest works.

## Models and data

- CPU models are trained on synthetic data when the app starts, so nothing to prepare.
- The device anomaly detector must be trained once (the model file is not in git):
  `python -m scripts.train_device_anomaly_detector`
- Device telemetry comes from a deterministic simulator (`modules/device/`). There is no
  SMART dataset in this project, so the SMART classifier is not used.

## Run

```
python -m backend.app
```

Open http://127.0.0.1:5000/dashboard/. Pages: Overview, CPU, Memory, Device, Activity.
"Step System" runs one coordinated CPU + memory + device step; "Reset" (with a confirmation)
clears the history so the next step is step 1.

## Test and lint

```
pip install -r requirements-dev.txt     # pytest, ruff, playwright
python -m pytest -q                     # or: make test
ruff check .                            # or: make lint
```

Browser check (needs `pip install playwright` and `playwright install chromium`):

```
python -m scripts.e2e_check --base-url http://127.0.0.1:5000
```

The manual rows of the checklist are in [docs/E2E_CHECKLIST.md](docs/E2E_CHECKLIST.md).

## Evaluate

```
python -m experiments.run_all          # all studies, writes CSVs and plots to results/
python -m experiments.cpu_eval         # or one study at a time
```

Studies: 7.1 CPU, 7.2 memory, 7.3 device, 7.4 and 7.5 robustness and stress. Results and
interpretation: [docs/RESULTS.md](docs/RESULTS.md).

## Demo script

1. Open the Overview and click **Step System** a few times: watch the pipeline strip and the rings update.
2. CPU page: see which scheduler the AI chose for the workload and why.
3. Memory page: run page replacement and the leak and pressure analysis.
4. Device page: pick `disk_0`, a fault and a severity, click **Scan with detector**, then **Run action**.
   A backup record appears in the S3 bucket when `BACKUP_S3_BUCKET` is set.
5. Activity page: all of the above as one event log.
6. Stop the backend: the pages show an "unreachable" banner. Start it again and they recover. **Reset** returns to step 1.

## API

REST endpoints under `/api/v1/`, described with real request and response samples in
[docs/API_CONTRACT.md](docs/API_CONTRACT.md). Regenerate the samples with
`python -m scripts.capture_api_samples`.

## Deploy

AWS EC2 with gunicorn, nginx, HTTPS and Basic Auth: [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md).

## What is real and what is simulated

See [docs/CAPABILITIES.md](docs/CAPABILITIES.md) for a table of every capability and its status.

## Known limits

- State (step history, device actions) is in memory in one gunicorn worker and is lost on restart.
- Stress testing was sequential on one machine. Concurrent load was not tested.
- The memory-leak detector flags allocate-and-free and random-walk patterns as leaks, and the
  pressure forecaster alerts at about the moment memory crosses the threshold, not before.
- The CPU scheduler choice beats FCFS, Priority and Round Robin on waiting time but not SJF.
- All fault and workload data is simulated, so results do not predict real hardware.
