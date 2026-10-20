# What is real and what is simulated

Each capability is marked **Implemented** (the code exists and is tested), **Simulated** (it
acts on simulated data or a simulated resource), **Live host** (it reads the machine it runs
on), **Real dataset** (evaluated on data that did not come from our simulators), **Deployed**
(runs on the AWS instance) or **Limitation**. Evidence points to a section of
[RESULTS.md](RESULTS.md) or to the API contract.

| Capability | Status | Evidence and limits |
|------------|--------|---------------------|
| CPU workload classifier and burst predictor | Implemented, Simulated, Deployed | Trained on synthetic workloads at start-up. 100% class accuracy on workloads from the same generator, so it shows consistency, not real-world accuracy (7.1). |
| CPU scheduling (FCFS, SJF, Priority, Round Robin) | Simulated, Deployed | A scheduling simulator, not the OS scheduler. The AI choice beats FCFS, Priority and Round Robin on average waiting time (-11.5% vs FCFS) but is 6.4% worse than SJF (7.1). |
| Fallback when a CPU model fails | Implemented, Deployed | Falls back to FCFS and says so in the response (7.4). |
| Page replacement (FIFO, LRU, Optimal) | Simulated, Deployed | Runs on supplied reference strings (API contract). |
| Page prediction (Markov, Transformer) | Simulated | Gains only on patterns with learnable structure; synthetic patterns only (7.2). The Transformer page predictor has its own study (7.2 D); its numbers are pending a run on a machine with PyTorch. |
| Memory leak detector | Live host, Simulated | Monitors this backend's own RAM. Evaluated on injected leaks: recall 1.0 but 11.9% false positives, mostly allocate-and-free and random-walk patterns (7.2). |
| Memory pressure forecaster | Live host, Simulated | Predicts one step ahead, so alerts arrive at about the time memory crosses 90%, not before (7.2). |
| RAM and swap monitor | Live host, Deployed | Real psutil readings of the host (API contract). |
| Device telemetry | Simulated | A deterministic simulator of 4 devices. No real hardware is monitored. |
| Device anomaly detector (Transformer) | Implemented, Simulated, Deployed | Recall 1.0 on large faults, 0.82 at 5% of full size; queue-length faults at 5% are mostly missed (recall 0.31). 1.7% false positives under 5% noise. Trained on noise-free simulated data (7.3). |
| Failure-risk value | Implemented | A monotonic rescaling of the anomaly score (0.7 at the threshold, 1.0 at 1000x). Not a calibrated probability. |
| SMART disk-failure classifier | Implemented, Real dataset (results pending) | Random Forest trained on a labelled sample of real Backblaze drive data: predicts failure within 30 days from SMART raw values. Served by `/api/v1/device/smart-risk` and usable by predictive action. Evaluation splits by drive (7.3); numbers pending the dataset build on the PC. The probability is not calibrated to the real failure rate. |
| Predictive action (backup and reroute) | Simulated, Deployed | The failure probability comes from the caller, or from the SMART classifier when the caller sends SMART values. Reroute is simulated (API contract). |
| Cloud backup to S3 | Implemented, Deployed | Real uploads to a real bucket on the instance, with checksum, retries and a context record. The record says `simulated: true` because the devices hold no real files (API contract). |
| Dashboard (5 pages, Step System, Reset) | Implemented, Deployed | Browser-tested on the live site (E2E_CHECKLIST.md). The header status chip updates on every page (on load, after a step and every 30 seconds). |
| Behaviour when AI parts or S3 fail | Implemented | All 7 forced failures handled without a 500 (7.4). |
| Failed-request log | Implemented | Every `/api/` response with status 400 or higher is kept in memory (last 200) and shown on the Activity page (API contract). |
| Malformed input and load | Implemented | 145 malformed requests, no 5xx (7.5). |
| Concurrent users | Implemented | Up to 32 parallel clients locally with no failed request and no lost step, after fixing a start-up race that lost 7 of 8 parallel first steps. Throughput does not grow with clients (one process). Live-instance run pending (7.5 C). |
| Deployment (nginx, gunicorn, HTTPS, Basic Auth, fail2ban) | Deployed | https://cloudproject.cv/dashboard/ ([DEPLOYMENT.md](DEPLOYMENT.md)). One worker, state in memory, lost on restart. |

## Known limitations

- Every workload, memory pattern and device fault is simulated; the numbers do not predict real systems.
- Step history and device actions live in one process and reset on restart.
- Authentication is a single shared Basic Auth login.
- The anomaly detector's threshold comes from noise-free data, which is why it is very sensitive and raises false alarms under noise.
- Not evaluated: subtle gradual degradation over time, real hardware. The SMART classifier is evaluated on one Backblaze quarter only, without a test on a later quarter.
