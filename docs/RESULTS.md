# Evaluation results

Every study is a script under `experiments/` that writes CSVs and plots to `results/`.
Rerun all with `python -m experiments.run_all`.

## 7.1 CPU: AI-selected scheduler vs FCFS, SJF, Priority and Round Robin

Script: `experiments/cpu_eval.py`. Outputs: `results/cpu_runs.csv`, `cpu_summary.csv`,
`cpu_paired.csv`, `cpu_selection.csv`, `cpu_waiting.png`, `cpu_utilization.png`.

**Setup.** The burst predictor and workload classifier are trained once (seed 42). Each of
the four workload types (CPU-bound, I/O-bound, priority-critical, balanced) is evaluated on
30 seeded 20-process workloads (seeds 1000 to 1029, none shared with training), 120 workloads
in total. Every workload is run through the AI-selected scheduler and through FCFS, SJF
(predicted bursts), Priority and Round Robin (quantum 2) on identical copies. Comparisons are
paired on the same workloads. A lower number is better. "Urgent waiting" is the mean waiting
time of the processes with the lowest priority number.

| Workload type | Scheduler chosen | Avg waiting: AI | FCFS | SJF | Priority | Round Robin |
|---------------|------------------|-----------------|------|-----|----------|-------------|
| CPU-bound | SJF (30/30) | 664.7 | 776.1 | 664.7 | 787.8 | 1299.4 |
| I/O-bound | Round Robin (30/30) | 40.7 | 34.5 | 21.6 | 35.7 | 40.7 |
| Priority-critical | Priority (30/30) | 52.2 | 53.3 | 33.2 | 52.2 | 64.2 |
| Balanced | FCFS (30/30) | 56.9 | 56.9 | 45.7 | 56.6 | 88.6 |

Paired differences in average waiting time over all 120 workloads (negative means the AI
choice is better, all p < 0.001 unless noted): AI vs FCFS -11.5%, vs Priority -12.6%, vs
Round Robin -45.4%, and vs SJF +6.4% (worse). Per type, the AI choice is 14% to 16% better
than FCFS and Priority on CPU-bound work, identical to SJF there, and 25% to 88% worse than
SJF on the other three types. Against FCFS on I/O-bound work it is 17.8% worse on waiting time.

Other metrics where the AI choice wins:
- **Response time, I/O-bound:** 14.5 vs 34.5 (FCFS), 21.6 (SJF) and 35.7 (Priority), which is
  33% to 59% lower. Over all workloads Round Robin still has the best response time (16.5 vs
  197.1 for the AI choice) because it is the only scheduler built for it.
- **Urgent-process waiting, priority-critical:** 6.0 vs 52.9 (FCFS), 30.0 (SJF) and 64.7
  (Round Robin), which is 80% to 91% lower. It is identical to the Priority scheduler there
  because that is the scheduler the classifier selected.

Other findings:
- **Utilization** is 99.3% to 99.97% for every scheduler (the simulator never leaves the CPU
  idle with work waiting), so utilization does not separate them.
- **Scheduling overhead** inside the simulator is 0.05 to 1.0 ms for every scheduler (the largest is Round Robin on CPU-bound work). Selecting
  the scheduler (burst prediction plus classification) adds about 10 ms of wall time per
  workload (9.9 to 11.2 ms vs 0.3 to 1.4 ms for a fixed scheduler).
- **Classifier choices:** on all 120 workloads the predicted class matched the ground-truth
  label and the fallback was never used.

**Interpretation.** The AI layer picks a sensible scheduler for what each workload class is
meant to optimise: Round Robin for responsiveness on I/O-bound work, Priority for urgent
processes, SJF for long CPU-bound batches. It clearly beats FCFS, Priority and Round Robin
on average waiting time, but it does not beat plain SJF, which gives the lowest average
waiting time on three of the four types (6.4% lower overall). So the claim the numbers
support is "better than the other three fixed schedulers on average waiting time, and better
on response time or urgent-process waiting for the types where the policy targets those", not
"better than the best fixed scheduler". Mapping balanced and priority-critical work to SJF
would lower average waiting time but would lose the urgent-process benefit. The 100%
class accuracy is measured on workloads from the same generator and labelling rules as the
training data, so it shows the pipeline is consistent, not how it would classify real workloads.

## 7.3 Device: anomaly detector on labelled simulated telemetry

Script: `experiments/device_eval.py`. Outputs: `results/device_anomaly_metrics.csv`,
`device_anomaly_scores.csv`, `device_detection_delay.csv`, `device_actions.csv`,
`device_roc.png`, `device_scores.png`, `device_recall.png`.

**Setup.** The detector (Transformer reconstructor, threshold = 1.5 x the largest anomaly
score on held-out normal data) was trained on sample indices 0-299 of the four-device
fleet. It is evaluated on windows of 10 samples starting at indices 400 and up (30 starts per
device). Each start gives one normal window and one window per fault pattern (5 patterns),
so 120 normal and 600 fault windows per condition. The simulator is deterministic, so
there is no run-to-run variation to average; a second condition adds Gaussian noise (seed 42,
5% of each feature's normal standard deviation) to every sample.

| Condition | Precision | Recall | F1 | False-positive rate | ROC-AUC |
|-----------|-----------|--------|----|---------------------|---------|
| Clean | 1.000 | 1.000 | 1.000 | 0.000 (0 of 120) | 1.000 |
| Noisy | 0.997 | 1.000 | 0.998 | 0.017 (2 of 120) | 1.000 |

All five fault patterns have recall 1.0 and AUC 1.0 in both conditions.

**Detection delay.** A fault starts at sample 30 of a 60-sample stream and a 10-sample
window slides over it. In both conditions every fault was detected in the first window that
contained a fault sample (delay 1 sample, standard deviation 0; 40 streams per pattern and
condition). In the noisy condition 7.5% to 15% of the streams also had at least one false
alarm before the fault started, because each stream holds about 21 overlapping normal windows.

**Actions.** Every flagged window was passed to the predictive-action pipeline with
simulated cloud backup: backup success 100% and reroute success 100% in both conditions
(600 and 602 actions; the 2 extra in the noisy case are the false alarms). Backup latency is
about 0.01 ms in process; this measures the simulation only, not a real S3 upload.

**Interpretation.** The detector separates normal and faulty windows perfectly on clean data
and misses nothing under noise, and its only cost under noise is a 1.7% false-positive rate,
which would trigger an unnecessary backup. This is expected: the simulated faults are large
(for example +45 C or +150 ms) and the training data is noise-free, so these numbers show the
detection, backup and reroute chain works end to end, not how it would perform on real
hardware or on subtle degradation. Only one noise level was tested, so the noise
robustness is not characterised beyond that point.

**Not done: SMART classifier.** No Backblaze dataset or trained SMART model exists in this
project, so the plan's SMART row (precision/recall/ROC-AUC on a held-out split) was not run.

## 7.4 Robustness: AI components forced to fail, and AI on vs off

Script: `experiments/robustness_eval.py`. Output: `results/robustness_forced_failures.csv`.
Each scenario breaks one component inside a test copy of the Flask app and sends the normal request.
"Graceful" means the HTTP status is the intended one (never an unhandled 500), the response says what
happened, and a warning is logged.

| Failure injected | Endpoint | HTTP | Response signal | Warning logged |
|------------------|----------|------|-----------------|----------------|
| CPU workload classifier raises | `/api/v1/cpu/action` | 200 | `is_fallback=True`, FCFS used, reason RuntimeError | yes |
| CPU burst predictor raises | `/api/v1/cpu/action` | 200 | `is_fallback=True`, FCFS used, reason RuntimeError | yes |
| Markov page predictor raises | `/api/v1/memory/analysis` | 200 | prediction block `available: False`, reason "Predictor failed: RuntimeError" | yes |
| Anomaly model file removed | `/api/v1/device/anomaly-score` | 503 | "not trained yet", with the retrain command | yes |
| Anomaly model file corrupt | `/api/v1/device/anomaly-score` | 503 | "could not be loaded", with the retrain command | yes |
| Anomaly model removed, predictive action still used | `/api/v1/device/predictive-action` | 200 | backup still ran, no reroute (none requested) | n/a |
| S3 unreachable | `/api/v1/device/predictive-action` | 200 | `success=False`, "S3 upload failed: ConnectionError" | yes |

All 7 scenarios were handled gracefully. Before this fix a raising model produced an HTTP 500 on the CPU
and memory endpoints; that was found by probing and fixed in this round, so these rows describe the fixed
code. Latency is a few milliseconds except the Markov row (8,987 ms). That was the first call in a fresh
app and probably includes building the predictor; it was not separated out, so treat it as cold-start
cost rather than a measured failure cost.

**AI on vs off.** The "off" state is the fallback the system already uses, or the baseline from each study.

| Component | AI on | AI off | Source |
|-----------|-------|--------|--------|
| CPU scheduler | average waiting 11.5% lower than FCFS over 120 workloads (p < 0.001); 6.4% higher than SJF | FCFS fallback, which is what runs when the model fails | 7.1 |
| Page replacement | predictive-protect cuts demand faults 5.6% (4 frames) and 12.9% (8 frames) on locality strings; no gain on random access | baseline replacement policy | 7.2 |
| Device anomaly detection | recall 1.000, false-positive rate 0.000 clean and 0.017 noisy | no anomaly score, so the system raises no flag of its own. This follows from the call chain; it was not measured | 7.3 |

**Interpretation.** Losing any single AI component degrades the response instead of breaking it: CPU
scheduling falls back to FCFS, memory analysis reports the predictor as unavailable, and the device
endpoints return a clear 503 with the fix. The cost of "AI off" is the difference in the table, which
is real for the CPU scheduler and for locality-based page prediction, and modest elsewhere.

## 7.5 Robustness: malformed requests and stress

Outputs: `results/stress_malformed.csv`, `results/stress_latency.csv`.

**Malformed requests.** 145 bad requests (empty or wrong-typed bodies, nulls, nested garbage, a very long
string, unicode, NaN and infinity, huge numbers, too many processes) were sent across all endpoints.
Result: 0 returned a 5xx, and every error response had a JSON error body. Statuses were 123 x 400,
13 x 200, 7 x 405 and 2 x 404. The 13 requests accepted with 200 are intended: `/api/v1/memory/monitor`
has no required fields (an empty or garbled body just returns a snapshot, 12 cases), and
`/api/v1/device/predictive-action` accepts any non-empty device id, including one that does not exist
(1 case).

**Stress.** One worker, in-process timing, so network time is excluded.

| Scenario | Requests | Non-200 | p50 (ms) | p95 (ms) | p99 (ms) | Max (ms) | Memory growth (MB) |
|----------|----------|---------|----------|----------|----------|----------|--------------------|
| CPU action, 20 processes | 20 | 0 | 9.7 | 10.9 | 13.2 | 13.8 | 0.04 |
| CPU action, 200 processes | 20 | 0 | 21.2 | 25.9 | 29.9 | 30.9 | 2.96 |
| CPU action, 1000 processes | 20 | 0 | 72.6 | 266.7 | 309.2 | 319.8 | 18.09 |
| CPU action, 2000 repeated steps | 2000 | 0 | 10.1 | 12.9 | 14.9 | 214.7 | 0.54 (0.01 in second half) |
| System step, 2000 repeated steps | 2000 | 0 | 12.3 | 15.9 | 19.0 | 34.6 | 10.78 (-0.03 in second half) |

No request failed and none returned a 5xx. Latency grows with workload size (about 10 ms at 20 processes,
73 ms at 1000). Over 2000 repeated steps, latency in the last 10% of requests matched the first 10%
and memory growth in the second half was about zero, so no leak was visible. History stays at 1000
entries because of the cap. The limit of 1000 processes per request is deliberate: without it a single
huge request hung the one worker.

**Limits of this test.** It is sequential, from one client, on the development machine. It says nothing
about concurrent users, and the live instance (about 1 GB RAM, one gunicorn worker) was not stress
tested. Requests there queue behind each other.
