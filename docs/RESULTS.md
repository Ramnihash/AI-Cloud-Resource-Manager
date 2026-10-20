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

## 7.2 Memory: page replacement, page prediction, leak and pressure detection

Script: `experiments/memory_eval.py`. Outputs: `results/memory_replacement_runs.csv`,
`memory_replacement_summary.csv`, `memory_replacement_paired.csv`,
`memory_prediction_accuracy.csv`, `memory_leak_metrics.csv`, `memory_leak_delay.csv`,
`memory_pressure_metrics.csv`, `memory_page_faults.png`, `memory_prediction_accuracy.png`,
`memory_leak.png`.

### A. Page replacement and page prediction

**Setup.** For each pattern (uniform, locality, sequential) there are 30 seeded strings over 20
pages (seeds 2000 to 2029). The first 1000 references train the Markov predictor and the next
500 are the evaluation string, run from empty frames with 4 and with 8 frames. FIFO, LRU and
Optimal are the project's classical algorithms. The two predictive policies are defined in
the experiment script, because the memory action layer only emits PREFETCH and PROTECT
directives: *Predictive-protect* is LRU that refreshes the recency of the predicted next page when
it is already resident, and *Predictive-prefetch* also loads the predicted page ahead of time when
it is not resident. A prefetch is not a demand fault, so the table also gives total page loads
(demand faults plus prefetches). The Transformer predictor is evaluated separately in part D.

Demand page faults out of 500 references (mean over 30 strings, standard deviation in
`memory_replacement_summary.csv`):

| Pattern | Frames | FIFO | LRU | Optimal | Predictive-protect | Predictive-prefetch (total loads) |
|---------|--------|------|-----|---------|--------------------|-----------------------------------|
| Uniform | 4 | 400.0 | 400.5 | 282.2 | 399.7 | 401.2 (767.3) |
| Uniform | 8 | 304.1 | 304.6 | 168.0 | 304.6 | 300.6 (545.1) |
| Locality | 4 | 266.6 | 254.4 | 157.8 | 240.2 | 221.2 (341.3) |
| Locality | 8 | 146.3 | 106.0 | 64.5 | 92.3 | 87.7 (94.2) |
| Sequential | 4 | 500 | 500 | 422 | 500 | 1 (501) |
| Sequential | 8 | 500 | 500 | 320 | 500 | 1 (501) |

Next-page prediction accuracy of the Markov model on the evaluation strings (top-1 / top-3):
uniform 4.9% / 15.1%, locality 16.1% / 47.6%, sequential 100% / 100%.

Paired against LRU on the same strings:
- **Uniform:** no real change in demand faults (-0.2% to -1.3%, p >= 0.056). Prefetching adds 79%
  to 92% more page loads (p < 0.001), with only 12.5% to 29.8% of prefetched pages used.
- **Locality:** Predictive-protect lowers demand faults by 5.6% (4 frames) and 12.9% (8 frames),
  with no extra loads (p < 0.001). Predictive-prefetch lowers demand faults by 13.0% and 17.3%, but
  with 4 frames it loads 34.2% more pages in total, while with 8 frames it loads 11.1% fewer.
- **Sequential:** protecting alone changes nothing. Prefetching cuts demand faults from 500 to 1,
  but total page loads stay at 501 against 500, so the same I/O happens earlier instead of
  disappearing.
- **Runtime:** the predictive policies take about 2 ms per 500 references against 0.2 to 0.4 ms
  for FIFO and LRU, since they query the model each step (this is simulator wall time).
- **Optimal** is the lower bound and no online policy gets close to it on uniform or locality
  strings.

**Interpretation.** Prediction only helps when the access pattern has structure that a
first-order model can learn. On the cyclic pattern it predicts perfectly and prefetching hides
almost every stall, though it does not reduce the total pages moved. On the locality pattern it gives a
modest, statistically clear gain, with the largest effect from simply protecting the predicted
page. On random access it gives nothing and prefetching wastes I/O, which is why the claim is
limited to patterns with learnable structure. The patterns are synthetic, so the size of the
gains says nothing about real programs.

### B. Memory-leak detector against injected leaks

40 seeded sequences of 60 samples per type, with the leak starting at sample 30. Leak types are
fast (20 to 80 KB per step) and slow (4 to 12 KB per step). Non-leak types are flat, sawtooth
(allocate and free cycles), random walk, and a one-time allocation jump. The decision is the
detector's verdict on the final 10-sample window.

| Scope | Sequences | Precision | Recall | F1 | False-positive rate |
|-------|-----------|-----------|--------|----|---------------------|
| All | 240 | 0.808 | 1.000 | 0.894 | 0.119 |

Both leak types were detected 40 of 40 times, on average 5.7 samples (fast, std 0.5) and 6.0
samples (slow, std 1.0) after the leak started. False positives on the final window: sawtooth 35%
(14 of 40), random walk 12.5% (5 of 40), flat 0%, one-time jump 0%. Counting a flag in any
window of the stream, false alarms rise to 82.5% for sawtooth, 80% for random walk, 100% for the
one-time jump (flagged for a few windows just after the allocation) and 2.5% for flat memory.

**Interpretation.** The detector never misses a sustained leak, but it flags any stretch of
steady growth, so allocate-and-free cycles and random drift look like leaks and would trigger a
process restart. It needs a longer window or a check that growth persists before it can be trusted
for automatic restarts.

### C. Memory-pressure forecaster against injected pressure ramps

40 seeded series per type. A ramp rises 0.4 to 1.2 points per step until it reaches the 90%
threshold; stable (about 55%) and high-stable (about 80%) series stay below it. The forecaster
(linear, 5-step window) is trained on a separate mix of the three types, and an alert is raised
when its predicted next value reaches 90%.

| Series | Alerted | Alerted before reaching 90% | Mean lead (steps) | False alerts | Forecast MAE |
|--------|---------|-----------------------------|-------------------|--------------|--------------|
| Ramp | 100% | 0% | -0.3 (std 0.6) | n/a | 0.71 |
| Stable | n/a | n/a | n/a | 0% | 1.06 |
| High-stable | n/a | n/a | n/a | 0% | 0.69 |

**Interpretation.** Every ramp raised an alert and none of the 80 non-ramp series did, so it
separates pressure from normal load. But it predicts only one step ahead, so the alert arrives at
about the moment the series crosses 90%, on average 0.3 steps late. It confirms pressure
rather than giving early warning.

### D. Transformer page predictor on its own

Script: `experiments/transformer_page_eval.py` (needs PyTorch). Outputs: `results/memory_transformer_accuracy.csv`,
`memory_transformer_replacement.csv`, `memory_transformer_paired.csv`, `memory_transformer_request.csv`,
`memory_transformer_accuracy.png`.

**Setup.** The same 90 strings as part A (3 patterns x 30 seeds, train on the first 1000 references,
score the next 500), so every number is paired with the Markov result on the same string. Two
Transformer settings: the one `/api/v1/memory/analysis` runs (context 2, d_model 16, 20 epochs at
lr 0.001) and a trained one (context 4, d_model 32, 300 epochs at lr 0.01). D1 is next-page accuracy,
training time and prediction latency. D2 drives the part A replacement harness (LRU plus PROTECT,
and plus PREFETCH) with each predictor. D3 copies what the Memory page does: train on a short
request-sized string (29 references) and predict the 30th, 300 strings per pattern.

**Results.** Pending: this study needs PyTorch, so it runs on the development PC
(`python -m experiments.transformer_page_eval`) and the numbers are added here from its CSVs.

 on labelled simulated telemetry

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

**Milder faults.** The faults above are large, so a second sweep scales every fault down
(severity 1.0 is the full fault; 30 starts x 4 devices x 5 patterns = 600 windows per severity,
clean telemetry, same threshold). Output: `results/device_severity.csv`.

| Severity | Recall | Median score / threshold | Median risk | Windows with risk capped at 1.0 |
|----------|--------|--------------------------|-------------|---------------------------------|
| 1.0 | 1.000 | 47,320 | 1.000 | 100% |
| 0.5 | 1.000 | 9,467 | 1.000 | 87% |
| 0.25 | 1.000 | 1,462 | 1.000 | 59% |
| 0.1 | 0.978 | 83 | 0.892 | 20% |
| 0.05 | 0.822 | 6.9 | 0.784 | 20% |

By pattern at severity 0.05 the recall is 1.0 for abnormal I/O, error rate and latency,
0.80 for temperature and 0.31 for queue length; at 0.1 queue length is 0.89 and every other
pattern is 1.0. Queue length is an integer, so at severity 0.05 the fault is about one extra
queue entry, which is within normal variation.

Two findings. First, the detector is very sensitive: its threshold comes from noise-free
simulated data, so it is tiny, and even a fault at 5% of full size scores about 7 times the
threshold. That is also why 5% measurement noise produces false alarms (1.7%). Second, the
original risk mapping (linear, capped at 1.0 at 1.4x the threshold) gave 1.00 for almost every
flagged window, which made the Device page's risk always read 1.00. The mapping now rises on a
log scale from 0.7 at the threshold to 1.0 at 1000x the threshold, so mild faults give
graded values (about 0.78 at 7x and 0.89 at 83x) while severe ones still saturate. This is a
rescaling for display and for the backup trigger, not a calibrated probability.
The table is measured under the new mapping. With the old mapping 80% to 100% of flagged
windows were capped at 1.0 at every severity, including 80% at the mildest. Recall does not depend on the mapping.

### SMART disk-failure classifier on real Backblaze data

Scripts: `scripts/build_smart_dataset.py` (download and labelling), `experiments/smart_eval.py`.
Outputs: `results/smart_folds.csv`, `smart_metrics.csv`, `smart_recall_by_lead.csv`,
`smart_feature_importance.csv`, `smart_roc_pr.png`.

**Data.** One quarter of the public Backblaze Drive Stats data (real drives, about 90 daily files
with one row per drive per day). Backblaze marks `failure = 1` only on the day a drive fails, so the
builder relabels it as a forecast: a day is positive when that drive fails within 30 days, and
negatives are a seeded sample of 400 drive-days per day from drives that did not fail in the
quarter. Days of failed drives from before their 30-day window are dropped. The features are the raw
values of 19 SMART attributes (Backblaze's five failure indicators 5, 187, 188, 197, 198, plus wear,
power-on time and temperature); attributes a vendor does not report are imputed with the training
median and flagged with a missing-value indicator. The labelled sample is committed as
`data/device/smart/backblaze_failure_windows.csv.gz` with a `.meta.json` summary; the raw quarter
is not.

**Setup.** 5-fold cross-validation split by drive serial number (`StratifiedGroupKFold`), so no
drive is in both training and test data; rows of one drive are nearly identical from day to day
and a random row split would leak. Compared on the same folds: Random Forest (used by the backend),
Logistic Regression, and a no-training rule that flags a drive when any of SMART 5, 187, 188, 197 or
198 is above zero. Metrics: precision, recall, F1 and false-positive rate at the 0.5 threshold,
ROC-AUC, PR-AUC, recall at 1% and 0.1% false-positive rate, the share of failed drives flagged at
least once in their last 30 days, and recall by days left. The sample holds roughly as many failing
as healthy days while the real fleet has about 1 failing day in 1000, so precision on the sample is
optimistic; `precision_at_prevalence` converts recall and false-positive rate to the precision
expected at the quarter's real share of failing drive-days.

**Results.** Pending: Backblaze downloads are blocked from the build environment, so the dataset is
built and the study run on the development PC; the numbers are added here from its CSVs.

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

**Limits of this test.** It is sequential, from one client, on the development machine. Concurrent
users are covered in the next part.

### C. Concurrent users

Script: `experiments/concurrency_eval.py`. Outputs: `results/concurrency_latency.csv`,
`results/concurrency_integrity.csv` (one block of rows per target, named by `label`).

**Setup.** Parallel HTTP clients send requests to a running server. Load: at each concurrency level,
200 requests with a fixed mix (60% overview, 30% system step, 10% history). Integrity: 100 system
steps fired in parallel; every one must return 200, get its own step index and appear in the
history exactly once. Cold start: 8 parallel steps sent to a fresh server before anything has
initialised it. The local run uses the real app on a threaded server (one thread per connection)
in the 4-core build environment; the live instance (one gunicorn worker, 4 threads, about 1 GB RAM)
is a separate run from the instance itself.

**Bug found and fixed.** Before the fix, 8 parallel first requests to a fresh app all returned 200
but only 1 of the 8 steps reached the history, and the CPU models were trained 8 times. Each request
saw that the shared execution manager did not exist yet and built its own, and the last one
replaced the others. The backend now creates its shared objects once under a lock, and step records
are numbered and stored under a lock. A regression test (`tests/integration/test_concurrency.py`)
fails on the old code and passes now. On the live site this race could only happen on the first
requests after a restart, which is when the first visitors and the Step button usually arrive.

**Results (local, after the fix).**

| Concurrent clients | Requests | Throughput (req/s) | Non-2xx | p50 (ms) | p95 (ms) | p99 (ms) | Max (ms) |
|--------------------|----------|--------------------|---------|----------|----------|----------|----------|
| 1 | 200 | 169.0 | 0 | 2.6 | 13.9 | 18.9 | 19.4 |
| 4 | 200 | 77.8 | 0 | 21.1 | 140.8 | 159.5 | 189.8 |
| 8 | 200 | 82.9 | 0 | 47.4 | 267.0 | 298.1 | 313.1 |
| 16 | 200 | 71.1 | 0 | 117.3 | 554.7 | 585.4 | 645.8 |
| 32 | 200 | 75.1 | 0 | 288.4 | 870.3 | 913.4 | 984.5 |

Integrity: cold start 8 of 8 steps recorded with indices 0 to 7; 100 parallel steps at concurrency 16
and at 32 all returned 200 with 100 distinct indices and 100 new history entries. No request failed
at any level.

**Interpretation.** Correctness holds under parallel requests now, but throughput does not grow with
clients. The backend is one Python process and the work is CPU-bound (scikit-learn scoring and the
scheduler simulation), so threads take turns; with more than one client the total rate fell from
about 169 to about 70 to 80 requests per second and latency grew roughly in proportion to the number
of clients. A system step went from a median of 12 ms alone to 121 ms with 4 clients and 683 ms with 32.
The drop between 1 and 4 clients is larger than queueing alone explains and is probably thread
contention inside the one process; it was not investigated further. More workers would add
throughput but would split the in-memory state, which is why the deployment keeps one. The live
instance has fewer and slower cores, so its absolute numbers will be lower. Pending: the run on the
EC2 instance against gunicorn (`--url http://127.0.0.1:8000 --label ec2-gunicorn`).
