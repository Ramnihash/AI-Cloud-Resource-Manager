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
| CPU-bound | SJF (30/30) | 661.3 | 776.1 | 661.3 | 787.8 | 1299.4 |
| I/O-bound | Round Robin (30/30) | 40.7 | 34.5 | 21.6 | 35.7 | 40.7 |
| Priority-critical | Priority (30/30) | 52.2 | 53.3 | 33.0 | 52.2 | 64.2 |
| Balanced | FCFS (30/30) | 56.9 | 56.9 | 45.4 | 56.6 | 88.6 |

Paired differences in average waiting time over all 120 workloads (negative means the AI
choice is better, all p < 0.001 unless noted): AI vs FCFS -11.9%, vs Priority -13.0%, vs
Round Robin -45.7%, and vs SJF +6.6% (worse). Per type, the AI choice is 15% to 16% better
than FCFS and Priority on CPU-bound work, identical to SJF there, and 25% to 89% worse than
SJF on the other three types. Against FCFS on I/O-bound work it is 17.8% worse on waiting time.

Other metrics where the AI choice wins:
- **Response time, I/O-bound:** 14.5 vs 34.5 (FCFS), 21.6 (SJF) and 35.7 (Priority), which is
  33% to 59% lower. Over all workloads Round Robin still has the best response time (16.5 vs
  196.2 for the AI choice) because it is the only scheduler built for it.
- **Urgent-process waiting, priority-critical:** 6.0 vs 52.9 (FCFS), 29.4 (SJF) and 64.7
  (Round Robin), which is 80% to 91% lower. It is identical to the Priority scheduler there
  because that is the scheduler the classifier selected.

Other findings:
- **Utilization** is 99.3% to 99.97% for every scheduler (the simulator never leaves the CPU
  idle with work waiting), so utilization does not separate them.
- **Scheduling overhead** inside the simulator is 0.04 to 0.6 ms for every scheduler (the largest is Round Robin on CPU-bound work). Selecting
  the scheduler (burst prediction plus classification) adds about 4 ms of wall time per
  workload (3.9 to 4.7 ms vs 0.3 to 1.0 ms for a fixed scheduler).
- **Classifier choices:** on all 120 workloads the predicted class matched the ground-truth
  label and the fallback was never used.

**Interpretation.** The AI layer picks a sensible scheduler for what each workload class is
meant to optimise: Round Robin for responsiveness on I/O-bound work, Priority for urgent
processes, SJF for long CPU-bound batches. It clearly beats FCFS, Priority and Round Robin
on average waiting time, but it does not beat plain SJF, which gives the lowest average
waiting time on three of the four types (6.6% lower overall). So the claim the numbers
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

**D1. Next-page accuracy** (mean over 30 strings, top-1 / top-3; run on the development PC):

| Pattern | Markov | Transformer (API config) | Transformer (trained) |
|---------|--------|--------------------------|-----------------------|
| Uniform | 4.9% / 15.1% | 4.8% / 15.0% | 5.1% / 14.8% |
| Locality | 16.1% / 47.6% | 14.6% / 39.4% | 13.3% / 36.8% |
| Sequential | 100% / 100% | 25.1% / 49.9% | 100% / 100% |

Training takes a median of 0.15 ms for Markov, 150 ms for the API configuration and 4 s for the trained
one, per 1000-reference string (a few training runs took far longer, probably because the PC was busy,
so means are not reported). Scoring is about 0.005 ms per prediction when batched.

**D2. Page replacement**, demand faults (total loads for prefetch) out of 500, paired with Markov on the
same strings:

| Pattern | Frames | Protect: Markov / API / trained | Prefetch: Markov / API / trained |
|---------|--------|----------------------------------|----------------------------------|
| Uniform | 4 | 399.7 / 400.7 / 401.0 | 401.2 (767) / 402.1 (757) / 400.4 (793) |
| Uniform | 8 | 304.6 / 303.5 / 304.3 | 300.6 (545) / 302.1 (529) / 302.9 (597) |
| Locality | 4 | 240.2 / 241.6 / 245.0 | 221.2 (341) / 240.6 (356) / 245.0 (474) |
| Locality | 8 | 92.3 / 98.5 / 94.8 | 87.7 (94) / 101.1 (123) / 100.8 (186) |
| Sequential | 4 | 500 / 500 / 500 | 1 (501) / 375 (776) / 1 (501) |
| Sequential | 8 | 500 / 476 / 500 | 1 (501) / 351 (578) / 1 (501) |

On locality strings the Transformer is worse than Markov in every case (protect +0.6% to +6.7%, prefetch
+8.8% to +15.3% demand faults; p < 0.001 except the API configuration with protect at 4 frames, p = 0.30). On
uniform strings there is no difference in demand faults (p >= 0.17). On sequential strings the trained
Transformer matches Markov exactly, and the API configuration, which predicts the next page only 25% of
the time, hides 25% to 30% of stalls instead of nearly all of them.

**D3. Request-sized strings** (300 per pattern, top-1 / top-3 on the 30th reference):

| Pattern | Markov | Transformer (API config) | Transformer (trained) |
|---------|--------|--------------------------|-----------------------|
| Uniform | 2.7% / 6.0% | 2.7% / 11.7% | 4.3% / 14.7% |
| Locality | 9.7% / 26.7% | 11.3% / 33.0% | 12.3% / 33.0% |
| Sequential | 100% / 100% | 14.3% / 33.7% | 100% / 100% |

Training on a 29-reference string takes a median of 40 ms (API configuration) and 600 ms (trained).

**Interpretation.** On these synthetic patterns the Transformer adds nothing over a first-order Markov
table once there is enough history: the patterns are first-order by construction, so Markov is already the
right model, and the Transformer is at best equal (sequential, trained) and otherwise slightly worse, at
1,000 to 25,000 times the training cost. Its only edge is on very short strings, where its top-3
guesses are better than Markov's (for example 33% vs 27% on locality), because Markov has seen too few
transitions to rank three candidates; top-1 is about the same there. The configuration the API runs is
undertrained: it fails to learn even the cyclic pattern (25% on 1000 references, 14% on 29), so the
Transformer suggestion on the Memory page should be read as a demonstration, not as a better predictor.
More epochs fix the cyclic case but cost about 0.6 s per request on the PC, and the live instance has
no PyTorch, so the Transformer is not available there at all.

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

**Data used.** Backblaze Q1 2025 (2025-01-01 to 2025-03-31): 27.8 million drive-days, 1,067 failed
drives. The labelled sample has 61,833 rows from 35,045 drives: 25,833 failing days (all 1,067 failed
drives) and 36,000 healthy days. Only 0.093% of real drive-days are within 30 days of a failure.
The study ran on the development PC.

| Model | ROC-AUC | PR-AUC | Precision | Recall | FPR | Recall at 1% FPR | Precision at real prevalence (0.5 / 1% FPR) | Failed drives flagged |
|-------|---------|--------|-----------|--------|-----|------------------|---------------------------------------------|-----------------------|
| Random Forest | 0.902 ± 0.009 | 0.908 | 0.937 | 0.752 | 3.6% | 0.573 ± 0.019 | 1.9% / 5.1% | 85.6% |
| Logistic Regression | 0.894 ± 0.015 | 0.900 | 0.921 | 0.746 | 4.6% | 0.507 ± 0.028 | 1.5% / 4.5% | 85.6% |
| Rule (any of 5, 187, 188, 197, 198 > 0) | 0.869 ± 0.018 | 0.833 | 0.889 | 0.779 | 7.0% | 0.346 ± 0.026 | 1.0% / 5.1% | 87.2% |

Mean over 5 drive-grouped folds (± standard deviation over folds); precision, recall and FPR are at the
0.5 threshold for the models and "any attribute above 0" for the rule. Recall by days left (Random
Forest): 0.81 in the last week, 0.75 at 7-13 days and 0.72 at 14-29 days; the other two follow the same
pattern. The first flag on a caught drive comes a median of 29 days before failure, the start of the
window. The most important features are SMART 197 (pending sectors, 0.25), 5 (reallocated sectors,
0.25), 198 (0.13) and 187 (0.10); together the four warning attributes carry 72% of the importance.

**Interpretation.** On real drives the classifier ranks failing drives well above healthy ones (ROC-AUC
0.90), and most failing drives (86%) show it at some point in their last month, usually from the
start. Two caveats decide how useful that is. First, the simple rule is already strong: it catches
slightly more drives (87%) and its AUC is only 0.03 lower. The model's real gain is at strict
thresholds, where at a 1% false-positive rate it catches 57% of failing days against 35% for the rule.
Second, failures are rare. At the real rate of about 1 failing day in 1,000, even 1% false positives
means most alerts are false: about 1 in 20 flagged days (5%) belongs to a drive that fails within 30
days, and about 1 in 50 at the default 0.5 threshold. That is fine for triggering a cheap action like
a backup and not enough for replacing drives. About 14% of failed drives give no SMART warning at all,
which matches what Backblaze reports. Limits: one quarter only, so there is no test on a later period;
drive models are mixed, so the SMART 5/187/197 meanings differ slightly between vendors; and the
returned probability comes from a balanced sample, so it overstates real risk.

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
| CPU scheduler | average waiting 11.9% lower than FCFS over 120 workloads (p < 0.001); 6.6% higher than SJF | FCFS fallback, which is what runs when the model fails | 7.1 |
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
throughput but would split the in-memory state, which is why the deployment keeps one.

**Results (live EC2 instance, gunicorn with 1 worker and 4 threads, run on the instance against
127.0.0.1:8000 so nginx and the network are excluded).**

| Concurrent clients | Requests | Throughput (req/s) | Non-2xx | p50 (ms) | p95 (ms) | p99 (ms) | Max (ms) |
|--------------------|----------|--------------------|---------|----------|----------|----------|----------|
| 1 | 200 | 184.6 | 0 | 1.7 | 11.9 | 12.7 | 15.4 |
| 2 | 200 | 147.2 | 0 | 13.7 | 32.4 | 34.5 | 36.5 |
| 4 | 200 | 147.7 | 0 | 16.0 | 72.7 | 82.2 | 82.7 |
| 8 | 200 | 123.7 | 0 | 55.8 | 127.6 | 138.9 | 162.9 |

Integrity on the instance: 100 parallel steps at concurrency 4 and at 8 all returned 200 with 100
distinct step indices and 100 new history entries. The live run did not repeat the cold-start case,
because the service had already handled requests after the deploy; that case is covered by the local
run and the regression test.

The instance behaves like the local run, with slightly better numbers: no failures, and throughput
falls from about 185 to 125 requests per second as clients go from 1 to 8 instead of rising. With 8
clients a request waits a median of 56 ms (p99 139 ms), because gunicorn has only 4 threads and the
work inside the one process is serialised. For a demo with a handful of users that is fine. Many
simultaneous users would queue, and the next step would be more worker processes with the state
moved out of process memory (for example Redis or a small database).

## 7.6 Model selection: why each module uses the model it does

Script: `experiments/model_selection_eval.py` (scikit-learn only). Outputs: `results/model_selection_<part>.csv`
and `results/model_selection_<part>_folds.csv` for `cpu_burst`, `cpu_workload`, `smart` and `device`.

**Setup.** Each learned component is compared with common alternatives on identical splits. CPU models: 5
generated datasets (seeds 42 to 46) x 5 folds = 25 paired splits. SMART: the 5 drive-grouped folds of 7.3.
Device: the same normal training windows, threshold rule (1.5 x the largest held-out normal score) and
evaluation windows as 7.3; the Transformer row is copied from 7.3 because this study runs without PyTorch.
Every alternative is tested against the model in use with a paired t-test over folds (fold scores are not
independent, so p-values are indicative). Fit time, single-prediction latency and pickled size were recorded
too, since the server has about 1 GB of RAM.

| Part | Model in use | Best alternative | Result |
|------|--------------|------------------|--------|
| Burst-time regression | Polynomial ridge (degree 3): MAE 0.36, R² 0.9995, 1.4 KB (Random Forest until this study) | SVR (RBF): MAE 0.52 | **Changed to polynomial ridge.** Better than all 9 alternatives (p < 0.001), including the Random Forest used before (MAE 2.18, R² 0.977, 8.2 MB) |
| Workload classification | Random Forest: macro-F1 0.980 | SVM (RBF): 0.977 (p = 0.018) | **Confirmed.** Best of 9; significantly better than all 8 alternatives |
| SMART failure risk | Random Forest (100 trees, depth 10): AUC 0.902, recall 0.573 at 1% FPR, 3.8 MB | Unlimited RF (200 trees): AUC 0.900 (p = 0.50), 61 MB; logistic regression 0.894 (p = 0.057) | **Confirmed.** Best AUC and best recall at 1% FPR of 8; the depth cap costs nothing |
| Device anomaly | Transformer: recall 0.82 at severity 0.05, 0.98 at 0.1; FPR 1.7% noisy | One-Class SVM: 0.37 and 0.70; FPR 0% | **Confirmed for mild faults.** At full severity every detector except Isolation Forest is perfect |

**Interpretation.** Three of the four original choices held up. For burst time the Random Forest was beaten by
five models, because the synthetic burst time is a smooth product of the features divided by (1 + I/O wait),
which a cubic polynomial nearly reproduces while a forest approximates it with steps. The burst predictor was
therefore switched to polynomial ridge regression (DECISIONS D011). With the new predictor the workload
classifier was retrained and retested (its training features include predicted bursts): it is still the best
classifier, and 7.1 was rerun (scheduler choices unchanged, SJF slightly better because its predicted bursts are
more accurate, selection about 4 ms instead of 10 ms). The study was run before the switch with the Random Forest
as the reference; the table shows the rerun after the switch. Isolation Forest fails on device windows because
its scores sit in a narrow band (the 1.5x margin puts the threshold above every window) and, even at the
largest normal score, random splits on a flattened 50-value window rarely isolate a fault that fills half of it
(AUC 0.76).
