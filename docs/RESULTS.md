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
