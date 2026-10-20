# Evaluation results

Every study is a script under `experiments/` that writes CSVs and plots to `results/`.
Rerun all with `python -m experiments.run_all`.

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
