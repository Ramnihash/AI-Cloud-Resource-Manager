# SMART / Backblaze Disk Telemetry Dataset (CP2.5)

## 1. Purpose

This component provides data preparation for real-world hard-drive telemetry from the Backblaze Hard Drive Data project. The prepared dataset directly supports the later **CP3.8 Disk Failure-Risk Classifier** (using Logistic Regression / Random Forest) and the **CP4.5 Disk Backup Action** (triggering cloud backups upon high failure risk).

## 2. Strict Separation from CP2.4

The AI-Driven Cloud Resource Manager maintains two completely distinct telemetry datasets:

| Pipeline | Dataset Origin | Scope | Target ML Component |
|---|---|---|---|
| **CP2.4** (`data/device/`) | Synthetic device simulator | Multi-device fleet (`disk`, `printer`, `NIC`, `sensor`) metrics: temperature, latency, error rate, queue length, I/O activity | Transformer-based Sequence Anomaly Detector (CP3.7) |
| **CP2.5** (`data/device/smart/`) | Real Backblaze drive telemetry | Hard disk drive S.M.A.R.T. (Self-Monitoring, Analysis and Reporting Technology) attributes | Disk Failure-Risk Classifier (CP3.8) |

**These two datasets must never be mixed.**

## 3. Expected Backblaze Input

The preparation pipeline expects standard tabular Backblaze CSV records (such as daily drive performance logs). A typical raw Backblaze CSV includes:

- Operational date (`date`)
- Drive identifiers (`serial_number`, `model`, `capacity_bytes`)
- Daily failure status (`failure`)
- S.M.A.R.T. attribute pairs (`smart_<id>_normalized`, `smart_<id>_raw`)

## 4. Target Definition

The target column is:
- **`failure`**
  - `0`: Operational / healthy drive
  - `1`: Drive failed on this observation record

The pipeline validates that `failure` exists and contains strictly binary values ($0$ or $1$).

## 5. Metadata Exclusion

To ensure models generalize across hardware fleets and avoid learning drive-specific memorization or date leakage, non-feature metadata columns are automatically excluded:
- `serial_number`
- `date`
- `model`
- `capacity_bytes`

If any of these columns are absent from the input, the pipeline handles their absence gracefully without error.

## 6. SMART Feature Discovery Rule

The pipeline does **not** hardcode a brittle list of SMART attributes. Instead, it dynamically discovers all available feature columns prefixed with:
```python
smart_
```
This enables flexible compatibility across different Backblaze quarterly releases and vendor-specific SMART attribute subsets.

## 7. Numeric Conversion

All discovered SMART attributes are cast to numeric floats using deterministic conversion (`pd.to_numeric(..., errors="coerce")`). Any non-numeric artifacts, corrupt strings, or vendor blanks are coerced to `NaN` without terminating execution.

## 8. Missing Values and Median Imputation

Raw hard drive data frequently exhibits missing values because different drive manufacturers report different subsets of SMART attributes. For the initial CP2.5 preparation pipeline:
- Missing values in SMART feature columns are imputed using the **median value** computed from the input dataset.
- If a column contains entirely `NaN` values, a deterministic fallback of `0.0` is applied.
- *Design Note*: At this offline dataset preparation stage, median imputation is computed directly on the ingested input. Production-grade leak-free train/test transformers belong to the model-training phase in CP3.8.

## 9. Output Schema

The output is a clean pandas DataFrame (or exported CSV) containing:
```
[smart_feature_1, smart_feature_2, ..., failure]
```
The SMART feature columns are deterministically sorted alphabetically, followed by the target column `failure`.

## 10. Repository Policy

- **No Raw Data in Git**: Raw Backblaze CSV archives are large external assets and must **not** be committed to Git.
- **No Automatic Downloads**: Importing the preparation module will never initiate network requests or automatic downloads.
- **No Machine Learning Predictions**: This module is strictly responsible for dataset ingestion, sanitization, and preparation. Modeling and backup decisions are implemented in subsequent checkpoints.

## Validation Against Official Backblaze Schema

The CP2.5 preparation pipeline was validated against the official Backblaze Hard Drive Data schema.

The validation confirmed that:

- Backblaze daily drive records contain a binary `failure` target using `0` and `1`.
- SMART attributes follow the `smart_*` naming convention.
- Different drive models report different SMART attributes, resulting in substantial missing values.
- The current CP2.5 preparation pipeline is compatible with the Backblaze schema without code changes.
- Additional Backblaze metadata is excluded because the pipeline selects only `smart_*` attributes and the `failure` target.
- Full raw Backblaze datasets are not committed to the Git repository.
- Severe class imbalance, downstream SMART feature selection, and construction of future failure prediction windows are concerns for CP3.8/CP7 rather than changes to the CP2.5 preparation pipeline.

## Labelled forecasting set (CP7.3)

`build_failure_windows.py` turns a raw quarter into the set the classifier trains on: a day is
labelled 1 when that drive fails within 30 days, and healthy days are a seeded sample from drives
that did not fail. `scripts/build_smart_dataset.py` downloads the quarter and runs it:

    python -m scripts.build_smart_dataset            # downloads data_Q1_2025.zip (about 1 GB)
    python -m scripts.build_smart_dataset --zip path/to/data_Q1_2025.zip

The raw zip stays in `raw/` (git-ignored). The output `backblaze_failure_windows.csv.gz` and its
`.meta.json` (period, row counts, real prevalence, missing-value rates) are small and committed.
Data source: Backblaze Drive Stats, used with attribution to Backblaze.
