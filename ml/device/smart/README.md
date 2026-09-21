# SMART Disk Failure-Risk Classifier

## Purpose

This component implements CP3.8 — Disk Failure-Risk Classifier.

It uses prepared SMART/Backblaze disk data from CP2.5 to predict disk failure risk.

## Input

Prepared dataset from:

`data/device/smart/`

The dataset contains:
- SMART attributes
- failure target

The target is:
- `0` — healthy
- `1` — failed

## Models

The project specification permits:
- Logistic Regression
- Random Forest

CP3.8 will evaluate the permitted models before selecting the classifier used by the system.

## Prediction

The classifier produces:
- predicted class
- failure probability / risk score

The failure probability represents the estimated disk failure risk.

## Decision

The prediction will later be consumed by CP4.5.

When the configured failure-risk threshold is reached, the system will trigger the disk backup action.

CP3.8 is responsible for prediction and evaluation. CP4.5 is responsible for the backup action.

## Evaluation

The classifier will be evaluated using:
- Precision
- Recall
- Failure-risk prediction performance

Because Backblaze data is highly imbalanced, evaluation must not rely on accuracy alone.

## Data Handling

Raw Backblaze data is not committed to the repository.

The CP2.5 preparation pipeline is responsible for:
- SMART feature discovery
- numeric conversion
- missing-value handling
- failure-target validation

CP3.8 is responsible for model training and evaluation.

## Scope

This component does not implement:
- cloud backup
- device recovery
- dashboard integration
- final prediction-to-action integration