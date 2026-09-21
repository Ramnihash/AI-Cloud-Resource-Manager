"""Model comparison utilities for SMART disk failure-risk classification."""

from __future__ import annotations

import pandas as pd

from ml.device.smart.classifier import ClassifierConfig
from ml.device.smart.train import TrainingResult, train_classifier


def evaluate_models(
    dataframe: pd.DataFrame,
    config: ClassifierConfig | None = None,
) -> dict[str, TrainingResult]:
    """Train and evaluate both permitted SMART classifiers."""
    results = {}

    for model_name in (
        "logistic_regression",
        "random_forest",
    ):
        results[model_name] = train_classifier(
            dataframe,
            model_name,
            config,
        )

    return results