"""Training and evaluation utilities for the SMART disk failure classifier."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import train_test_split

from ml.device.smart.classifier import (
    ClassifierConfig,
    create_logistic_regression,
    create_random_forest,
)
from ml.device.smart.dataset import split_features_target


@dataclass
class TrainingResult:
    """Result produced by training and evaluating a classifier."""

    model_name: str
    model: object
    precision: float
    recall: float
    predictions: pd.Series
    probabilities: pd.Series


def train_classifier(
    dataframe: pd.DataFrame,
    model_name: str,
    config: ClassifierConfig | None = None,
) -> TrainingResult:
    """Train and evaluate one permitted SMART failure classifier."""
    if config is None:
        config = ClassifierConfig()

    features, target = split_features_target(dataframe)

    if target.nunique() < 2:
        raise ValueError(
            "Training requires both healthy (0) and failed (1) samples."
        )

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=target,
    )

    if model_name == "logistic_regression":
        model = create_logistic_regression(config)
    elif model_name == "random_forest":
        model = create_random_forest(config)
    else:
        raise ValueError(
            "Unsupported model. Use 'logistic_regression' or 'random_forest'."
        )

    model.fit(x_train, y_train)

    predictions = pd.Series(
        model.predict(x_test),
        index=x_test.index,
        name="prediction",
    )

    probabilities = pd.Series(
        model.predict_proba(x_test)[:, 1],
        index=x_test.index,
        name="failure_probability",
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0,
    )

    return TrainingResult(
        model_name=model_name,
        model=model,
        precision=float(precision),
        recall=float(recall),
        predictions=predictions,
        probabilities=probabilities,
    )
    