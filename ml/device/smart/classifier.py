"""Disk failure-risk classifier for SMART/Backblaze data."""

from __future__ import annotations

from dataclasses import dataclass

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


@dataclass(frozen=True)
class ClassifierConfig:
    """Configuration for SMART disk failure-risk classification."""

    random_state: int = 42
    test_size: float = 0.2
    logistic_max_iter: int = 1000
    random_forest_estimators: int = 100
    class_weight: str = "balanced"


def create_logistic_regression(config: ClassifierConfig):
    """Create the permitted Logistic Regression classifier."""
    return LogisticRegression(
        max_iter=config.logistic_max_iter,
        random_state=config.random_state,
        class_weight=config.class_weight,
    )


def create_random_forest(config: ClassifierConfig):
    """Create the permitted Random Forest classifier."""
    return RandomForestClassifier(
        n_estimators=config.random_forest_estimators,
        random_state=config.random_state,
        n_jobs=-1,
        class_weight=config.class_weight,
    )