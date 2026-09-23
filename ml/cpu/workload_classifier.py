"""CPU Workload/Job-Mix Classifier for CP3.2.

Architecture & Pipeline:
    Processes in Workload Batch
        ↓
    CPUBurstPredictor (CP3.1)
        ↓
    Process.predicted_burst_time populated
        ↓
    Batch Feature Extraction (5 batch-level features)
        ↓
    CPUWorkloadClassifier (RandomForestClassifier, CP3.2)
        ↓
    ClassificationResult (effective_class, confidence, is_fallback)

Locked Classification Scope:
    Classifies the ENTIRE workload batch (not individual processes).

Locked Classes (4 classes):
    - CPU_BOUND
    - IO_BOUND
    - PRIORITY_CRITICAL
    - BALANCED

Locked Features (strictly in this exact order):
    1. mean_predicted_burst_time
    2. std_predicted_burst_time
    3. mean_cpu_intensity
    4. mean_io_wait_ratio
    5. priority_variance

Preprocessing:
    StandardScaler is explicitly NOT used. RandomForestClassifier is an invariant
    tree partitioner based on feature rank splits; omitting feature scaling
    eliminates scaling overhead and avoids train-serve synchronization hazards.
"""

from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

from ml.cpu.burst_predictor import CPUBurstPredictor
from modules.cpu.process import Process


CLASS_CPU_BOUND = "CPU_BOUND"
CLASS_IO_BOUND = "IO_BOUND"
CLASS_PRIORITY_CRITICAL = "PRIORITY_CRITICAL"
CLASS_BALANCED = "BALANCED"

WORKLOAD_CLASSES = [
    CLASS_BALANCED,
    CLASS_CPU_BOUND,
    CLASS_IO_BOUND,
    CLASS_PRIORITY_CRITICAL,
]


@dataclass(frozen=True)
class ClassificationResult:
    """Immutable result of a CPU workload batch classification."""

    predicted_class: str
    confidence: float
    class_probabilities: dict[str, float]
    is_fallback: bool
    effective_class: str
    features_used: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary."""
        return asdict(self)


class CPUWorkloadClassifier:
    """Machine learning classifier for CPU workload/job-mix classification.

    Classifies entire process batches into one of four locked classes:
    CPU_BOUND, IO_BOUND, PRIORITY_CRITICAL, or BALANCED.
    """

    FEATURE_NAMES = [
        "mean_predicted_burst_time",
        "std_predicted_burst_time",
        "mean_cpu_intensity",
        "mean_io_wait_ratio",
        "priority_variance",
    ]

    CLASSES = [
        CLASS_BALANCED,
        CLASS_CPU_BOUND,
        CLASS_IO_BOUND,
        CLASS_PRIORITY_CRITICAL,
    ]

    DEFAULT_FALLBACK_CLASS = CLASS_BALANCED

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = 10,
        random_state: int = 42,
        min_confidence: float = 0.40,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.min_confidence = float(min_confidence)

        self.model: Optional[RandomForestClassifier] = None
        self.feature_medians: dict[str, float] = {}
        self.evaluation_metrics: dict[str, Any] = {}
        self.classes_: list[str] = []
        self.is_fitted: bool = False

    def fit(
        self,
        X: np.ndarray | list[list[float]] | pd.DataFrame,
        y: np.ndarray | list[str] | pd.Series,
        test_size: float = 0.20,
        random_state: Optional[int] = None,
    ) -> dict[str, Any]:
        """Train the RandomForestClassifier on feature matrix X and string labels y.

        Calculates and stores feature medians from X_train ONLY for imputation fallback.
        Evaluates on held-out test split using actual Accuracy, Macro F1, and Confusion Matrix.
        """
        if isinstance(X, pd.DataFrame):
            if not all(col in X.columns for col in self.FEATURE_NAMES):
                missing = [col for col in self.FEATURE_NAMES if col not in X.columns]
                raise ValueError(f"Missing required feature columns in DataFrame: {missing}")
            X_arr = X[self.FEATURE_NAMES].to_numpy(dtype=np.float64)
        else:
            X_arr = np.asarray(X, dtype=np.float64)

        if isinstance(y, pd.Series):
            y_arr = y.to_numpy(dtype=str)
        else:
            y_arr = np.asarray(y, dtype=str)

        if X_arr.ndim != 2 or X_arr.shape[1] != len(self.FEATURE_NAMES):
            raise ValueError(
                f"Expected X with shape (N, {len(self.FEATURE_NAMES)}), got {X_arr.shape}"
            )
        if len(X_arr) != len(y_arr):
            raise ValueError(
                f"X and y must have same length, got {len(X_arr)} and {len(y_arr)}"
            )
        if len(X_arr) == 0:
            raise ValueError("Cannot fit on empty dataset.")

        split_seed = random_state if random_state is not None else self.random_state
        X_train, X_test, y_train, y_test = train_test_split(
            X_arr,
            y_arr,
            test_size=test_size,
            random_state=split_seed,
            stratify=y_arr,
        )


        self.feature_medians = {
            feat: float(np.median(X_train[:, idx]))
            for idx, feat in enumerate(self.FEATURE_NAMES)
        }


        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
        )
        self.model.fit(X_train, y_train)
        self.classes_ = [str(c) for c in self.model.classes_]
        self.is_fitted = True


        y_pred = self.model.predict(X_test)
        accuracy = float(accuracy_score(y_test, y_pred))
        macro_f1 = float(f1_score(y_test, y_pred, average="macro"))
        cm = confusion_matrix(y_test, y_pred, labels=self.classes_)
        report = classification_report(
            y_test,
            y_pred,
            labels=self.classes_,
            output_dict=True,
            zero_division=0,
        )

        self.evaluation_metrics = {
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "target_criteria_met": bool(accuracy >= 0.90 and macro_f1 >= 0.90),
            "confusion_matrix": {
                "labels": self.classes_,
                "matrix": cm.tolist(),
            },
            "classification_report": report,
            "train_size": len(X_train),
            "test_size": len(X_test),
        }
        return self.evaluation_metrics

    def train_on_synthetic_dataset(
        self,
        seed: int = 42,
        target_per_class: int = 250,
        test_size: float = 0.20,
    ) -> dict[str, Any]:
        """Convenience helper: generate 1,000 balanced batches via CP3.2 Step 1 and fit."""
        from data.cpu.workload_classifier_data import (
            WorkloadClassificationDatasetGenerator,
        )

        generator = WorkloadClassificationDatasetGenerator(
            seed=seed,
            target_per_class=target_per_class,
        )
        samples = generator.generate_dataset(seed=seed, target_per_class=target_per_class)
        X, y = WorkloadClassificationDatasetGenerator.get_feature_matrix_and_labels(samples)
        return self.fit(X, y, test_size=test_size, random_state=seed)

    def _prepare_input_matrix(
        self, X: np.ndarray | list[list[float]] | pd.DataFrame
    ) -> np.ndarray:
        """Validate and format input into an (N, 5) float64 ndarray."""
        if not self.is_fitted or self.model is None:
            raise RuntimeError(
                "CPUWorkloadClassifier is not fitted yet. Call fit() or load() before predicting."
            )

        if isinstance(X, pd.DataFrame):
            if not all(col in X.columns for col in self.FEATURE_NAMES):
                missing = [col for col in self.FEATURE_NAMES if col not in X.columns]
                raise ValueError(f"Missing required feature columns: {missing}")
            X_arr = X[self.FEATURE_NAMES].to_numpy(dtype=np.float64)
        else:
            X_arr = np.asarray(X, dtype=np.float64)
            if X_arr.ndim == 1:
                X_arr = X_arr.reshape(1, -1)
            if X_arr.shape[1] != len(self.FEATURE_NAMES):
                raise ValueError(
                    f"Expected input with {len(self.FEATURE_NAMES)} features, got {X_arr.shape[1]}"
                )

        return X_arr

    def predict(self, X: np.ndarray | list[list[float]] | pd.DataFrame) -> np.ndarray:
        """Predict workload classes for a batch of feature vectors."""
        X_arr = self._prepare_input_matrix(X)
        return self.model.predict(X_arr)

    def predict_proba(self, X: np.ndarray | list[list[float]] | pd.DataFrame) -> np.ndarray:
        """Return class probabilities array of shape (N, 4)."""
        X_arr = self._prepare_input_matrix(X)
        return self.model.predict_proba(X_arr)

    def predict_with_confidence(
        self, X: np.ndarray | list[list[float]] | pd.DataFrame
    ) -> list[tuple[str, float, bool]]:
        """Return list of (predicted_class, confidence, is_fallback) for each row."""
        probas = self.predict_proba(X)
        results: list[tuple[str, float, bool]] = []
        for prob_row in probas:
            best_idx = int(np.argmax(prob_row))
            confidence = round(float(prob_row[best_idx]), 4)
            pred_class = str(self.classes_[best_idx])
            is_fallback = confidence < self.min_confidence
            results.append((pred_class, confidence, is_fallback))
        return results

    def _extract_workload_features(self, processes: list[Process]) -> dict[str, float]:
        """Extract the 5 batch features, imputing missing/non-finite metadata with training medians.

        Uses population variance and standard deviation (ddof=0).
        True burst_time, arrival_time, and pid are never accessed.
        """
        pred_bursts: list[float] = []
        priorities: list[float] = []
        cpu_intensities: list[float] = []
        io_waits: list[float] = []

        default_median = 0.5

        for p in processes:

            p_burst = p.predicted_burst_time
            if p_burst is None or not math.isfinite(float(p_burst)):
                fallback_burst = self.feature_medians.get("mean_predicted_burst_time", 10.0)
                pred_bursts.append(float(fallback_burst))
            else:
                pred_bursts.append(float(p_burst))


            priorities.append(float(p.priority))


            meta = p.metadata if isinstance(p.metadata, dict) else {}

            cpu_val = meta.get("cpu_intensity", None)
            if cpu_val is None or not isinstance(cpu_val, (int, float)) or not math.isfinite(float(cpu_val)):
                fallback_cpu = self.feature_medians.get("mean_cpu_intensity", default_median)
                cpu_intensities.append(float(fallback_cpu))
            else:
                cpu_intensities.append(float(cpu_val))

            io_val = meta.get("io_wait_ratio", None)
            if io_val is None or not isinstance(io_val, (int, float)) or not math.isfinite(float(io_val)):
                fallback_io = self.feature_medians.get("mean_io_wait_ratio", default_median)
                io_waits.append(float(fallback_io))
            else:
                io_waits.append(float(io_val))

        mean_pred = float(np.mean(pred_bursts))
        std_pred = float(np.std(pred_bursts, ddof=0))
        mean_cpu = float(np.mean(cpu_intensities))
        mean_io = float(np.mean(io_waits))
        p_var = float(np.var(priorities, ddof=0))


        if not math.isfinite(mean_pred):
            mean_pred = self.feature_medians.get("mean_predicted_burst_time", 10.0)
        if not math.isfinite(std_pred):
            std_pred = self.feature_medians.get("std_predicted_burst_time", 2.0)
        if not math.isfinite(mean_cpu):
            mean_cpu = self.feature_medians.get("mean_cpu_intensity", default_median)
        if not math.isfinite(mean_io):
            mean_io = self.feature_medians.get("mean_io_wait_ratio", default_median)
        if not math.isfinite(p_var):
            p_var = self.feature_medians.get("priority_variance", 1.0)

        return {
            "mean_predicted_burst_time": round(mean_pred, 4),
            "std_predicted_burst_time": round(std_pred, 4),
            "mean_cpu_intensity": round(mean_cpu, 4),
            "mean_io_wait_ratio": round(mean_io, 4),
            "priority_variance": round(p_var, 4),
        }

    def predict_workload(
        self,
        processes: list[Process],
        burst_predictor: Optional[CPUBurstPredictor] = None,
    ) -> ClassificationResult:
        """Classify an entire incoming process batch.

        Leaves caller's Process objects unmutated.
        If any process lacks predicted_burst_time, burst_predictor must be provided to predict
        on a workload copy, or else ValueError is raised.
        """
        if not self.is_fitted or self.model is None:
            raise RuntimeError(
                "CPUWorkloadClassifier is not fitted yet. Call fit() or load() before predicting."
            )
        if not processes:
            raise ValueError("Cannot classify an empty workload batch.")

        needs_prediction = any(p.predicted_burst_time is None for p in processes)
        if needs_prediction:
            if burst_predictor is None:
                raise ValueError(
                    "Workload contains processes with missing predicted_burst_time, "
                    "but no burst_predictor was provided."
                )
            if not burst_predictor.is_fitted:
                raise RuntimeError("Provided burst_predictor is not fitted.")

            target_processes = [p.copy() for p in processes]
            burst_predictor.predict_workload(target_processes, inplace=True)
        else:
            target_processes = processes


        features_dict = self._extract_workload_features(target_processes)
        feature_vector = [features_dict[name] for name in self.FEATURE_NAMES]


        probas = self.predict_proba([feature_vector])[0]
        best_idx = int(np.argmax(probas))
        confidence = round(float(probas[best_idx]), 4)
        pred_class = str(self.classes_[best_idx])

        class_probabilities = {
            str(cls_name): round(float(p), 4)
            for cls_name, p in zip(self.classes_, probas)
        }


        if confidence < self.min_confidence:
            is_fallback = True
            effective_class = self.DEFAULT_FALLBACK_CLASS
        else:
            is_fallback = False
            effective_class = pred_class

        return ClassificationResult(
            predicted_class=pred_class,
            confidence=confidence,
            class_probabilities=class_probabilities,
            is_fallback=is_fallback,
            effective_class=effective_class,
            features_used=features_dict,
        )

    def save(self, filepath: str | Path) -> Path:
        """Serialize the fitted classifier, medians, metrics, and parameters using joblib."""
        if not self.is_fitted or self.model is None:
            raise RuntimeError(
                "Cannot save an unfitted CPUWorkloadClassifier. Fit the model first."
            )

        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "model_type": "CPUWorkloadClassifier",
            "format_version": "1.0",
            "model": self.model,
            "feature_names": list(self.FEATURE_NAMES),
            "classes": list(self.classes_),
            "feature_medians": dict(self.feature_medians),
            "evaluation_metrics": dict(self.evaluation_metrics),
            "hyperparameters": {
                "n_estimators": self.n_estimators,
                "max_depth": self.max_depth,
                "random_state": self.random_state,
                "min_confidence": self.min_confidence,
            },
        }
        joblib.dump(payload, path)
        return path

    @classmethod
    def load(cls, filepath: str | Path) -> CPUWorkloadClassifier:
        """Deserialize and validate a saved CPUWorkloadClassifier artifact."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Model artifact not found at {path}")

        try:
            payload = joblib.load(path)
        except Exception as e:
            raise ValueError(f"Failed to load artifact from {path}: {e}") from e

        if not isinstance(payload, dict):
            raise ValueError(f"Invalid artifact format in {path}: expected dict payload")

        if payload.get("model_type") != "CPUWorkloadClassifier":
            raise ValueError(
                f"Artifact model_type mismatch: expected 'CPUWorkloadClassifier', got '{payload.get('model_type')}'"
            )

        required_keys = [
            "model",
            "feature_names",
            "classes",
            "feature_medians",
            "evaluation_metrics",
            "hyperparameters",
        ]
        for key in required_keys:
            if key not in payload:
                raise ValueError(f"Corrupted artifact: missing '{key}' in payload")

        if payload["feature_names"] != cls.FEATURE_NAMES:
            raise ValueError(
                f"Feature names mismatch: expected {cls.FEATURE_NAMES}, got {payload['feature_names']}"
            )

        hp = payload["hyperparameters"]
        instance = cls(
            n_estimators=hp.get("n_estimators", 100),
            max_depth=hp.get("max_depth", 10),
            random_state=hp.get("random_state", 42),
            min_confidence=hp.get("min_confidence", 0.40),
        )
        instance.model = payload["model"]
        instance.classes_ = list(payload["classes"])
        instance.feature_medians = dict(payload["feature_medians"])
        instance.evaluation_metrics = dict(payload["evaluation_metrics"])
        instance.is_fitted = True
        return instance
