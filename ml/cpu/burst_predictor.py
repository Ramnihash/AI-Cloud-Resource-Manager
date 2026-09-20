"""CPU burst-time regression predictor using scikit-learn RandomForestRegressor."""

from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from data.cpu.generator import CPUWorkloadGenerator, WorkloadConfig
from modules.cpu.process import Process


class CPUBurstPredictor:
    """Machine learning regression model predicting CPU burst time from process workload features.

    Architecture & Preprocessing:
        - Estimator: RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
        - Preprocessing Note: StandardScaler is explicitly NOT used. RandomForestRegressor
          is a decision tree ensemble invariant to monotonic feature scaling. Operating
          directly on original feature scales avoids unnecessary transformation overhead
          and guarantees scale transparency.
        - Features (strictly in this order):
            1. instruction_count
            2. cpu_intensity
            3. io_wait_ratio
            4. process_complexity
        - Target:
            burst_time (continuous positive float)
    """

    FEATURE_NAMES = [
        "instruction_count",
        "cpu_intensity",
        "io_wait_ratio",
        "process_complexity",
    ]

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = 10,
        random_state: int = 42,
        min_prediction_floor: float = 0.1,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.min_prediction_floor = float(min_prediction_floor)

        self.model: Optional[RandomForestRegressor] = None
        self.feature_medians: dict[str, float] = {}
        self.evaluation_metrics: dict[str, float] = {}
        self.is_fitted: bool = False

    def fit(
        self,
        X: np.ndarray | list[list[float]],
        y: np.ndarray | list[float],
        test_size: float = 0.20,
        random_state: Optional[int] = None,
    ) -> dict[str, float]:
        """Train the RandomForestRegressor on feature matrix X and target vector y.

        Calculates and stores feature medians from the training split for fallback.
        Evaluates on the held-out test split and returns MAE, MSE, RMSE, and R2.
        """
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64)

        if X_arr.ndim != 2 or X_arr.shape[1] != len(self.FEATURE_NAMES):
            raise ValueError(
                f"Expected X with shape (N, {len(self.FEATURE_NAMES)}), got {X_arr.shape}"
            )
        if len(X_arr) != len(y_arr):
            raise ValueError(
                f"X and y must have same length, got {len(X_arr)} and {len(y_arr)}"
            )

        split_seed = random_state if random_state is not None else self.random_state
        X_train, X_test, y_train, y_test = train_test_split(
            X_arr, y_arr, test_size=test_size, random_state=split_seed
        )

        # Calculate and store training-set medians for inference fallback
        self.feature_medians = {
            feat: float(np.median(X_train[:, idx]))
            for idx, feat in enumerate(self.FEATURE_NAMES)
        }

        # RandomForestRegressor does not require feature scaling
        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
        )
        self.model.fit(X_train, y_train)
        self.is_fitted = True

        # Evaluate on held-out test split
        y_pred = self.model.predict(X_test)
        mae = float(mean_absolute_error(y_test, y_pred))
        mse = float(mean_squared_error(y_test, y_pred))
        rmse = float(math.sqrt(mse))
        r2 = float(r2_score(y_test, y_pred))

        self.evaluation_metrics = {
            "mae": mae,
            "mse": mse,
            "rmse": rmse,
            "r2": r2,
        }
        return self.evaluation_metrics

    def train_on_synthetic_workload(
        self,
        num_processes: int = 2000,
        noise_level: float = 0.2,
        seed: int = 42,
        test_size: float = 0.20,
    ) -> dict[str, float]:
        """Generate 2,000 synthetic processes via CPUWorkloadGenerator and train the predictor."""
        config = WorkloadConfig(
            num_processes=num_processes,
            correlate_burst_to_features=True,
            noise_level=noise_level,
            seed=seed,
        )
        generator = CPUWorkloadGenerator(config=config)
        workload = generator.generate()
        df = generator.to_dataframe(workload)

        # Strictly extract only the four approved ML features
        X = df[self.FEATURE_NAMES].to_numpy(dtype=np.float64)
        y = df["burst_time"].to_numpy(dtype=np.float64)

        return self.fit(
            X,
            y,
            test_size=test_size,
            random_state=seed,
        )

    def predict(self, X: np.ndarray | list[list[float]]) -> np.ndarray:
        """Predict burst times for an array of feature vectors.

        Ensures predictions are strictly positive (>= min_prediction_floor).
        """
        if not self.is_fitted or self.model is None:
            raise RuntimeError(
                "CPUBurstPredictor is not fitted yet. Call fit() or load() before predicting."
            )

        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)
        if X_arr.shape[1] != len(self.FEATURE_NAMES):
            raise ValueError(
                f"Expected input with {len(self.FEATURE_NAMES)} features, got {X_arr.shape[1]}"
            )

        raw_pred = self.model.predict(X_arr)
        # Guarantee strictly positive burst-time prediction
        bounded_pred = np.maximum(self.min_prediction_floor, raw_pred)
        return bounded_pred

    def _extract_features_from_process(self, process: Process) -> list[float]:
        """Extract only the four approved features from Process.metadata.

        Fallback behavior:
        If any ML feature is missing, NaN, or non-numeric in process.metadata,
        the stored training-set median for that feature is used as an imputation fallback.
        """
        row: list[float] = []
        meta = process.metadata if isinstance(process.metadata, dict) else {}
        for feat in self.FEATURE_NAMES:
            val = meta.get(feat, None)
            if val is None or not isinstance(val, (int, float)) or math.isnan(val):
                # Fallback to stored training-set median
                fallback_val = self.feature_medians.get(feat, 1.0)
                row.append(float(fallback_val))
            else:
                row.append(float(val))
        return row

    def predict_process(self, process: Process) -> float:
        """Predict CPU burst time for a single Process instance.

        Extracts only the four approved ML features from process.metadata.
        Returns a finite, strictly positive float rounded to 2 decimal places.
        """
        if not self.is_fitted or self.model is None:
            raise RuntimeError(
                "CPUBurstPredictor is not fitted yet. Call fit() or load() before predicting."
            )

        features = self._extract_features_from_process(process)
        pred_arr = self.predict([features])
        return round(float(pred_arr[0]), 2)

    def predict_workload(
        self,
        processes: list[Process],
        inplace: bool = False,
    ) -> list[Process]:
        """Batch-predict burst times and populate predicted_burst_time on each Process.

        Leaves true Process.burst_time, pid, arrival_time, and priority completely unmutated.
        """
        if not self.is_fitted or self.model is None:
            raise RuntimeError(
                "CPUBurstPredictor is not fitted yet. Call fit() or load() before predicting."
            )

        target_processes = processes if inplace else [p.copy() for p in processes]
        if not target_processes:
            return []

        feature_matrix = [self._extract_features_from_process(p) for p in target_processes]
        predictions = self.predict(feature_matrix)

        for p, pred in zip(target_processes, predictions):
            p.predicted_burst_time = round(float(pred), 2)

        return target_processes

    def save(self, filepath: str | Path) -> Path:
        """Serialize the fitted predictor and metadata using joblib."""
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Cannot save an unfitted CPUBurstPredictor. Fit the model first.")

        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        artifact = {
            "model": self.model,
            "feature_names": list(self.FEATURE_NAMES),
            "feature_medians": dict(self.feature_medians),
            "evaluation_metrics": dict(self.evaluation_metrics),
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "random_state": self.random_state,
            "min_prediction_floor": self.min_prediction_floor,
            "model_type": "RandomForestRegressor",
        }
        joblib.dump(artifact, path)
        return path

    @classmethod
    def load(cls, filepath: str | Path) -> CPUBurstPredictor:
        """Restore a CPUBurstPredictor from a serialized joblib artifact."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Model artifact not found: {path}")

        artifact = joblib.load(path)
        if not isinstance(artifact, dict) or "model" not in artifact:
            raise ValueError(f"Invalid model artifact at {path}: Missing required model payload.")

        required_keys = {"model", "feature_names", "feature_medians"}
        if not required_keys.issubset(artifact.keys()):
            raise ValueError(
                f"Corrupted model artifact at {path}: Missing keys {required_keys - artifact.keys()}"
            )

        predictor = cls(
            n_estimators=artifact.get("n_estimators", 100),
            max_depth=artifact.get("max_depth", 10),
            random_state=artifact.get("random_state", 42),
            min_prediction_floor=artifact.get("min_prediction_floor", 0.1),
        )
        predictor.model = artifact["model"]
        predictor.feature_medians = artifact.get("feature_medians", {})
        predictor.evaluation_metrics = artifact.get("evaluation_metrics", {})
        predictor.is_fitted = True
        return predictor
