"""Serving helpers for the SMART disk failure-risk classifier (CP3.8 -> API).

The model is trained on the committed Backblaze sample
(``data/device/smart/backblaze_failure_windows.csv.gz``, built by
``scripts/build_smart_dataset.py``) the first time it is needed, then kept in
memory. Training a Random Forest on that sample takes a few seconds, so no model
file has to be copied to the server.

The probability means "this drive fails within the dataset's horizon (30 days)",
learned on a sample where failing days are heavily over-represented. It ranks
drives well but is not a calibrated real-world probability; RESULTS.md 7.3
gives the precision to expect at the real failure rate.
"""

from __future__ import annotations

import json
import logging
import math
import os
import threading
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from data.device.smart.build_failure_windows import DEFAULT_OUTPUT, FEATURE_COLUMNS, meta_path_for

logger = logging.getLogger(__name__)

DATASET_PATH = DEFAULT_OUTPUT
RISK_THRESHOLD = 0.5
SEED = 42

_lock = threading.Lock()
_cache: dict[str, Any] = {}


class SmartModelUnavailableError(RuntimeError):
    """The SMART dataset is missing, so no model can be trained."""


def _signed_log1p(x: np.ndarray) -> np.ndarray:
    # SMART raw values span many orders of magnitude (hours, sector counts, LBAs).
    return np.sign(x) * np.log1p(np.abs(x))


def build_pipeline(model_name: str, seed: int = SEED) -> Pipeline:
    """Imputation (with missing-value indicators) plus one of the permitted models."""
    if model_name == "logistic_regression":
        return Pipeline([
            ("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
            ("log", FunctionTransformer(_signed_log1p)),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)),
        ])
    if model_name == "random_forest":
        return Pipeline([
            ("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
            ("model", RandomForestClassifier(
                n_estimators=100, max_depth=10, min_samples_leaf=5,
                class_weight="balanced_subsample", n_jobs=1, random_state=seed,
            )),
        ])
    raise ValueError("Unsupported model. Use 'logistic_regression' or 'random_forest'.")


def load_dataset(path: str | None = None) -> pd.DataFrame:
    path = path or DATASET_PATH
    if not os.path.exists(path):
        raise SmartModelUnavailableError(
            "SMART dataset not found. Build it with: python -m scripts.build_smart_dataset"
        )
    frame = pd.read_csv(path)
    missing = [c for c in (*FEATURE_COLUMNS, "failure") if c not in frame.columns]
    if missing:
        raise SmartModelUnavailableError(f"SMART dataset is missing columns: {', '.join(missing)}")
    return frame


def get_model(path: str | None = None) -> dict[str, Any]:
    """Train (once per dataset file version) and return the serving bundle."""
    path = path or DATASET_PATH
    if not os.path.exists(path):
        raise SmartModelUnavailableError(
            "SMART dataset not found. Build it with: python -m scripts.build_smart_dataset"
        )
    mtime = os.path.getmtime(path)
    with _lock:
        cached = _cache.get(path)
        if cached and cached["mtime"] == mtime:
            return cached
        frame = load_dataset(path)
        features = frame[list(FEATURE_COLUMNS)].apply(pd.to_numeric, errors="coerce")
        target = frame["failure"].astype(int)
        if target.nunique() < 2:
            raise SmartModelUnavailableError("SMART dataset needs both failing and healthy rows.")
        pipeline = build_pipeline("random_forest")
        pipeline.fit(features, target)
        meta: dict[str, Any] = {}
        meta_path = meta_path_for(path)
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                meta = {}
        bundle = {
            "mtime": mtime,
            "pipeline": pipeline,
            "model": "random_forest",
            "threshold": RISK_THRESHOLD,
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "training_rows": int(len(frame)),
            "horizon_days": meta.get("horizon_days"),
            "data_period": [meta.get("first_day"), meta.get("last_day")] if meta else None,
        }
        _cache.clear()
        _cache[path] = bundle
        logger.info("SMART classifier trained on %d rows", len(frame))
        return bundle


def validate_attributes(attributes: Any) -> tuple[dict[str, float], str | None]:
    """Check a ``{"smart_5_raw": 12, ...}`` mapping; unknown keys are an error."""
    if not isinstance(attributes, dict) or not attributes:
        return {}, "Field 'smart_attributes' must be a non-empty object such as {\"smart_5_raw\": 0}"
    clean: dict[str, float] = {}
    for key, value in attributes.items():
        if key not in FEATURE_COLUMNS:
            return {}, f"Unknown SMART attribute '{key}'. Allowed: {', '.join(FEATURE_COLUMNS)}"
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            return {}, f"SMART attribute '{key}' must be a finite number or null"
        if float(value) < 0:
            return {}, f"SMART attribute '{key}' must not be negative"
        clean[key] = float(value)
    if not clean:
        return {}, "Field 'smart_attributes' must contain at least one numeric value"
    return clean, None


def predict_risk(attributes: dict[str, float], path: str | None = None) -> dict[str, Any]:
    """Failure probability for one drive's SMART raw values (missing ones are imputed)."""
    bundle = get_model(path)
    row = pd.DataFrame([{c: attributes.get(c, np.nan) for c in FEATURE_COLUMNS}])
    probability = float(bundle["pipeline"].predict_proba(row)[0, 1])
    return {
        "failure_probability": probability,
        "at_risk": probability >= bundle["threshold"],
        "threshold": bundle["threshold"],
        "model": bundle["model"],
        "horizon_days": bundle["horizon_days"],
        "attributes_used": sorted(attributes),
        "attributes_imputed": [c for c in FEATURE_COLUMNS if c not in attributes],
        "trained_at": bundle["trained_at"],
        "training_rows": bundle["training_rows"],
        "data_period": bundle["data_period"],
    }
