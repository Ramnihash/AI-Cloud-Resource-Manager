"""Model-selection study: why each module uses the model it does.

Usage (from the repository root; scikit-learn only, PyTorch is not needed):

    python -m experiments.model_selection_eval            # all four parts
    python -m experiments.model_selection_eval --only cpu_burst,smart

Each part compares the model the backend uses against common alternatives on
the same data and the same folds, and records accuracy, training time,
single-prediction latency and the size of the fitted model (the server has
about 1 GB of RAM, so size and latency matter as much as accuracy).

Part A  cpu_burst      burst-time regression (4 process features), 5 dataset seeds x 5 folds
Part B  cpu_workload   workload-class classification (5 batch features), 5 seeds x 5 stratified folds
Part C  smart          SMART failure classification on the Backblaze sample, 5 folds split by drive
Part D  device         unsupervised telemetry anomaly detectors on the CP7.3 windows; the
                       Transformer row is read from results/device_anomaly_metrics.csv and
                       results/device_severity.csv (it needs PyTorch to rerun)

For A to C a paired t-test over folds compares every model with the one in use.

Writes to ``results/``: model_selection_<part>_folds.csv and model_selection_<part>.csv
"""

from __future__ import annotations

import argparse
import os
import pickle
import time
import warnings
from typing import Callable

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    IsolationForest,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import KFold, StratifiedGroupKFold, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor, LocalOutlierFactor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVC, SVR, OneClassSVM
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

RESULTS_DIR = "results"
SEED = 42
N_FOLDS = 5
DATA_SEEDS = [42, 43, 44, 45, 46]
PARTS = ["cpu_burst", "cpu_workload", "smart", "device"]


def _size_kb(model) -> float:
    return len(pickle.dumps(model)) / 1024.0


def _latency_ms(model, row, repeats: int = 50, method: str = "predict") -> float:
    fn = getattr(model, method)
    fn(row)
    t0 = time.perf_counter()
    for _ in range(repeats):
        fn(row)
    return (time.perf_counter() - t0) * 1000.0 / repeats


def _paired(folds: pd.DataFrame, metric: str, reference: str, higher_is_better: bool) -> pd.DataFrame:
    """Paired t-test over identical folds: each model against the one in use."""
    ref = folds[folds.model == reference].sort_values(["seed", "fold"])[metric].to_numpy()
    rows = []
    for name, g in folds.groupby("model", sort=False):
        vals = g.sort_values(["seed", "fold"])[metric].to_numpy()
        diff = vals - ref
        if name == reference or np.allclose(diff, 0):
            p = float("nan")
        else:
            p = float(stats.ttest_rel(vals, ref).pvalue)
        better = diff.mean() > 0 if higher_is_better else diff.mean() < 0
        rows.append({"model": name, f"{metric}_diff_vs_{reference}": float(diff.mean()), "p_value": p,
                     "verdict": "in use" if name == reference else
                     ("not significant" if not (p < 0.05) else ("better" if better else "worse"))})
    return pd.DataFrame(rows)


def _summarise(folds: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    agg = folds.groupby("model", sort=False)[metrics].agg(["mean", "std"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    return agg.reset_index()


# ---------------------------------------------------------------- Part A
def burst_models() -> dict[str, object]:
    scaled = lambda m: make_pipeline(StandardScaler(), m)  # noqa: E731
    return {
        "random_forest (in use)": RandomForestRegressor(n_estimators=100, max_depth=10, random_state=SEED, n_jobs=1),
        "linear_regression": LinearRegression(),
        "polynomial_ridge_deg3": make_pipeline(StandardScaler(), PolynomialFeatures(3), Ridge(alpha=1.0)),
        "decision_tree": DecisionTreeRegressor(max_depth=10, random_state=SEED),
        "knn_k5": scaled(KNeighborsRegressor(n_neighbors=5)),
        "svr_rbf": scaled(SVR(C=10.0)),
        "extra_trees": ExtraTreesRegressor(n_estimators=100, max_depth=10, random_state=SEED, n_jobs=1),
        "gradient_boosting": GradientBoostingRegressor(random_state=SEED),
        "hist_gradient_boosting": HistGradientBoostingRegressor(random_state=SEED),
        "mlp_64x64": scaled(MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=2000, early_stopping=True,
                                         random_state=SEED)),
    }


def burst_data(seed: int) -> tuple[np.ndarray, np.ndarray]:
    from data.cpu.generator import CPUWorkloadGenerator, WorkloadConfig
    from ml.cpu.burst_predictor import CPUBurstPredictor

    gen = CPUWorkloadGenerator(config=WorkloadConfig(num_processes=2000, correlate_burst_to_features=True,
                                                     noise_level=0.2, seed=seed))
    df = gen.to_dataframe(gen.generate())
    return df[CPUBurstPredictor.FEATURE_NAMES].to_numpy(float), df["burst_time"].to_numpy(float)


def run_cpu_burst(out: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for seed in DATA_SEEDS:
        X, y = burst_data(seed)
        for fold, (tr, te) in enumerate(KFold(N_FOLDS, shuffle=True, random_state=seed).split(X)):
            for name, proto in burst_models().items():
                model = clone(proto)
                t0 = time.perf_counter()
                model.fit(X[tr], y[tr])
                fit_ms = (time.perf_counter() - t0) * 1000.0
                pred = model.predict(X[te])
                rows.append({"model": name, "seed": seed, "fold": fold,
                             "mae": mean_absolute_error(y[te], pred),
                             "rmse": float(np.sqrt(mean_squared_error(y[te], pred))),
                             "r2": r2_score(y[te], pred), "fit_ms": fit_ms,
                             "predict_one_ms": _latency_ms(model, X[te][:1], 20) if fold == 0 else np.nan,
                             "size_kb": _size_kb(model) if fold == 0 else np.nan})
    folds = pd.DataFrame(rows)
    summary = _summarise(folds, ["mae", "rmse", "r2", "fit_ms", "predict_one_ms", "size_kb"])
    summary = summary.merge(_paired(folds, "mae", "random_forest (in use)", higher_is_better=False), on="model")
    return folds, summary.sort_values("mae_mean").reset_index(drop=True)


# ---------------------------------------------------------------- Part B
def workload_models() -> dict[str, object]:
    scaled = lambda m: make_pipeline(StandardScaler(), m)  # noqa: E731
    return {
        "random_forest (in use)": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=SEED, n_jobs=1),
        "logistic_regression": scaled(LogisticRegression(max_iter=5000)),
        "gaussian_naive_bayes": GaussianNB(),
        "decision_tree": DecisionTreeClassifier(max_depth=10, random_state=SEED),
        "knn_k5": scaled(KNeighborsClassifier(n_neighbors=5)),
        "svm_rbf": scaled(SVC(C=10.0, probability=True, random_state=SEED)),
        "extra_trees": ExtraTreesClassifier(n_estimators=100, max_depth=10, random_state=SEED, n_jobs=1),
        "hist_gradient_boosting": HistGradientBoostingClassifier(random_state=SEED),
        "mlp_64x64": scaled(MLPClassifier(hidden_layer_sizes=(64, 64), max_iter=2000, early_stopping=True,
                                          random_state=SEED)),
    }


def run_cpu_workload(out: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    from data.cpu.workload_classifier_data import WorkloadClassificationDatasetGenerator as Gen

    rows = []
    for seed in DATA_SEEDS:
        samples = Gen(seed=seed, target_per_class=250).generate_dataset(seed=seed, target_per_class=250)
        X, y = Gen.get_feature_matrix_and_labels(samples)
        X = np.asarray(X, float)
        y = np.unique(np.asarray(y), return_inverse=True)[1]  # MLP early stopping needs numeric labels
        for fold, (tr, te) in enumerate(StratifiedKFold(N_FOLDS, shuffle=True, random_state=seed).split(X, y)):
            for name, proto in workload_models().items():
                model = clone(proto)
                t0 = time.perf_counter()
                model.fit(X[tr], y[tr])
                fit_ms = (time.perf_counter() - t0) * 1000.0
                pred = model.predict(X[te])
                rows.append({"model": name, "seed": seed, "fold": fold,
                             "accuracy": accuracy_score(y[te], pred),
                             "macro_f1": f1_score(y[te], pred, average="macro"), "fit_ms": fit_ms,
                             "predict_one_ms": _latency_ms(model, X[te][:1], 20, "predict_proba") if fold == 0 else np.nan,
                             "size_kb": _size_kb(model) if fold == 0 else np.nan})
    folds = pd.DataFrame(rows)
    summary = _summarise(folds, ["accuracy", "macro_f1", "fit_ms", "predict_one_ms", "size_kb"])
    summary = summary.merge(_paired(folds, "macro_f1", "random_forest (in use)", higher_is_better=True), on="model")
    return folds, summary.sort_values("macro_f1_mean", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------- Part C
def smart_models() -> dict[str, object]:
    from ml.device.smart.service import build_pipeline

    impute = lambda: SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)  # noqa: E731
    return {
        "random_forest (in use)": build_pipeline("random_forest"),
        "logistic_regression": build_pipeline("logistic_regression"),
        "decision_tree": Pipeline([("impute", impute()), ("model", DecisionTreeClassifier(
            max_depth=10, min_samples_leaf=5, class_weight="balanced", random_state=SEED))]),
        "extra_trees": Pipeline([("impute", impute()), ("model", ExtraTreesClassifier(
            n_estimators=100, max_depth=10, min_samples_leaf=5, class_weight="balanced_subsample",
            n_jobs=1, random_state=SEED))]),
        "random_forest_unlimited": Pipeline([("impute", impute()), ("model", RandomForestClassifier(
            n_estimators=200, min_samples_leaf=1, class_weight="balanced_subsample", n_jobs=1, random_state=SEED))]),
        # HistGradientBoosting handles missing values natively, so no imputer.
        "hist_gradient_boosting": HistGradientBoostingClassifier(class_weight="balanced", random_state=SEED),
        "knn_k25": Pipeline([("impute", impute()), ("scale", StandardScaler()),
                             ("model", KNeighborsClassifier(n_neighbors=25))]),
        "gaussian_naive_bayes": Pipeline([("impute", impute()), ("model", GaussianNB())]),
    }


def _tpr_at_fpr(y, scores, target=0.01) -> tuple[float, float]:
    fpr, tpr, _ = roc_curve(y, scores)
    ok = fpr <= target
    i = int(np.argmax(np.where(ok, tpr, -1)))
    return float(tpr[i]), float(fpr[i])


def run_smart(out: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    import json

    from data.device.smart.build_failure_windows import FEATURE_COLUMNS, meta_path_for
    from ml.device.smart.service import DATASET_PATH, load_dataset

    data = load_dataset(DATASET_PATH).reset_index(drop=True)
    meta_path = meta_path_for(DATASET_PATH)
    prevalence = float(json.loads(meta_path.read_text())["prevalence"]) if meta_path.exists() else float("nan")
    X = data[list(FEATURE_COLUMNS)].apply(pd.to_numeric, errors="coerce")
    y = data["failure"].astype(int).to_numpy()
    groups = data["serial_number"].astype(str).to_numpy()
    rows = []
    splitter = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for fold, (tr, te) in enumerate(splitter.split(X, y, groups)):
        for name, proto in smart_models().items():
            model = clone(proto)
            t0 = time.perf_counter()
            model.fit(X.iloc[tr], y[tr])
            fit_ms = (time.perf_counter() - t0) * 1000.0
            scores = model.predict_proba(X.iloc[te])[:, 1]
            tpr, fpr = _tpr_at_fpr(y[te], scores)
            denom = prevalence * tpr + (1 - prevalence) * fpr
            rows.append({"model": name, "seed": SEED, "fold": fold,
                         "roc_auc": roc_auc_score(y[te], scores),
                         "pr_auc": average_precision_score(y[te], scores),
                         "recall_at_fpr_1pct": tpr,
                         "precision_at_prevalence_fpr_1pct": prevalence * tpr / denom if denom > 0 else np.nan,
                         "fit_ms": fit_ms,
                         "predict_one_ms": _latency_ms(model, X.iloc[te][:1], 20, "predict_proba") if fold == 0 else np.nan,
                         "size_kb": _size_kb(model) if fold == 0 else np.nan})
            print(f"  smart fold {fold} {name}: AUC {rows[-1]['roc_auc']:.3f}", flush=True)
    folds = pd.DataFrame(rows)
    metrics = ["roc_auc", "pr_auc", "recall_at_fpr_1pct", "precision_at_prevalence_fpr_1pct",
               "fit_ms", "predict_one_ms", "size_kb"]
    summary = _summarise(folds, metrics)
    summary = summary.merge(_paired(folds, "roc_auc", "random_forest (in use)", higher_is_better=True), on="model")
    return folds, summary.sort_values("roc_auc_mean", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------- Part D
TRAIN_SAMPLES, VAL_SAMPLES, THRESHOLD_MARGIN = 200, 100, 1.5


def _normal_windows(fleet, start: int, count: int) -> np.ndarray:
    from ml.device.anomaly_service import WINDOW_SIZE
    from ml.device.sequence import create_telemetry_sequences
    from modules.device.telemetry import generate_normal_telemetry

    frame = pd.DataFrame([generate_normal_telemetry(d, float(i), sample_index=i).to_dict()
                          for d in fleet for i in range(start, start + count)])
    return create_telemetry_sequences(frame, WINDOW_SIZE)


class _Detector:
    """Wraps a scikit-learn novelty detector so that a higher score means more anomalous."""

    def __init__(self, fit: Callable, score: Callable):
        self._fit, self._score = fit, score

    def fit(self, X):
        self.state = self._fit(X)
        return self

    def score(self, X) -> np.ndarray:
        return self._score(self.state, X)


def device_detectors() -> dict[str, _Detector]:
    def sk(make):
        return _Detector(lambda X: make().fit(X), lambda m, X: -m.score_samples(X))

    def pca_fit(X):
        return PCA(n_components=0.95, random_state=SEED).fit(X)

    def pca_score(p, X):
        return ((X - p.inverse_transform(p.transform(X))) ** 2).mean(axis=1)

    return {
        "z_score_max": _Detector(lambda X: None, lambda _, X: np.abs(X).max(axis=1)),
        "pca_reconstruction": _Detector(pca_fit, pca_score),
        "isolation_forest": sk(lambda: IsolationForest(n_estimators=200, random_state=SEED)),
        "one_class_svm": sk(lambda: OneClassSVM(nu=0.01, gamma="scale")),
        "local_outlier_factor": sk(lambda: LocalOutlierFactor(n_neighbors=20, novelty=True)),
    }


def run_device(out: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    from experiments.device_eval import EVAL_START, N_STARTS, NOISE_LEVELS, SEVERITIES, START_STEP, _noisy
    from ml.device.anomaly_service import FAULT_PATTERNS, WINDOW_SIZE, build_window
    from ml.device.sequence import TELEMETRY_FEATURES
    from modules.device.fleet import create_default_fleet

    fleet = create_default_fleet()
    train = _normal_windows(fleet, 0, TRAIN_SAMPLES)
    val = _normal_windows(fleet, TRAIN_SAMPLES, VAL_SAMPLES)
    flat = train.reshape(-1, len(TELEMETRY_FEATURES))
    mean, std = flat.mean(axis=0), np.maximum(flat.std(axis=0), 1e-6)
    norm = lambda w: ((w - mean) / std).reshape(len(w), -1)  # noqa: E731

    # The same evaluation windows as experiments/device_eval.py, built once.
    cases = []
    for cond_index, (condition, level) in enumerate(NOISE_LEVELS.items()):
        rng = np.random.default_rng(SEED + cond_index)
        for device in fleet:
            for k in range(N_STARTS):
                start = EVAL_START + k * START_STEP
                for pattern in [None, *FAULT_PATTERNS]:
                    window, _ = build_window(device, pattern, WINDOW_SIZE, start)
                    cases.append((condition, 1.0, pattern or "normal", int(pattern is not None),
                                  _noisy(window, std, level, rng)))
    for device in fleet:
        for k in range(N_STARTS):
            start = EVAL_START + k * START_STEP
            for severity in SEVERITIES:
                for pattern in FAULT_PATTERNS:
                    window, _ = build_window(device, pattern, WINDOW_SIZE, start, severity=severity)
                    cases.append(("severity", severity, pattern, 1, window))
    windows = norm(np.stack([c[4] for c in cases]))
    meta = pd.DataFrame([c[:4] for c in cases], columns=["condition", "severity", "pattern", "label"])

    rows = []
    for name, det in device_detectors().items():
        t0 = time.perf_counter()
        det.fit(norm(train))
        fit_ms = (time.perf_counter() - t0) * 1000.0
        val_scores = det.score(norm(val))
        # Same rule as the Transformer: 1.5 x the highest score on held-out normal windows.
        # Scores can be negative for score_samples-based detectors, so the margin is applied to the
        # distance above the lowest validation score.
        floor = float(val_scores.min())
        scores = det.score(windows)
        for margin in (THRESHOLD_MARGIN, 1.0):
            threshold = floor + (float(val_scores.max()) - floor) * margin
            label = name if margin == THRESHOLD_MARGIN else f"{name} [threshold = max normal]"
            frame = meta.assign(score=scores, flagged=scores > threshold)
            for condition in ["clean", "noisy"]:
                g = frame[frame.condition == condition]
                neg = g.label == 0
                fp = int((neg & g.flagged).sum())
                tp = int((~neg & g.flagged).sum())
                rows.append({"model": label, "condition": condition,
                             "precision": tp / (tp + fp) if tp + fp else 0.0,
                             "recall": float(g.flagged[~neg].mean()),
                             "false_positive_rate": fp / int(neg.sum()),
                             "roc_auc": roc_auc_score(g.label, g.score),
                             "fit_ms": fit_ms, "size_kb": _size_kb(det.state)})
            sev = frame[frame.condition == "severity"].groupby("severity").flagged.mean()
            for s, r in sev.items():
                rows.append({"model": label, "condition": f"severity {s:g}", "recall": float(r)})

    # Transformer results from experiments/device_eval.py (needs PyTorch, not rerun here).
    tm = os.path.join(out, "device_anomaly_metrics.csv")
    ts = os.path.join(out, "device_severity.csv")
    if os.path.exists(tm):
        t = pd.read_csv(tm)
        for _, r in t[t.pattern == "all"].iterrows():
            rows.append({"model": "transformer (in use)", "condition": r.condition, "precision": r.precision,
                         "recall": r.recall, "false_positive_rate": r.false_positive_rate, "roc_auc": r.roc_auc})
    if os.path.exists(ts):
        s = pd.read_csv(ts)
        if "pattern" in s:
            s = s[s.pattern.notna() & (s.pattern != "all")]  # per-pattern rows have equal window counts
        for severity, r in s.groupby("severity").recall.mean().items():
            rows.append({"model": "transformer (in use)", "condition": f"severity {severity:g}", "recall": r})
    detail = pd.DataFrame(rows)
    wide = detail.pivot_table(index="model", columns="condition", values="recall", aggfunc="first")
    main = detail[detail.condition.isin(["clean", "noisy"])].pivot_table(
        index="model", columns="condition", values=["false_positive_rate", "roc_auc"], aggfunc="first")
    main.columns = [f"{a}_{b}" for a, b in main.columns]
    wide.columns = [f"recall_{c}" for c in wide.columns]
    summary = wide.join(main).reset_index()
    return detail, summary


# ----------------------------------------------------------------
RUNNERS = {"cpu_burst": run_cpu_burst, "cpu_workload": run_cpu_workload, "smart": run_smart, "device": run_device}


def run(parts: list[str] | None = None, out: str = RESULTS_DIR) -> dict[str, pd.DataFrame]:
    os.makedirs(out, exist_ok=True)
    warnings.filterwarnings("ignore", category=UserWarning)
    results = {}
    for part in parts or PARTS:
        print(f"== {part}", flush=True)
        t0 = time.perf_counter()
        folds, summary = RUNNERS[part](out)
        folds.to_csv(os.path.join(out, f"model_selection_{part}_folds.csv"), index=False)
        summary.to_csv(os.path.join(out, f"model_selection_{part}.csv"), index=False)
        with pd.option_context("display.width", 250, "display.max_columns", 40):
            print(summary.round(4).to_string(index=False))
        print(f"   ({time.perf_counter() - t0:.0f} s)\n", flush=True)
        results[part] = summary
    return results


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Model-selection comparison for each module.")
    parser.add_argument("--only", help=f"comma-separated parts: {','.join(PARTS)}")
    args = parser.parse_args(argv)
    run(args.only.split(",") if args.only else None)


if __name__ == "__main__":
    main()
