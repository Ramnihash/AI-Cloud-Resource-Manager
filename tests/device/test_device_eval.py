import numpy as np

from experiments import device_eval
from ml.device.anomaly_service import build_window
from modules.device.fleet import create_default_fleet


def _bundle_and_scorer():
    fleet = create_default_fleet()
    normal = np.concatenate([build_window(d, None, 10, i)[0] for d in fleet for i in range(0, 300, 10)])
    bundle = {"mean": normal.mean(axis=0), "std": np.maximum(normal.std(axis=0), 1e-6), "threshold": 3.0}

    def scorer(b, window):
        return float(np.mean(((window - b["mean"]) / b["std"]) ** 2))

    return bundle, scorer


def test_device_eval_writes_outputs(tmp_path):
    bundle, scorer = _bundle_and_scorer()
    results = device_eval.run(bundle, scorer, str(tmp_path))
    names = {p.name for p in tmp_path.iterdir()}
    assert {
        "device_anomaly_scores.csv",
        "device_anomaly_metrics.csv",
        "device_detection_delay.csv",
        "device_actions.csv",
        "device_roc.png",
        "device_severity.csv",
    } <= names
    severity = results["severity"].query("pattern == 'all'").set_index("severity")
    assert severity.loc[1.0, "recall"] >= severity.loc[0.05, "recall"]
    assert 0.0 <= severity.loc[0.05, "recall"] <= 1.0
    overall = results["metrics"].query("condition == 'clean' and pattern == 'all'").iloc[0]
    assert overall["recall"] > 0.9
    assert overall["false_positive_rate"] < 0.1
    assert results["delay"]["detected"].all()
    assert (results["actions"]["backup_success_rate"] == 1.0).all()
