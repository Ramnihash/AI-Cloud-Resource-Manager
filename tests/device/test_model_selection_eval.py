import numpy as np
import pandas as pd

from experiments import model_selection_eval as ms


def test_paired_marks_reference_and_direction():
    rows = []
    for fold in range(5):
        rows.append({"model": "ref", "seed": 1, "fold": fold, "mae": 1.0 + 0.01 * fold})
        rows.append({"model": "worse", "seed": 1, "fold": fold, "mae": 2.0 + 0.02 * fold})
    out = ms._paired(pd.DataFrame(rows), "mae", "ref", higher_is_better=False).set_index("model")
    assert out.loc["ref", "verdict"] == "in use"
    assert out.loc["worse", "verdict"] == "worse"
    assert out.loc["worse", "p_value"] < 0.05


def test_device_part_compares_detectors(tmp_path):
    detail, summary = ms.run_device(str(tmp_path))
    models = set(summary["model"])
    assert {"isolation_forest", "one_class_svm", "pca_reconstruction", "z_score_max"} <= models
    clean = detail[(detail.condition == "clean") & (detail.model == "pca_reconstruction")].iloc[0]
    assert clean["recall"] == 1.0 and clean["false_positive_rate"] == 0.0
    assert np.isfinite(summary["recall_severity 0.05"]).all()
