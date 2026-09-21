import numpy as np
import pytest
import torch

from ml.device.anomaly_detector import TelemetryTransformerReconstructor
from ml.device.evaluate_anomaly_detector import (
    calculate_anomaly_threshold,
    classify_anomaly_scores,
    evaluate_anomaly_detector,
)


def create_model():
    torch.manual_seed(42)

    return TelemetryTransformerReconstructor(
        input_dim=5,
        d_model=32,
        nhead=4,
        num_layers=2,
        dim_feedforward=64,
        dropout=0.0,
    )


def create_scores():
    return np.array(
        [0.01, 0.02, 0.03, 0.04, 0.05],
        dtype=np.float64,
    )


def create_sequences(num_sequences=5, window_size=6):
    rng = np.random.default_rng(42)

    return rng.normal(
        0.0,
        1.0,
        size=(num_sequences, window_size, 5),
    ).astype(np.float32)


def test_calculate_threshold():
    scores = create_scores()

    threshold = calculate_anomaly_threshold(
        scores,
        percentile=80.0,
    )

    assert np.isfinite(threshold)
    assert threshold >= 0.0
    assert threshold == pytest.approx(0.042)


def test_threshold_uses_requested_percentile():
    scores = np.array(
        [1.0, 2.0, 3.0, 4.0],
        dtype=np.float64,
    )

    threshold = calculate_anomaly_threshold(
        scores,
        percentile=50.0,
    )

    assert threshold == pytest.approx(2.5)


def test_threshold_rejects_non_1d_scores():
    scores = np.zeros((2, 2), dtype=np.float64)

    with pytest.raises(ValueError):
        calculate_anomaly_threshold(scores)


def test_threshold_rejects_empty_scores():
    scores = np.array([], dtype=np.float64)

    with pytest.raises(ValueError):
        calculate_anomaly_threshold(scores)


def test_threshold_rejects_nan():
    scores = np.array([0.1, np.nan, 0.2])

    with pytest.raises(ValueError):
        calculate_anomaly_threshold(scores)


def test_threshold_rejects_infinity():
    scores = np.array([0.1, np.inf, 0.2])

    with pytest.raises(ValueError):
        calculate_anomaly_threshold(scores)


def test_threshold_rejects_negative_scores():
    scores = np.array([0.1, -0.2, 0.3])

    with pytest.raises(ValueError):
        calculate_anomaly_threshold(scores)


def test_threshold_rejects_invalid_percentile():
    scores = create_scores()

    with pytest.raises(ValueError):
        calculate_anomaly_threshold(scores, percentile=0.0)

    with pytest.raises(ValueError):
        calculate_anomaly_threshold(scores, percentile=100.0)


def test_classify_anomaly_scores():
    scores = np.array(
        [0.1, 0.5, 0.9],
        dtype=np.float64,
    )

    predictions = classify_anomaly_scores(
        scores,
        threshold=0.5,
    )

    np.testing.assert_array_equal(
        predictions,
        np.array([False, False, True]),
    )


def test_classification_returns_boolean_array():
    scores = create_scores()

    predictions = classify_anomaly_scores(
        scores,
        threshold=0.03,
    )

    assert isinstance(predictions, np.ndarray)
    assert predictions.dtype == bool
    assert predictions.shape == scores.shape


def test_classification_rejects_invalid_scores():
    with pytest.raises(ValueError):
        classify_anomaly_scores(
            np.array([0.1, np.nan]),
            threshold=0.2,
        )


def test_classification_rejects_negative_scores():
    with pytest.raises(ValueError):
        classify_anomaly_scores(
            np.array([0.1, -0.1]),
            threshold=0.2,
        )


def test_classification_rejects_invalid_threshold():
    scores = create_scores()

    with pytest.raises(ValueError):
        classify_anomaly_scores(scores, threshold=-1.0)

    with pytest.raises(ValueError):
        classify_anomaly_scores(scores, threshold=np.inf)


def test_full_evaluation_returns_expected_fields():
    model = create_model()

    normal = create_sequences(5)
    anomalous = create_sequences(3)

    result = evaluate_anomaly_detector(
        model,
        normal,
        anomalous,
    )

    assert "threshold" in result
    assert "normal_scores" in result
    assert "anomaly_scores" in result
    assert "anomaly_predictions" in result
    assert "normal_count" in result
    assert "anomaly_count" in result
    assert "detected_anomaly_count" in result


def test_full_evaluation_counts_are_correct():
    model = create_model()

    normal = create_sequences(5)
    anomalous = create_sequences(3)

    result = evaluate_anomaly_detector(
        model,
        normal,
        anomalous,
    )

    assert result["normal_count"] == 5
    assert result["anomaly_count"] == 3

    predictions = result["anomaly_predictions"]

    assert len(predictions) == 3
    assert result["detected_anomaly_count"] == int(np.sum(predictions))


def test_full_evaluation_scores_are_finite():
    model = create_model()

    normal = create_sequences(5)
    anomalous = create_sequences(3)

    result = evaluate_anomaly_detector(
        model,
        normal,
        anomalous,
    )

    assert np.isfinite(result["threshold"])
    assert np.isfinite(result["normal_scores"]).all()
    assert np.isfinite(result["anomaly_scores"]).all()


def test_full_evaluation_predictions_are_boolean():
    model = create_model()

    normal = create_sequences(5)
    anomalous = create_sequences(3)

    result = evaluate_anomaly_detector(
        model,
        normal,
        anomalous,
    )

    assert result["anomaly_predictions"].dtype == bool


def test_custom_percentile_is_used():
    model = create_model()

    normal = create_sequences(5)
    anomalous = create_sequences(3)

    result = evaluate_anomaly_detector(
        model,
        normal,
        anomalous,
        percentile=90.0,
    )

    expected = np.percentile(
        result["normal_scores"],
        90.0,
    )

    assert result["threshold"] == pytest.approx(expected)