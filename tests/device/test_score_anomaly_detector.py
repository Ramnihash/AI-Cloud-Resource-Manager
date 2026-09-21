import numpy as np
import pytest
import torch

from ml.device.anomaly_detector import TelemetryTransformerReconstructor
from ml.device.score_anomaly_detector import score_anomaly_sequences


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


def create_sequences(num_sequences=3, window_size=6):
    rng = np.random.default_rng(42)

    return rng.normal(
        loc=0.0,
        scale=1.0,
        size=(num_sequences, window_size, 5),
    ).astype(np.float32)


def test_batch_numpy_input():
    model = create_model()
    sequences = create_sequences()

    scores = score_anomaly_sequences(model, sequences)

    assert isinstance(scores, np.ndarray)
    assert scores.shape == (3,)


def test_single_numpy_sequence():
    model = create_model()
    sequences = create_sequences()

    scores = score_anomaly_sequences(model, sequences[0])

    assert isinstance(scores, np.ndarray)
    assert scores.shape == (1,)


def test_batch_torch_input():
    model = create_model()
    sequences = torch.from_numpy(create_sequences())

    scores = score_anomaly_sequences(model, sequences)

    assert isinstance(scores, np.ndarray)
    assert scores.shape == (3,)


def test_single_torch_sequence():
    model = create_model()
    sequences = torch.from_numpy(create_sequences())

    scores = score_anomaly_sequences(model, sequences[0])

    assert isinstance(scores, np.ndarray)
    assert scores.shape == (1,)


def test_scores_are_finite():
    model = create_model()
    sequences = create_sequences()

    scores = score_anomaly_sequences(model, sequences)

    assert np.isfinite(scores).all()


def test_scores_are_non_negative():
    model = create_model()
    sequences = create_sequences()

    scores = score_anomaly_sequences(model, sequences)

    assert np.all(scores >= 0.0)


def test_invalid_dimension_is_rejected():
    model = create_model()

    invalid = np.zeros((2, 3, 4, 5), dtype=np.float32)

    with pytest.raises(ValueError):
        score_anomaly_sequences(model, invalid)


def test_wrong_feature_count_is_rejected():
    model = create_model()

    invalid = np.zeros((3, 6, 4), dtype=np.float32)

    with pytest.raises(ValueError):
        score_anomaly_sequences(model, invalid)


def test_nan_is_rejected():
    model = create_model()
    sequences = create_sequences()
    sequences[0, 0, 0] = np.nan

    with pytest.raises(ValueError):
        score_anomaly_sequences(model, sequences)


def test_infinity_is_rejected():
    model = create_model()
    sequences = create_sequences()
    sequences[0, 0, 0] = np.inf

    with pytest.raises(ValueError):
        score_anomaly_sequences(model, sequences)


def test_empty_input_is_rejected():
    model = create_model()

    invalid = np.empty((0, 6, 5), dtype=np.float32)

    with pytest.raises(ValueError):
        score_anomaly_sequences(model, invalid)


def test_input_is_not_modified():
    model = create_model()
    sequences = create_sequences()
    original = sequences.copy()

    score_anomaly_sequences(model, sequences)

    np.testing.assert_array_equal(sequences, original)


def test_no_gradients_are_created():
    model = create_model()
    sequences = create_sequences()

    scores = score_anomaly_sequences(model, sequences)

    assert scores.shape == (3,)

    for parameter in model.parameters():
        assert parameter.grad is None


def test_model_evaluation_state_is_restored():
    model = create_model()
    sequences = create_sequences()

    model.train()
    assert model.training is True

    score_anomaly_sequences(model, sequences)

    assert model.training is True


def test_model_evaluation_mode_remains_evaluation():
    model = create_model()
    sequences = create_sequences()

    model.eval()
    assert model.training is False

    score_anomaly_sequences(model, sequences)

    assert model.training is False


def test_each_sequence_gets_separate_score():
    model = create_model()
    sequences = create_sequences(num_sequences=4)

    scores = score_anomaly_sequences(model, sequences)

    assert scores.shape == (4,)

    for score in scores:
        assert np.isscalar(score)


def test_non_numeric_numpy_input_is_rejected():
    model = create_model()

    invalid = np.array(
        [[["a", "b", "c", "d", "e"]]],
        dtype=object,
    )

    with pytest.raises(TypeError):
        score_anomaly_sequences(model, invalid)


def test_non_tensor_non_numpy_input_is_rejected():
    model = create_model()

    invalid = [[[0.0] * 5] * 6]

    with pytest.raises(TypeError):
        score_anomaly_sequences(model, invalid)