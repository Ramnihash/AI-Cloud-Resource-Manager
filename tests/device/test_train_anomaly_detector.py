"""Tests for CP3.7 training utility (train_anomaly_detector.py).

All tests use tiny synthetic NORMAL sequences so the suite remains fast.
No test checks for a specific accuracy or exact loss value; only shape,
type, finiteness, and parameter-change properties are asserted.
"""

from __future__ import annotations

import math
import time
import unittest

import numpy as np
import torch

from ml.device.anomaly_detector import TelemetryTransformerReconstructor
from ml.device.train_anomaly_detector import train_anomaly_detector





_N_SEQ = 20
_WIN = 8
_FEAT = 5


def _normal_numpy(n: int = _N_SEQ, win: int = _WIN) -> np.ndarray:
    """Return a deterministic NumPy array of shape (n, win, 5)."""
    rng = np.random.default_rng(0)
    return rng.standard_normal((n, win, _FEAT)).astype(np.float32)


def _normal_tensor(n: int = _N_SEQ, win: int = _WIN) -> torch.Tensor:
    """Return a deterministic torch.Tensor of shape (n, win, 5)."""
    torch.manual_seed(0)
    return torch.randn(n, win, _FEAT)


def _tiny_model() -> TelemetryTransformerReconstructor:
    """Minimal-sized model so tests run quickly."""
    torch.manual_seed(42)
    return TelemetryTransformerReconstructor(
        d_model=16, nhead=2, num_layers=1, dim_feedforward=32
    )






class TestValidNumPyInput(unittest.TestCase):
    def test_numpy_sequences_train_successfully(self):
        torch.manual_seed(0)
        data = _normal_numpy()
        model, history = train_anomaly_detector(
            data, model=_tiny_model(), epochs=2, batch_size=8
        )
        self.assertIsNotNone(model)
        self.assertEqual(len(history), 2)






class TestValidTensorInput(unittest.TestCase):
    def test_tensor_sequences_train_successfully(self):
        torch.manual_seed(0)
        data = _normal_tensor()
        model, history = train_anomaly_detector(
            data, model=_tiny_model(), epochs=2, batch_size=8
        )
        self.assertIsNotNone(model)
        self.assertEqual(len(history), 2)






class TestReturnedModelType(unittest.TestCase):
    def test_returned_model_is_correct_type(self):
        torch.manual_seed(0)
        model, _ = train_anomaly_detector(
            _normal_numpy(), model=_tiny_model(), epochs=1, batch_size=8
        )
        self.assertIsInstance(model, TelemetryTransformerReconstructor)

    def test_none_model_creates_default_instance(self):
        torch.manual_seed(0)
        model, _ = train_anomaly_detector(
            _normal_numpy(), model=None, epochs=1, batch_size=8
        )
        self.assertIsInstance(model, TelemetryTransformerReconstructor)






class TestHistoryLength(unittest.TestCase):
    def test_history_length_equals_epochs(self):
        torch.manual_seed(0)
        for n_epochs in (1, 3, 5):
            with self.subTest(epochs=n_epochs):
                _, history = train_anomaly_detector(
                    _normal_numpy(), model=_tiny_model(),
                    epochs=n_epochs, batch_size=8
                )
                self.assertEqual(len(history), n_epochs)






class TestHistoryFiniteness(unittest.TestCase):
    def test_all_history_values_are_finite(self):
        torch.manual_seed(0)
        _, history = train_anomaly_detector(
            _normal_numpy(), model=_tiny_model(), epochs=5, batch_size=8
        )
        for i, loss in enumerate(history):
            with self.subTest(epoch=i):
                self.assertTrue(
                    math.isfinite(loss),
                    f"Epoch {i} loss is not finite: {loss}"
                )






class TestParametersChange(unittest.TestCase):
    def test_training_changes_model_parameters(self):
        torch.manual_seed(42)
        model = _tiny_model()


        params_before = [p.clone().detach() for p in model.parameters()]

        torch.manual_seed(0)
        train_anomaly_detector(
            _normal_numpy(), model=model, epochs=3, batch_size=8
        )

        params_after = list(model.parameters())
        changed = any(
            not torch.equal(before, after.detach())
            for before, after in zip(params_before, params_after)
        )
        self.assertTrue(changed, "No model parameters changed after training")






class TestInvalidShape2D(unittest.TestCase):
    def test_2d_input_raises_value_error(self):
        data_2d = np.ones((_WIN, _FEAT), dtype=np.float32)
        with self.assertRaises(ValueError):
            train_anomaly_detector(data_2d, epochs=1)






class TestInvalidFeatureDim(unittest.TestCase):
    def test_wrong_feature_dim_raises_value_error(self):
        data_wrong = np.ones((_N_SEQ, _WIN, 3), dtype=np.float32)
        with self.assertRaises(ValueError):
            train_anomaly_detector(data_wrong, epochs=1)

    def test_too_many_features_raises_value_error(self):
        data_wrong = np.ones((_N_SEQ, _WIN, 7), dtype=np.float32)
        with self.assertRaises(ValueError):
            train_anomaly_detector(data_wrong, epochs=1)






class TestEmptySequences(unittest.TestCase):
    def test_zero_sequences_raises_value_error(self):
        data_empty = np.ones((0, _WIN, _FEAT), dtype=np.float32)
        with self.assertRaises(ValueError):
            train_anomaly_detector(data_empty, epochs=1)






class TestNonFiniteValues(unittest.TestCase):
    def test_nan_raises_value_error(self):
        data = _normal_numpy()
        data[0, 0, 0] = float("nan")
        with self.assertRaises(ValueError):
            train_anomaly_detector(data, epochs=1)

    def test_inf_raises_value_error(self):
        data = _normal_numpy()
        data[1, 2, 3] = float("inf")
        with self.assertRaises(ValueError):
            train_anomaly_detector(data, epochs=1)

    def test_neg_inf_raises_value_error(self):
        data = _normal_numpy()
        data[2, 3, 4] = float("-inf")
        with self.assertRaises(ValueError):
            train_anomaly_detector(data, epochs=1)






class TestInvalidEpochs(unittest.TestCase):
    def test_zero_epochs_raises(self):
        with self.assertRaises(ValueError):
            train_anomaly_detector(_normal_numpy(), epochs=0)

    def test_negative_epochs_raises(self):
        with self.assertRaises(ValueError):
            train_anomaly_detector(_normal_numpy(), epochs=-5)

    def test_float_epochs_raises(self):
        with self.assertRaises(ValueError):
            train_anomaly_detector(_normal_numpy(), epochs=2.0)

    def test_bool_epochs_raises(self):

        with self.assertRaises(ValueError):
            train_anomaly_detector(_normal_numpy(), epochs=True)






class TestInvalidLearningRate(unittest.TestCase):
    def test_zero_lr_raises(self):
        with self.assertRaises(ValueError):
            train_anomaly_detector(_normal_numpy(), epochs=1, learning_rate=0.0)

    def test_negative_lr_raises(self):
        with self.assertRaises(ValueError):
            train_anomaly_detector(_normal_numpy(), epochs=1, learning_rate=-1e-3)

    def test_bool_lr_raises(self):
        with self.assertRaises(ValueError):
            train_anomaly_detector(_normal_numpy(), epochs=1, learning_rate=True)






class TestInvalidBatchSize(unittest.TestCase):
    def test_zero_batch_size_raises(self):
        with self.assertRaises(ValueError):
            train_anomaly_detector(_normal_numpy(), epochs=1, batch_size=0)

    def test_negative_batch_size_raises(self):
        with self.assertRaises(ValueError):
            train_anomaly_detector(_normal_numpy(), epochs=1, batch_size=-1)

    def test_float_batch_size_raises(self):
        with self.assertRaises(ValueError):
            train_anomaly_detector(_normal_numpy(), epochs=1, batch_size=8.0)

    def test_bool_batch_size_raises(self):
        with self.assertRaises(ValueError):
            train_anomaly_detector(_normal_numpy(), epochs=1, batch_size=True)






class TestCPUTraining(unittest.TestCase):
    def test_trained_model_is_on_cpu(self):
        torch.manual_seed(0)
        model, _ = train_anomaly_detector(
            _normal_numpy(), model=_tiny_model(), epochs=1, batch_size=8
        )
        for param in model.parameters():
            self.assertEqual(
                param.device.type, "cpu",
                f"Parameter {param.shape} is not on CPU"
            )

    def test_cpu_tensor_input_trains(self):
        torch.manual_seed(0)
        data = _normal_tensor().cpu()
        model, history = train_anomaly_detector(
            data, model=_tiny_model(), epochs=1, batch_size=8
        )
        self.assertEqual(len(history), 1)






class TestTrainingSpeed(unittest.TestCase):
    def test_small_run_completes_within_time_limit(self):
        torch.manual_seed(0)
        data = _normal_numpy(n=16, win=8)
        start = time.monotonic()
        train_anomaly_detector(
            data, model=_tiny_model(), epochs=5, batch_size=8
        )
        elapsed = time.monotonic() - start
        self.assertLess(
            elapsed, 30.0,
            f"Training took {elapsed:.1f}s, expected < 30s on CPU"
        )






class TestInputNotModified(unittest.TestCase):
    def test_numpy_input_not_modified(self):
        data = _normal_numpy()
        original = data.copy()
        torch.manual_seed(0)
        train_anomaly_detector(
            data, model=_tiny_model(), epochs=2, batch_size=8
        )
        np.testing.assert_array_equal(
            data, original,
            err_msg="train_anomaly_detector modified the NumPy input array in-place"
        )

    def test_tensor_input_not_modified(self):
        data = _normal_tensor()
        original = data.clone()
        torch.manual_seed(0)
        train_anomaly_detector(
            data, model=_tiny_model(), epochs=2, batch_size=8
        )
        self.assertTrue(
            torch.equal(data, original),
            "train_anomaly_detector modified the torch.Tensor input in-place"
        )


if __name__ == "__main__":
    unittest.main()
