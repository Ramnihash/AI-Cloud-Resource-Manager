"""Tests for CP3.7 Transformer reconstruction model (anomaly_detector.py)."""

import unittest

import torch

from ml.device.anomaly_detector import TelemetryTransformerReconstructor


def _make_model(**kwargs) -> TelemetryTransformerReconstructor:
    """Instantiate model with default (or overridden) parameters."""
    return TelemetryTransformerReconstructor(**kwargs)


def _seeded_model(**kwargs) -> TelemetryTransformerReconstructor:
    """Return a model instantiated with a fixed random seed."""
    torch.manual_seed(42)
    return TelemetryTransformerReconstructor(**kwargs)


class TestInstantiation(unittest.TestCase):
    def test_default_parameters(self):
        model = _make_model()
        self.assertEqual(model.input_dim, 5)
        self.assertEqual(model.d_model, 32)

    def test_custom_parameters(self):
        model = _make_model(d_model=64, nhead=8, num_layers=3, dim_feedforward=128)
        self.assertEqual(model.d_model, 64)

    def test_invalid_d_model_not_divisible_by_nhead(self):
        with self.assertRaises(ValueError):
            _make_model(d_model=32, nhead=7)

    def test_invalid_num_layers(self):
        with self.assertRaises(ValueError):
            _make_model(num_layers=0)

    def test_invalid_input_dim(self):
        with self.assertRaises(ValueError):
            _make_model(input_dim=0)

    def test_invalid_dropout(self):
        with self.assertRaises(ValueError):
            _make_model(dropout=1.5)


class TestOutputShape(unittest.TestCase):
    def setUp(self):
        self.model = _make_model()
        self.model.eval()

    def test_shape_batch2_seq8_feat5(self):
        x = torch.randn(2, 8, 5)
        with torch.no_grad():
            out = self.model(x)
        self.assertEqual(out.shape, (2, 8, 5))

    def test_shape_batch4_seq12_feat5(self):
        x = torch.randn(4, 12, 5)
        with torch.no_grad():
            out = self.model(x)
        self.assertEqual(out.shape, (4, 12, 5))

    def test_different_sequence_lengths_no_model_change(self):
        """Model must handle variable sequence lengths without re-instantiation."""
        model = _make_model()
        model.eval()
        for seq_len in (4, 8, 16, 32):
            x = torch.randn(1, seq_len, 5)
            with torch.no_grad():
                out = model(x)
            self.assertEqual(out.shape, (1, seq_len, 5))

    def test_batch_size_one(self):
        x = torch.randn(1, 6, 5)
        with torch.no_grad():
            out = self.model(x)
        self.assertEqual(out.shape, (1, 6, 5))


class TestInputValidation(unittest.TestCase):
    def setUp(self):
        self.model = _make_model()

    def test_2d_input_raises(self):
        x = torch.randn(8, 5)
        with self.assertRaises(ValueError):
            self.model(x)

    def test_4d_input_raises(self):
        x = torch.randn(2, 8, 5, 1)
        with self.assertRaises(ValueError):
            self.model(x)

    def test_wrong_feature_dim_raises(self):

        x = torch.randn(2, 8, 7)
        with self.assertRaises(ValueError):
            self.model(x)

    def test_wrong_feature_dim_too_few_raises(self):
        x = torch.randn(2, 8, 3)
        with self.assertRaises(ValueError):
            self.model(x)


class TestForwardBehavior(unittest.TestCase):
    def setUp(self):
        self.model = _make_model()
        self.model.eval()

    def test_input_not_modified_in_place(self):
        x = torch.randn(2, 8, 5)
        x_clone = x.clone()
        with torch.no_grad():
            _ = self.model(x)
        self.assertTrue(torch.equal(x, x_clone), "Forward pass modified input tensor in-place")

    def test_output_is_finite(self):
        x = torch.randn(2, 8, 5)
        with torch.no_grad():
            out = self.model(x)
        self.assertTrue(torch.isfinite(out).all(), "Output contains non-finite values")

    def test_model_runs_on_cpu(self):
        model = _make_model()
        model.eval()
        x = torch.randn(2, 8, 5)
        with torch.no_grad():
            out = model(x)
        self.assertEqual(out.device.type, "cpu")

    def test_output_dtype_matches_input(self):
        x = torch.randn(2, 8, 5, dtype=torch.float32)
        with torch.no_grad():
            out = self.model(x)
        self.assertEqual(out.dtype, x.dtype)


class TestDeterminism(unittest.TestCase):
    def test_eval_mode_is_deterministic(self):
        """Same seed + eval mode must produce identical outputs."""
        x = torch.randn(2, 8, 5)

        model_a = _seeded_model()
        model_a.eval()
        with torch.no_grad():
            out_a = model_a(x)

        model_b = _seeded_model()
        model_b.eval()
        with torch.no_grad():
            out_b = model_b(x)

        self.assertTrue(
            torch.allclose(out_a, out_b),
            "Models with identical seeds differ in eval mode",
        )


if __name__ == "__main__":
    unittest.main()
