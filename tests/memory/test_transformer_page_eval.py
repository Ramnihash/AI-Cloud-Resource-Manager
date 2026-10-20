"""Smoke test for the separate Transformer page-predictor study (needs PyTorch)."""

import importlib.util
import tempfile
import unittest
from pathlib import Path


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is not installed")
class TestTransformerPageEval(unittest.TestCase):
    def test_writes_outputs_and_sequential_is_learnable(self):
        from experiments import transformer_page_eval as t

        with tempfile.TemporaryDirectory() as out:
            r = t.run(out=out, n_strings=1, n_request=3)
            for name in ("memory_transformer_accuracy.csv", "memory_transformer_replacement.csv",
                         "memory_transformer_paired.csv", "memory_transformer_request.csv",
                         "memory_transformer_accuracy.png"):
                self.assertTrue((Path(out) / name).exists(), name)
        acc = r["accuracy"]
        self.assertEqual(set(acc["predictor"]), set(t.PREDICTORS))
        trained = acc[(acc.pattern == "sequential") & (acc.predictor == "Transformer (trained)")]
        self.assertGreater(trained["top1_accuracy"].iloc[0], 0.9)
