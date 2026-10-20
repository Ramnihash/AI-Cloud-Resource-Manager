"""
Unit tests for FallbackPagePredictor (CP7.4).
"""
import os
import tempfile
import unittest
from pathlib import Path

from ml.memory.fallback_page_predictor import FallbackPagePredictor
from ml.memory.markov_predictor import MarkovPagePredictor
from ml.memory.transformer_predictor import TransformerPagePredictor
from modules.memory.page_prediction_action import PagePredictionActionLayer


class TestFallbackPagePredictor(unittest.TestCase):
    def setUp(self):
        # Sample sequence with clear transitions: 1->2->3->1->2->3
        self.sequence = [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3] * 4
        self.markov = MarkovPagePredictor()
        self.markov.train(self.sequence)

        # Transformer trained on same sequence with context_len=3
        self.transformer = TransformerPagePredictor(
            num_pages=5,
            context_len=3,
            d_model=16,
            nhead=2,
            d_ff=32,
            seed=42,
        )
        self.transformer.train(self.sequence, epochs=10)

    def test_normal_transformer_prediction(self):
        """When Transformer is healthy and input is valid, it handles prediction."""
        predictor = FallbackPagePredictor(
            primary_predictor=self.transformer,
            fallback_predictor=self.markov,
        )
        status = predictor.predict_with_status([1, 2, 3], top_k=2)

        self.assertFalse(status["used_fallback"])
        self.assertEqual(status["active_model"], "transformer")
        self.assertIsNone(status["fallback_reason"])
        self.assertEqual(len(status["candidates"]), 2)
        # Standard predict() API matches
        candidates = predictor.predict([1, 2, 3], top_k=2)
        self.assertEqual(candidates, status["candidates"])

    def test_fallback_when_primary_not_configured(self):
        """When no primary predictor is provided, it falls back to Markov."""
        predictor = FallbackPagePredictor(
            primary_predictor=None,
            fallback_predictor=self.markov,
        )
        status = predictor.predict_with_status([1, 2], top_k=1)

        self.assertTrue(status["used_fallback"])
        self.assertEqual(status["active_model"], "markov")
        self.assertIn("Primary predictor is not configured", status["fallback_reason"])
        self.assertEqual(status["candidates"][0]["page"], 3)  # After 2, expect 3

    def test_fallback_when_transformer_untrained(self):
        """When Transformer raises RuntimeError because it is not trained."""
        untrained_transformer = TransformerPagePredictor(num_pages=5, context_len=3)
        predictor = FallbackPagePredictor(
            primary_predictor=untrained_transformer,
            fallback_predictor=self.markov,
        )
        status = predictor.predict_with_status([2, 3, 1], top_k=1)

        self.assertTrue(status["used_fallback"])
        self.assertEqual(status["active_model"], "markov")
        self.assertEqual(status["error_type"], "RuntimeError")
        self.assertIn("RuntimeError", status["fallback_reason"])
        self.assertEqual(status["candidates"][0]["page"], 2)  # After 1, expect 2

    def test_fallback_on_context_length_mismatch(self):
        """When input context length is shorter than Transformer requires (e.g. cold start)."""
        # Transformer requires context_len=3, but input has only 2 pages
        predictor = FallbackPagePredictor(
            primary_predictor=self.transformer,
            fallback_predictor=self.markov,
        )
        status = predictor.predict_with_status([1, 2], top_k=1)

        self.assertTrue(status["used_fallback"])
        self.assertEqual(status["active_model"], "markov")
        self.assertEqual(status["error_type"], "ValueError")
        self.assertIn("ValueError", status["fallback_reason"])
        # Markov uses context[-1] = 2 -> predicts 3
        self.assertEqual(status["candidates"][0]["page"], 3)

    def test_fallback_on_integer_input(self):
        """When input is a single int rather than list, fallback handles it directly."""
        predictor = FallbackPagePredictor(
            primary_predictor=self.transformer,
            fallback_predictor=self.markov,
        )
        status = predictor.predict_with_status(2, top_k=1)

        self.assertTrue(status["used_fallback"])
        self.assertEqual(status["active_model"], "markov")
        self.assertEqual(status["candidates"][0]["page"], 3)

    def test_fallback_on_simulated_runtime_crash(self):
        """When primary predictor raises an arbitrary runtime error during inference."""
        class CrashingPredictor:
            def predict(self, context, top_k=1):
                raise RuntimeError("Simulated CUDA/Internal matrix calculation failure")

        predictor = FallbackPagePredictor(
            primary_predictor=CrashingPredictor(),
            fallback_predictor=self.markov,
        )
        status = predictor.predict_with_status([1, 2, 3], top_k=1)

        self.assertTrue(status["used_fallback"])
        self.assertEqual(status["active_model"], "markov")
        self.assertEqual(status["error_type"], "RuntimeError")
        self.assertIn("Simulated CUDA/Internal", status["fallback_reason"])
        # Markov uses context[-1] = 3 -> predicts 1
        self.assertEqual(status["candidates"][0]["page"], 1)

    def test_load_checkpoint_missing_file_falls_back(self):
        """When checkpoint weights file is missing, predictor records error and falls back."""
        predictor = FallbackPagePredictor(fallback_predictor=self.markov)
        success = predictor.load_transformer_checkpoint("non_existent_weights_12345.pt")

        self.assertFalse(success)
        self.assertEqual(predictor.last_status["error_type"], "FileNotFoundError")

        # Prediction still succeeds via Markov fallback
        candidates = predictor.predict([1, 2], top_k=1)
        self.assertTrue(len(candidates) > 0)
        self.assertEqual(candidates[0]["page"], 3)

    def test_load_checkpoint_corrupted_file_falls_back(self):
        """When checkpoint weights file is corrupted (invalid binary), it safely fails to load."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as tmp:
            tmp.write(b"NOT_A_VALID_PYTORCH_CHECKPOINT_GARBAGE_BYTES_12345")
            corrupted_path = tmp.name

        try:
            predictor = FallbackPagePredictor(fallback_predictor=self.markov)
            success = predictor.load_transformer_checkpoint(corrupted_path)
            self.assertFalse(success)
            self.assertIn(
                predictor.last_status["error_type"],
                ("UnpicklingError", "RuntimeError", "PytorchStreamReaderError", "Exception"),
            )

            # Prediction still succeeds via Markov fallback
            status = predictor.predict_with_status([2, 3], top_k=1)
            self.assertTrue(status["used_fallback"])
            self.assertEqual(status["candidates"][0]["page"], 1)
        finally:
            if os.path.exists(corrupted_path):
                os.remove(corrupted_path)

    def test_action_layer_integration_with_fallback(self):
        """Verify seamless integration between FallbackPagePredictor and PagePredictionActionLayer."""
        action_layer = PagePredictionActionLayer()
        predictor = FallbackPagePredictor(
            primary_predictor=None,  # Forces Markov fallback
            fallback_predictor=self.markov,
        )

        # Predict next page for context [1, 2] -> Markov predicts page 3
        candidates = predictor.predict([1, 2], top_k=1)
        active_source = predictor.last_status.get("active_model", "Unknown")

        # Case 1: Page 3 not in memory -> PREFETCH
        action_1 = action_layer.evaluate_and_act(
            prediction=candidates,
            memory_state=[1, 2, 4],
            predictor_source=f"Fallback({active_source})",
        )
        self.assertEqual(action_1["action"], "PREFETCH")
        self.assertEqual(action_1["predicted_page"], 3)
        self.assertEqual(action_1["predictor"], "Fallback(markov)")

        # Case 2: Page 3 is in memory -> PROTECT
        action_2 = action_layer.evaluate_and_act(
            prediction=candidates,
            memory_state=[1, 2, 3],
            predictor_source=f"Fallback({active_source})",
        )
        self.assertEqual(action_2["action"], "PROTECT")
        self.assertEqual(action_2["predicted_page"], 3)

    def test_input_validation(self):
        """Verify proper argument validation."""
        predictor = FallbackPagePredictor(fallback_predictor=self.markov)

        with self.assertRaises(ValueError):
            predictor.predict([1, 2], top_k=0)
        with self.assertRaises(ValueError):
            predictor.predict([1, 2], top_k=-1)
        with self.assertRaises(ValueError):
            predictor.predict([1, 2], top_k=True)
        with self.assertRaises(TypeError):
            predictor.predict({"not": "a_list_or_int"}, top_k=1)

        # Empty context returns empty list
        self.assertEqual(predictor.predict([], top_k=1), [])

    def test_clear_history(self):
        """Verify clearing fallback history."""
        predictor = FallbackPagePredictor(fallback_predictor=self.markov)
        predictor.predict([1, 2])
        self.assertEqual(len(predictor.fallback_history), 1)

        predictor.clear_history()
        self.assertEqual(len(predictor.fallback_history), 0)
        self.assertEqual(len(predictor.last_status), 0)


if __name__ == "__main__":
    unittest.main()
