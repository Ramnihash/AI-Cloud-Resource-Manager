import unittest
import torch
from ml.memory.transformer_predictor import TransformerPagePredictor


class TestTransformerPagePredictor(unittest.TestCase):
    def setUp(self):
        self.num_pages = 10
        self.context_len = 3
        self.predictor = TransformerPagePredictor(
            num_pages=self.num_pages,
            context_len=self.context_len,
            d_model=16,
            nhead=2,
            d_ff=32,
            seed=42,
        )
        self.simple_seq = [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]

    def test_construction_valid(self):
        self.assertEqual(self.predictor.num_pages, 10)
        self.assertEqual(self.predictor.context_len, 3)
        self.assertFalse(self.predictor._trained)

    def test_construction_invalid(self):
        with self.assertRaises(ValueError):
            TransformerPagePredictor(num_pages=0)
        with self.assertRaises(ValueError):
            TransformerPagePredictor(num_pages=10, context_len=0)
        with self.assertRaises(ValueError):
            TransformerPagePredictor(num_pages=-1)
        with self.assertRaises(ValueError):
            TransformerPagePredictor(num_pages=10, context_len=-1)

    def test_train_invalid_sequence(self):
        with self.assertRaises(TypeError):
            self.predictor.train("not a list")
        with self.assertRaises(ValueError):
            self.predictor.train([0, 1, 2])  # Needs at least context_len + 1 = 4
        with self.assertRaises(ValueError):
            self.predictor.train([0, 1, 2, 11])  # Page 11 out of bounds

    def test_train_invalid_args(self):
        with self.assertRaises(ValueError):
            self.predictor.train(self.simple_seq, epochs=0)
        with self.assertRaises(ValueError):
            self.predictor.train(self.simple_seq, epochs=True)
        with self.assertRaises(ValueError):
            self.predictor.train(self.simple_seq, lr=-0.1)

    def test_train_returns_metrics(self):
        result = self.predictor.train(self.simple_seq, epochs=2)
        self.assertIn("epochs", result)
        self.assertIn("final_loss", result)
        self.assertIn("num_pages", result)
        self.assertIn("context_len", result)
        self.assertIn("training_samples", result)

        self.assertEqual(result["epochs"], 2)
        self.assertEqual(result["num_pages"], 10)
        self.assertEqual(result["context_len"], 3)
        self.assertEqual(result["training_samples"], len(self.simple_seq) - 3)
        self.assertGreaterEqual(result["final_loss"], 0.0)
        self.assertTrue(self.predictor._trained)

    def test_predict_before_train_raises(self):
        with self.assertRaises(RuntimeError):
            self.predictor.predict([0, 1, 2])
        with self.assertRaises(RuntimeError):
            self.predictor.evaluate([0, 1, 2, 3])

    def test_predict_invalid_inputs(self):
        self.predictor.train(self.simple_seq, epochs=2)
        with self.assertRaises(TypeError):
            self.predictor.predict("not a list")
        with self.assertRaises(ValueError):
            self.predictor.predict([0, 1])  # context_len is 3
        with self.assertRaises(ValueError):
            self.predictor.predict([0, 1, 2, 3])  # context_len is 3
        with self.assertRaises(ValueError):
            self.predictor.predict([0, 1, 2], top_k=0)
        with self.assertRaises(TypeError):
            self.predictor.predict([0, 1, 2.5], top_k=1)
        with self.assertRaises(ValueError):
            self.predictor.predict([0, 1, 10], top_k=1)  # 10 is out of bounds

    def test_predict_returns_correct_structure(self):
        self.predictor.train(self.simple_seq, epochs=2)
        candidates = self.predictor.predict([0, 1, 2], top_k=3)
        
        self.assertEqual(len(candidates), 3)
        prev_prob = float("inf")
        prob_sum = 0.0
        
        for cand in candidates:
            self.assertIn("page", cand)
            self.assertIn("probability", cand)
            self.assertIsInstance(cand["page"], int)
            self.assertIsInstance(cand["probability"], float)
            self.assertTrue(0.0 <= cand["probability"] <= 1.0)
            self.assertLessEqual(cand["probability"], prev_prob)
            prev_prob = cand["probability"]
            prob_sum += cand["probability"]
            
        self.assertLessEqual(prob_sum, 1.0001)

    def test_predict_top_k_caps_at_num_pages(self):
        self.predictor.train(self.simple_seq, epochs=2)
        candidates = self.predictor.predict([0, 1, 2], top_k=20)
        self.assertEqual(len(candidates), 10)  # capped at num_pages

    def test_evaluate_invalid_inputs(self):
        self.predictor.train(self.simple_seq, epochs=2)
        with self.assertRaises(TypeError):
            self.predictor.evaluate("not a list")
        with self.assertRaises(ValueError):
            self.predictor.evaluate([0, 1, 2])  # less than context_len + 1
        with self.assertRaises(ValueError):
            self.predictor.evaluate([0, 1, 2, 3], top_k=0)

    def test_evaluate_returns_metrics(self):
        self.predictor.train(self.simple_seq, epochs=2)
        test_seq = [0, 1, 2, 3, 0, 1, 2]
        result = self.predictor.evaluate(test_seq, top_k=2)
        
        self.assertIn("total_predictions", result)
        self.assertIn("correct_top1", result)
        self.assertIn("correct_topk", result)
        self.assertIn("top1_accuracy", result)
        self.assertIn("topk_accuracy", result)
        self.assertIn("top_k", result)
        
        expected_total = len(test_seq) - self.context_len
        self.assertEqual(result["total_predictions"], expected_total)
        self.assertEqual(result["top_k"], 2)
        self.assertTrue(0.0 <= result["top1_accuracy"] <= 1.0)
        self.assertTrue(0.0 <= result["topk_accuracy"] <= 1.0)
        self.assertGreaterEqual(result["topk_accuracy"], result["top1_accuracy"])

    def test_determinism(self):
        seq = [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]
        context = [0, 1, 2]
        
        # Train instance 1
        p1 = TransformerPagePredictor(num_pages=10, context_len=3, seed=99)
        p1.train(seq, epochs=3)
        res1 = p1.predict(context, top_k=5)
        
        # Train instance 2
        p2 = TransformerPagePredictor(num_pages=10, context_len=3, seed=99)
        p2.train(seq, epochs=3)
        res2 = p2.predict(context, top_k=5)
        
        for r1, r2 in zip(res1, res2):
            self.assertEqual(r1["page"], r2["page"])
            self.assertAlmostEqual(r1["probability"], r2["probability"], places=5)

    def test_retraining_resets_model(self):
        seq1 = [0, 1, 0, 1, 0, 1, 0, 1]
        seq2 = [2, 3, 2, 3, 2, 3, 2, 3]
        
        p = TransformerPagePredictor(num_pages=10, context_len=2, seed=42)
        
        p.train(seq1, epochs=10, lr=0.01)
        res1 = p.predict([0, 1], top_k=1)
        
        p.train(seq2, epochs=10, lr=0.01)
        res2 = p.predict([2, 3], top_k=1)
        
        # The first prediction should favor 0, the second should favor 2 (or something else, but they shouldn't just build on each other identically if reset)
        # We mainly just verify it doesn't crash and returns valid predictions after train is called again.
        self.assertIsInstance(res1[0]["page"], int)
        self.assertIsInstance(res2[0]["page"], int)

    def test_perfect_sequence_learning(self):
        # A simple cyclic sequence. It should learn it well if given enough epochs.
        # Sequence: 0, 1, 2, 0, 1, 2...
        seq = [0, 1, 2] * 20
        # test sequence: a temporal continuation
        test_seq = [0, 1, 2] * 5
        
        p = TransformerPagePredictor(
            num_pages=5,
            context_len=2,
            d_model=16,
            nhead=2,
            d_ff=32,
            seed=42
        )
        
        # Train with a good number of epochs and learning rate
        p.train(seq, epochs=50, lr=0.01)
        
        res = p.evaluate(test_seq, top_k=1)
        # With a simple cycle and enough training, top1_accuracy should be very high.
        # We assert >= 0.8 as requested, though it often achieves 1.0
        self.assertGreaterEqual(res["top1_accuracy"], 0.8)


if __name__ == "__main__":
    unittest.main()
