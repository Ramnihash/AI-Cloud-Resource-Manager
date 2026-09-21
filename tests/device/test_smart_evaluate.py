"""Tests for CP3.8 SMART model comparison."""

import unittest

import pandas as pd

from ml.device.smart.evaluate import evaluate_models


class TestSmartEvaluation(unittest.TestCase):
    def setUp(self):
        self.dataframe = pd.DataFrame(
            {
                "smart_5_raw": [
                    10, 11, 12, 13, 14,
                    100, 101, 102, 103, 104,
                ],
                "smart_9_raw": [
                    100, 101, 102, 103, 104,
                    900, 901, 902, 903, 904,
                ],
                "failure": [
                    0, 0, 0, 0, 0,
                    1, 1, 1, 1, 1,
                ],
            }
        )

    def test_evaluate_models_returns_both_models(self):
        results = evaluate_models(self.dataframe)

        self.assertIn("logistic_regression", results)
        self.assertIn("random_forest", results)

    def test_evaluate_models_returns_training_results(self):
        results = evaluate_models(self.dataframe)

        for result in results.values():
            self.assertEqual(len(result.predictions), 2)
            self.assertEqual(len(result.probabilities), 2)
            self.assertGreaterEqual(result.precision, 0.0)
            self.assertLessEqual(result.precision, 1.0)
            self.assertGreaterEqual(result.recall, 0.0)
            self.assertLessEqual(result.recall, 1.0)


if __name__ == "__main__":
    unittest.main()