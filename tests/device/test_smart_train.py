"""Tests for CP3.8 SMART classifier training."""

import unittest

import pandas as pd

from ml.device.smart.train import train_classifier


class TestSmartTraining(unittest.TestCase):
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

    def test_logistic_regression_training(self):
        result = train_classifier(
            self.dataframe,
            "logistic_regression",
        )

        self.assertEqual(result.model_name, "logistic_regression")
        self.assertEqual(len(result.predictions), 2)
        self.assertEqual(len(result.probabilities), 2)
        self.assertGreaterEqual(result.precision, 0.0)
        self.assertLessEqual(result.precision, 1.0)
        self.assertGreaterEqual(result.recall, 0.0)
        self.assertLessEqual(result.recall, 1.0)

    def test_random_forest_training(self):
        result = train_classifier(
            self.dataframe,
            "random_forest",
        )

        self.assertEqual(result.model_name, "random_forest")
        self.assertEqual(len(result.predictions), 2)
        self.assertEqual(len(result.probabilities), 2)

    def test_unsupported_model_raises_error(self):
        with self.assertRaises(ValueError):
            train_classifier(
                self.dataframe,
                "unsupported_model",
            )

    def test_single_class_dataset_raises_error(self):
        dataframe = self.dataframe.copy()
        dataframe["failure"] = 0

        with self.assertRaises(ValueError):
            train_classifier(
                dataframe,
                "random_forest",
            )


if __name__ == "__main__":
    unittest.main()