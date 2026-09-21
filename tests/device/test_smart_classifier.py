"""Tests for CP3.8 SMART disk failure-risk classifier configuration."""

import unittest

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from ml.device.smart.classifier import (
    ClassifierConfig,
    create_logistic_regression,
    create_random_forest,
)


class TestSmartClassifier(unittest.TestCase):
    def test_default_config(self):
        config = ClassifierConfig()

        self.assertEqual(config.random_state, 42)
        self.assertEqual(config.test_size, 0.2)
        self.assertEqual(config.logistic_max_iter, 1000)
        self.assertEqual(config.random_forest_estimators, 100)
        self.assertEqual(config.class_weight, "balanced")

    def test_create_logistic_regression(self):
        config = ClassifierConfig()

        model = create_logistic_regression(config)

        self.assertIsInstance(model, LogisticRegression)
        self.assertEqual(model.max_iter, 1000)
        self.assertEqual(model.random_state, 42)
        self.assertEqual(model.class_weight, "balanced")

    def test_create_random_forest(self):
        config = ClassifierConfig()

        model = create_random_forest(config)

        self.assertIsInstance(model, RandomForestClassifier)
        self.assertEqual(model.n_estimators, 100)
        self.assertEqual(model.random_state, 42)
        self.assertEqual(model.n_jobs, -1)
        self.assertEqual(model.class_weight, "balanced")


if __name__ == "__main__":
    unittest.main()