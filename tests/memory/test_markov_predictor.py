"""
Unit tests for ml.memory.markov_predictor
==========================================

Run from the repository root::

    .venv\\Scripts\\python.exe -m unittest tests.memory.test_markov_predictor -v

Testing strategy:
    All sequences are hand-crafted so results are fully deterministic.
    No randomly generated sequences are used for accuracy assertions.

    Key test sequences:

    CYCLIC   = [0, 1, 2, 0, 1, 2, 0, 1, 2]
        Each transition is deterministic: 0→1, 1→2, 2→0 (all prob=1.0)

    BIFURC   = [0, 1, 0, 2, 0, 1, 0, 2]
        From page 0: equal split to page 1 and page 2 (each prob=0.5)

    ALTERN   = [0, 1, 0, 1, 0, 1]
        0→1 prob=1.0, 1→0 prob=1.0
"""

import unittest

from ml.memory.markov_predictor import MarkovPagePredictor



CYCLIC = [0, 1, 2, 0, 1, 2, 0, 1, 2]
BIFURC = [0, 1, 0, 2, 0, 1, 0, 2]
ALTERN = [0, 1, 0, 1, 0, 1]


def _trained_on(sequence):
    """Return a MarkovPagePredictor already trained on *sequence*."""
    p = MarkovPagePredictor()
    p.train(sequence)
    return p



class TestTrain(unittest.TestCase):
    """Verify that train() builds the correct transition table."""

    def setUp(self):
        self.predictor = MarkovPagePredictor()

    def test_train_returns_dict(self):
        result = self.predictor.train(CYCLIC)
        self.assertIsInstance(result, dict)

    def test_training_length_preserved(self):
        result = self.predictor.train(CYCLIC)
        self.assertEqual(result["training_length"], len(CYCLIC))

    def test_order_is_one(self):
        result = self.predictor.train(CYCLIC)
        self.assertEqual(result["order"], 1)

    def test_num_pages_seen(self):
        """num_pages_seen must equal the number of distinct page IDs."""
        result = self.predictor.train(CYCLIC)
        self.assertEqual(result["num_pages_seen"], len(set(CYCLIC)))

    def test_num_states(self):
        """num_states must equal the number of distinct from-states."""
        result = self.predictor.train(CYCLIC)

        self.assertEqual(result["num_states"], 3)

    def test_transition_table_in_result(self):
        result = self.predictor.train(CYCLIC)
        self.assertIn("transition_table", result)
        self.assertIsInstance(result["transition_table"], dict)

    def test_transition_table_on_instance(self):
        """transition_table must also be set on the instance after train()."""
        self.predictor.train(CYCLIC)
        self.assertIsInstance(self.predictor.transition_table, dict)
        self.assertGreater(len(self.predictor.transition_table), 0)

    def test_probabilities_sum_to_one_per_state(self):
        """Each row in the transition table must sum to 1.0."""
        self.predictor.train(CYCLIC)
        for from_page, to_probs in self.predictor.transition_table.items():
            total = sum(to_probs.values())
            self.assertAlmostEqual(
                total, 1.0, places=10,
                msg=f"Probabilities for from_page={from_page} sum to {total}"
            )

    def test_train_multiple_times_resets_state(self):
        """Re-training on a different sequence replaces the old model."""
        self.predictor.train(ALTERN)
        self.predictor.train(CYCLIC)

        result = self.predictor.transition_table
        self.assertIn(0, result)
        self.assertAlmostEqual(result[0][1], 1.0)



class TestKnownSequence(unittest.TestCase):
    """Verify exact transition probabilities on hand-crafted sequences."""

    def test_cyclic_p_0_to_1_is_one(self):
        p = _trained_on(CYCLIC)
        self.assertAlmostEqual(p.transition_table[0][1], 1.0)

    def test_cyclic_p_1_to_2_is_one(self):
        p = _trained_on(CYCLIC)
        self.assertAlmostEqual(p.transition_table[1][2], 1.0)

    def test_cyclic_p_2_to_0_is_one(self):
        p = _trained_on(CYCLIC)
        self.assertAlmostEqual(p.transition_table[2][0], 1.0)

    def test_bifurc_equal_split_from_page_0(self):
        """[0,1,0,2,0,1,0,2] → P(0→1) = P(0→2) = 0.5."""
        p = _trained_on(BIFURC)
        self.assertAlmostEqual(p.transition_table[0][1], 0.5)
        self.assertAlmostEqual(p.transition_table[0][2], 0.5)

    def test_bifurc_page_1_to_0_is_one(self):
        p = _trained_on(BIFURC)
        self.assertAlmostEqual(p.transition_table[1][0], 1.0)

    def test_minimal_two_element_sequence(self):
        """A two-element sequence forms exactly one transition."""
        p = MarkovPagePredictor()
        p.train([5, 9])
        self.assertAlmostEqual(p.transition_table[5][9], 1.0)
        self.assertEqual(len(p.transition_table), 1)



class TestPredict(unittest.TestCase):
    """Verify predict() return type, structure, ordering, and bounds."""

    def setUp(self):
        self.p = _trained_on(CYCLIC)

    def test_predict_returns_list(self):
        self.assertIsInstance(self.p.predict(0), list)

    def test_predict_top1_returns_one_entry(self):
        result = self.p.predict(0, top_k=1)
        self.assertEqual(len(result), 1)

    def test_predict_entry_has_page_key(self):
        result = self.p.predict(0)
        self.assertIn("page", result[0])

    def test_predict_entry_has_probability_key(self):
        result = self.p.predict(0)
        self.assertIn("probability", result[0])

    def test_predict_no_extra_keys(self):
        result = self.p.predict(0)
        self.assertEqual(set(result[0].keys()), {"page", "probability"})

    def test_predict_correct_page_cyclic(self):
        """On CYCLIC, predict(0) must return page 1 with probability 1.0."""
        result = self.p.predict(0)
        self.assertEqual(result[0]["page"], 1)
        self.assertAlmostEqual(result[0]["probability"], 1.0)

    def test_predict_top_k_limits_results(self):
        """top_k must cap the returned list length."""
        p = _trained_on(BIFURC)
        result = p.predict(0, top_k=1)
        self.assertEqual(len(result), 1)

    def test_predict_top_k_can_return_all_candidates(self):
        p = _trained_on(BIFURC)
        result = p.predict(0, top_k=10)
        self.assertEqual(len(result), 2)

    def test_predict_sorted_descending(self):
        """Candidates must be sorted by probability, highest first."""

        seq = [0, 1, 0, 1, 0, 1, 0, 2]
        p = _trained_on(seq)
        result = p.predict(0, top_k=2)

        self.assertEqual(result[0]["page"], 1)
        self.assertGreater(result[0]["probability"], result[1]["probability"])

    def test_predict_probabilities_positive(self):
        result = self.p.predict(0, top_k=3)
        for entry in result:
            self.assertGreater(entry["probability"], 0.0)

    def test_predict_probabilities_at_most_one(self):
        result = self.p.predict(0, top_k=3)
        for entry in result:
            self.assertLessEqual(entry["probability"], 1.0)



class TestUnseenState(unittest.TestCase):

    def setUp(self):
        self.p = _trained_on(CYCLIC)

    def test_unseen_page_returns_empty_list(self):
        """Page never seen during training must return [] without error."""
        result = self.p.predict(99)
        self.assertEqual(result, [])

    def test_unseen_page_return_type_is_list(self):
        result = self.p.predict(99)
        self.assertIsInstance(result, list)

    def test_page_only_seen_as_to_state_returns_empty(self):
        """A page that only ever appears as a destination (never as from-state)
        must return [] because it has no outgoing transitions."""



        p = MarkovPagePredictor()
        p.train([0, 9])
        result = p.predict(9)
        self.assertEqual(result, [])



class TestEvaluate(unittest.TestCase):

    def setUp(self):
        self.p = _trained_on(CYCLIC)

    def _eval(self, seq=None, top_k=1):
        seq = seq or CYCLIC
        return self.p.evaluate(seq, top_k=top_k)

    def test_total_predictions_equals_len_minus_one(self):
        result = self._eval()
        self.assertEqual(result["total_predictions"], len(CYCLIC) - 1)

    def test_correct_top1_is_int(self):
        self.assertIsInstance(self._eval()["correct_top1"], int)

    def test_correct_topk_is_int(self):
        self.assertIsInstance(self._eval()["correct_topk"], int)

    def test_top1_accuracy_range(self):
        acc = self._eval()["top1_accuracy"]
        self.assertGreaterEqual(acc, 0.0)
        self.assertLessEqual(acc, 1.0)

    def test_topk_accuracy_range(self):
        acc = self._eval(top_k=2)["topk_accuracy"]
        self.assertGreaterEqual(acc, 0.0)
        self.assertLessEqual(acc, 1.0)

    def test_topk_accuracy_gte_top1_accuracy(self):
        result = self._eval(top_k=2)
        self.assertGreaterEqual(result["topk_accuracy"], result["top1_accuracy"])

    def test_perfect_accuracy_on_trained_sequence(self):
        """Evaluating on the exact training sequence must give accuracy 1.0.

        CYCLIC is deterministic: every transition is unique, so the predictor
        always returns the correct page with probability 1.0.
        """
        result = self._eval()
        self.assertAlmostEqual(result["top1_accuracy"], 1.0)

    def test_top_k_preserved_in_result(self):
        result = self._eval(top_k=3)
        self.assertEqual(result["top_k"], 3)

    def test_correct_top1_lte_total_predictions(self):
        result = self._eval()
        self.assertLessEqual(result["correct_top1"], result["total_predictions"])

    def test_correct_topk_lte_total_predictions(self):
        result = self._eval(top_k=2)
        self.assertLessEqual(result["correct_topk"], result["total_predictions"])

    def test_evaluate_on_different_test_seq(self):
        """Predictor trained on CYCLIC evaluated on ALTERN.

        Only overlapping transitions (0→1) will be correct; 1→0 is not in the
        CYCLIC model, so those are unseen.  We just verify structural correctness.
        """
        result = self.p.evaluate(ALTERN, top_k=1)
        self.assertEqual(result["total_predictions"], len(ALTERN) - 1)
        self.assertIn("top1_accuracy", result)



class TestNotTrained(unittest.TestCase):

    def setUp(self):
        self.p = MarkovPagePredictor()

    def test_predict_before_train_raises_runtime_error(self):
        with self.assertRaises(RuntimeError):
            self.p.predict(0)

    def test_evaluate_before_train_raises_runtime_error(self):
        with self.assertRaises(RuntimeError):
            self.p.evaluate([0, 1, 2])



class TestInvalidInputs(unittest.TestCase):

    def setUp(self):
        self.p = _trained_on(CYCLIC)


    def test_train_non_list_raises_type_error(self):
        fresh = MarkovPagePredictor()
        with self.assertRaises(TypeError):
            fresh.train((0, 1, 2, 3))

    def test_train_string_raises_type_error(self):
        fresh = MarkovPagePredictor()
        with self.assertRaises(TypeError):
            fresh.train("012")


    def test_train_empty_list_raises_value_error(self):
        fresh = MarkovPagePredictor()
        with self.assertRaises(ValueError):
            fresh.train([])

    def test_train_single_element_raises_value_error(self):
        fresh = MarkovPagePredictor()
        with self.assertRaises(ValueError):
            fresh.train([7])


    def test_predict_top_k_zero_raises_value_error(self):
        with self.assertRaises(ValueError):
            self.p.predict(0, top_k=0)

    def test_predict_top_k_negative_raises_value_error(self):
        with self.assertRaises(ValueError):
            self.p.predict(0, top_k=-1)

    def test_predict_top_k_float_raises_value_error(self):
        with self.assertRaises(ValueError):
            self.p.predict(0, top_k=1.5)

    def test_predict_top_k_string_raises_value_error(self):
        with self.assertRaises(ValueError):
            self.p.predict(0, top_k="1")


    def test_evaluate_non_list_raises_type_error(self):
        with self.assertRaises(TypeError):
            self.p.evaluate((0, 1, 2))


    def test_evaluate_single_element_raises_value_error(self):
        with self.assertRaises(ValueError):
            self.p.evaluate([0])

    def test_evaluate_empty_raises_value_error(self):
        with self.assertRaises(ValueError):
            self.p.evaluate([])


    def test_evaluate_top_k_zero_raises_value_error(self):
        with self.assertRaises(ValueError):
            self.p.evaluate(CYCLIC, top_k=0)

    def test_evaluate_top_k_float_raises_value_error(self):
        with self.assertRaises(ValueError):
            self.p.evaluate(CYCLIC, top_k=2.0)




EXPECTED_TRAIN_KEYS = {
    "order", "num_pages_seen", "num_states", "training_length", "transition_table"
}
EXPECTED_EVAL_KEYS = {
    "total_predictions", "correct_top1", "correct_topk",
    "top1_accuracy", "topk_accuracy", "top_k"
}


class TestResultStructure(unittest.TestCase):

    def setUp(self):
        self.p = MarkovPagePredictor()
        self.train_result = self.p.train(CYCLIC)
        self.eval_result = self.p.evaluate(CYCLIC, top_k=2)


    def test_train_result_has_all_required_keys(self):
        missing = EXPECTED_TRAIN_KEYS - self.train_result.keys()
        self.assertEqual(missing, set(), msg=f"Missing train keys: {missing}")

    def test_train_result_no_extra_keys(self):
        extra = self.train_result.keys() - EXPECTED_TRAIN_KEYS
        self.assertEqual(extra, set(), msg=f"Unexpected train keys: {extra}")

    def test_train_result_order_is_int_one(self):
        self.assertIsInstance(self.train_result["order"], int)
        self.assertEqual(self.train_result["order"], 1)

    def test_train_result_training_length_is_int(self):
        self.assertIsInstance(self.train_result["training_length"], int)

    def test_train_result_num_pages_seen_is_int(self):
        self.assertIsInstance(self.train_result["num_pages_seen"], int)

    def test_train_result_num_states_is_int(self):
        self.assertIsInstance(self.train_result["num_states"], int)


    def test_evaluate_result_has_all_required_keys(self):
        missing = EXPECTED_EVAL_KEYS - self.eval_result.keys()
        self.assertEqual(missing, set(), msg=f"Missing eval keys: {missing}")

    def test_evaluate_result_no_extra_keys(self):
        extra = self.eval_result.keys() - EXPECTED_EVAL_KEYS
        self.assertEqual(extra, set(), msg=f"Unexpected eval keys: {extra}")

    def test_evaluate_top1_accuracy_is_float(self):
        self.assertIsInstance(self.eval_result["top1_accuracy"], float)

    def test_evaluate_topk_accuracy_is_float(self):
        self.assertIsInstance(self.eval_result["topk_accuracy"], float)


    def test_predict_entry_structure(self):
        result = self.p.predict(0, top_k=1)
        self.assertEqual(len(result), 1)
        entry = result[0]
        self.assertIsInstance(entry["page"], int)
        self.assertIsInstance(entry["probability"], float)


if __name__ == "__main__":
    unittest.main()
