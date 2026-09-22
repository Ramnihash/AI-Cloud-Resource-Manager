import unittest
from modules.memory.page_prediction_action import PagePredictionActionLayer

class TestPagePredictionActionLayer(unittest.TestCase):
    def setUp(self):
        self.action_layer = PagePredictionActionLayer()

    def test_evaluate_and_act_protect(self):
        prediction = [{"page": 5, "probability": 0.8}]
        memory_state = [1, 2, 5, 8]
        
        result = self.action_layer.evaluate_and_act(prediction, memory_state, predictor_source="Markov")
        
        self.assertEqual(result["action"], "PROTECT")
        self.assertEqual(result["predicted_page"], 5)
        self.assertEqual(result["predictor"], "Markov")
        self.assertEqual(result["memory_state"], [1, 2, 5, 8])
        self.assertEqual(len(self.action_layer.action_log), 1)

    def test_evaluate_and_act_prefetch(self):
        prediction = [{"page": 9, "probability": 0.65}]
        memory_state = [1, 2, 5, 8]
        
        result = self.action_layer.evaluate_and_act(prediction, memory_state, predictor_source="Transformer")
        
        self.assertEqual(result["action"], "PREFETCH")
        self.assertEqual(result["predicted_page"], 9)
        self.assertEqual(result["predictor"], "Transformer")
        self.assertEqual(len(self.action_layer.action_log), 1)

    def test_evaluate_and_act_no_action(self):
        prediction = []
        memory_state = [1, 2]
        
        result = self.action_layer.evaluate_and_act(prediction, memory_state)
        
        self.assertEqual(result["action"], "NO_ACTION")
        self.assertIsNone(result["predicted_page"])
        self.assertIsNone(result["probability"])
        self.assertEqual(result["predictor"], "Unknown")
        self.assertEqual(len(self.action_layer.action_log), 1)

    def test_invalid_prediction_type(self):
        with self.assertRaises(TypeError):
            self.action_layer.evaluate_and_act("not a list", [1, 2])

    def test_invalid_memory_state_type(self):
        with self.assertRaises(TypeError):
            self.action_layer.evaluate_and_act([{"page": 1}], "not a list")

    def test_invalid_prediction_dict(self):
        with self.assertRaises(ValueError):
            self.action_layer.evaluate_and_act([{"missing_page_key": 5}], [1, 2])

    def test_clear_log(self):
        prediction = [{"page": 5, "probability": 0.8}]
        memory_state = [1, 2, 5, 8]
        
        self.action_layer.evaluate_and_act(prediction, memory_state)
        self.assertEqual(len(self.action_layer.action_log), 1)
        
        self.action_layer.clear_log()
        self.assertEqual(len(self.action_layer.action_log), 0)

if __name__ == "__main__":
    unittest.main()
