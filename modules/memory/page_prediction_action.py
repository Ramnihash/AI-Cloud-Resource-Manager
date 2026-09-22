"""
Action layer for memory page prediction (CP4.2).

This module provides the :class:`PagePredictionActionLayer`, which takes
predictions from either the Markov or Transformer predictors and turns them
into a simulated memory-management action (PREFETCH or PROTECT).
"""

class PagePredictionActionLayer:
    """Evaluates page predictions and determines the corresponding action.
    
    This is a safe simulator-level component. It does not invasively modify
    the classical page replacement loops; instead, it outputs action directives
    and maintains a log for later evaluation.
    
    Actions:
        - PREFETCH: The predicted page is not currently in memory.
        - PROTECT: The predicted page is already in memory.
        - NO_ACTION: No valid prediction was provided.
    """

    def __init__(self):
        self.action_log = []

    def evaluate_and_act(
        self,
        prediction: list,
        memory_state: list,
        predictor_source: str = "Unknown"
    ) -> dict:
        """Evaluate the prediction against the current memory state.

        Args:
            prediction: List of candidates from a predictor (e.g., Markov or Transformer).
                        Expected format: [{"page": int, "probability": float}, ...]
            memory_state: List of page IDs currently in memory (frames).
            predictor_source: String identifying the predictor (e.g., "Markov").

        Returns:
            dict: The action directive containing the decision and reasoning.
        """
        if not isinstance(prediction, list):
            raise TypeError(f"prediction must be a list, got {type(prediction).__name__}")
        if not isinstance(memory_state, list):
            raise TypeError(f"memory_state must be a list, got {type(memory_state).__name__}")

        if len(prediction) == 0:
            action_dict = {
                "predictor": predictor_source,
                "predicted_page": None,
                "probability": None,
                "memory_state": list(memory_state),
                "action": "NO_ACTION",
                "reason": "No prediction candidates provided"
            }
            self.action_log.append(action_dict)
            return action_dict

        top_candidate = prediction[0]
        
        if not isinstance(top_candidate, dict) or "page" not in top_candidate:
            raise ValueError("Prediction candidate must be a dict with a 'page' key.")

        predicted_page = top_candidate["page"]
        probability = top_candidate.get("probability", 0.0)

        if predicted_page in memory_state:
            action = "PROTECT"
            reason = f"Predicted page {predicted_page} is already in memory."
        else:
            action = "PREFETCH"
            reason = f"Predicted page {predicted_page} is not in memory."

        action_dict = {
            "predictor": predictor_source,
            "predicted_page": predicted_page,
            "probability": probability,
            "memory_state": list(memory_state),
            "action": action,
            "reason": reason
        }
        
        self.action_log.append(action_dict)
        return action_dict

    def clear_log(self):
        """Clear the internal action log."""
        self.action_log.clear()
