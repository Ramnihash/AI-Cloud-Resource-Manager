"""
Action layer for memory health (CP4.3).

This module provides the :class:`MemoryHealthActionLayer`, which consumes
the outputs of the MemoryLeakDetector (CP3.6) and MemoryPressureForecaster (CP3.5)
to generate simulated management directives like RESTART_PROCESS or ALERT_HIGH_PRESSURE.
"""

from typing import Optional, Dict, Any

class MemoryHealthActionLayer:
    """Evaluates memory health signals and determines the corresponding action.
    
    This is a safe simulator-level component. It does not perform actual OS-level 
    process killing or restarting. It outputs action directives and maintains a log.
    
    Priority of actions:
        1. RESTART_PROCESS (if a memory leak is detected)
        2. ALERT_HIGH_PRESSURE (if predicted RAM >= threshold)
        3. NO_ACTION
    """

    def __init__(self, pressure_threshold: float = 90.0):
        """
        Args:
            pressure_threshold: RAM percentage (0-100) above which to trigger an alert.
        """
        if not isinstance(pressure_threshold, (int, float)) or isinstance(pressure_threshold, bool):
            raise TypeError(f"pressure_threshold must be numeric, got {type(pressure_threshold).__name__}")
        if not (0.0 <= pressure_threshold <= 100.0):
            raise ValueError(f"pressure_threshold must be in [0.0, 100.0], got {pressure_threshold}")
            
        self.pressure_threshold = float(pressure_threshold)
        self.action_log = []

    def evaluate_and_act(
        self,
        leak_detection: Optional[Dict[str, Any]] = None,
        pressure_prediction: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Evaluate intelligence outputs and return an action directive.

        Args:
            leak_detection: Output from MemoryLeakDetector.detect().
            pressure_prediction: Output from MemoryPressureForecaster.predict().

        Returns:
            dict: The action directive containing decision, context, and reasoning.
        """
        action = "NO_ACTION"
        reason = "Memory health is nominal"
        proc_pid = None
        predicted_ram = None
        source = []
        
        if leak_detection is not None:
            if not isinstance(leak_detection, dict):
                raise TypeError("leak_detection must be a dictionary")
            proc_pid = leak_detection.get("proc_pid")
            source.append("MemoryLeakDetector")
            
        if pressure_prediction is not None:
            if not isinstance(pressure_prediction, dict):
                raise TypeError("pressure_prediction must be a dictionary")
            predicted_ram = pressure_prediction.get("predicted_ram_percent")
            source.append("MemoryPressureForecaster")
            
        if not source:
            source.append("None")

        # Priority 1: Memory Leak
        if leak_detection and leak_detection.get("is_leak_detected") is True:
            action = "RESTART_PROCESS"
            reason = f"Memory leak detected for PID {proc_pid}"
        # Priority 2: High Memory Pressure
        elif predicted_ram is not None and predicted_ram >= self.pressure_threshold:
            action = "ALERT_HIGH_PRESSURE"
            reason = f"Predicted RAM ({predicted_ram:.2f}%) exceeds threshold ({self.pressure_threshold}%)"

        action_dict = {
            "source": ", ".join(source),
            "pid": proc_pid,
            "predicted_ram_percent": predicted_ram,
            "threshold": self.pressure_threshold,
            "action": action,
            "reason": reason
        }
        
        self.action_log.append(action_dict)
        return action_dict

    def clear_log(self):
        """Clear the internal action log."""
        self.action_log.clear()
