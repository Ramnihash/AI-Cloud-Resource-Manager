"""CPU Machine Learning module package."""

from ml.cpu.burst_predictor import CPUBurstPredictor
from ml.cpu.workload_classifier import CPUWorkloadClassifier, ClassificationResult

__all__ = [
    "CPUBurstPredictor",
    "CPUWorkloadClassifier",
    "ClassificationResult",
]
