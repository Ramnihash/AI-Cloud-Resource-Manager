"""Rerun every evaluation study. Usage: python -m experiments.run_all"""

import importlib.util
import os

from experiments import concurrency_eval, cpu_eval, device_eval, memory_eval, robustness_eval

STUDIES = [("7.1 CPU", cpu_eval.main), ("7.2 Memory", memory_eval.main), ("7.3 Device", device_eval.main), ("7.4-7.5 Robustness", robustness_eval.main)]


def _transformer_pages():
    if importlib.util.find_spec("torch") is None:
        print("Skipped: needs PyTorch.")
        return
    from experiments import transformer_page_eval
    transformer_page_eval.main()


def _smart():
    from ml.device.smart.service import DATASET_PATH
    if not os.path.exists(DATASET_PATH):
        print("Skipped: build the dataset first with python -m scripts.build_smart_dataset")
        return
    from experiments import smart_eval
    smart_eval.main()


STUDIES += [("7.2 D Transformer page predictor", _transformer_pages), ("7.3 SMART classifier", _smart),
            ("7.5 C Concurrent users", lambda: concurrency_eval.main([]))]

if __name__ == "__main__":
    for name, study in STUDIES:
        print(f"=== {name} ===")
        study()
