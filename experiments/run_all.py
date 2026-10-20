"""Rerun every evaluation study. Usage: python -m experiments.run_all"""

from experiments import cpu_eval, device_eval, memory_eval, robustness_eval

STUDIES = [("7.1 CPU", cpu_eval.main), ("7.2 Memory", memory_eval.main), ("7.3 Device", device_eval.main), ("7.4-7.5 Robustness", robustness_eval.main)]

if __name__ == "__main__":
    for name, study in STUDIES:
        print(f"=== {name} ===")
        study()
