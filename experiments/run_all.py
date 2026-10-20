"""Rerun every evaluation study. Usage: python -m experiments.run_all"""

from experiments import cpu_eval, device_eval

STUDIES = [("7.1 CPU", cpu_eval.main), ("7.3 Device", device_eval.main)]

if __name__ == "__main__":
    for name, study in STUDIES:
        print(f"=== {name} ===")
        study()
