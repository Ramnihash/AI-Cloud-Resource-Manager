#!/usr/bin/env python3
"""
Demonstration script for Module 2: Predictive Memory Management Pipeline.
This covers CP1.2 - CP4.3 using synthetic data traces.
"""

import sys
import os

# Ensure the project root is in the path so we can run this from anywhere
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 1. CLASSICAL PAGE REPLACEMENT
from modules.memory.page_replacement import FIFOPageReplacement, LRUPageReplacement, OptimalPageReplacement

# 2. MARKOV
from ml.memory.markov_predictor import MarkovPagePredictor

# 3. TRANSFORMER
try:
    from ml.memory.transformer_predictor import TransformerPagePredictor
except ImportError:
    TransformerPagePredictor = None

# 4. MEMORY PRESSURE FORECASTING
from ml.memory.memory_pressure_forecaster import MemoryPressureForecaster

# 5. MEMORY LEAK DETECTION
from ml.memory.memory_leak_detector import MemoryLeakDetector

# 6 & 7. ACTION LAYERS
from modules.memory.page_prediction_action import PagePredictionActionLayer
from modules.memory.memory_health_action import MemoryHealthActionLayer

def print_header(title):
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")

def main():
    print_header("1. CLASSICAL PAGE REPLACEMENT")
    ref_string = [1, 2, 3, 4, 1, 2, 5, 1, 2, 3, 4, 5]
    num_frames = 3
    print(f"Reference String: {ref_string}")
    print(f"Number of frames: {num_frames}\n")

    for Algo in [FIFOPageReplacement, LRUPageReplacement, OptimalPageReplacement]:
        algo = Algo(num_frames)
        res = algo.run(ref_string)
        print(f"Algorithm: {res['algorithm']}")
        print(f"  Page Faults: {res['page_faults']}")
        print(f"  Page Hits: {res['page_hits']}")
        print(f"  Hit Ratio: {res['hit_ratio']:.2f}")
        print(f"  Miss Ratio: {res['miss_ratio']:.2f}")
        # Show a subset of snapshots to keep it readable
        print(f"  Frame Snapshots (first 5): {res['frame_snapshots'][:5]} ...\n")

    print_header("2. PAGE PREDICTION — MARKOV")
    markov = MarkovPagePredictor()
    markov.train(ref_string)
    current_page = 2
    preds_markov = markov.predict(current_page, top_k=2)
    print("Dataset: Synthetic/Demo Trace")
    print(f"Current Page: {current_page}")
    print("Predictor: Markov (Order 1)")
    print(f"Predicted next page candidates: {preds_markov}")

    print_header("3. PAGE PREDICTION — TRANSFORMER")
    preds_trans = []
    if TransformerPagePredictor:
        transformer = TransformerPagePredictor(num_pages=10, context_len=2, d_model=16, nhead=2)
        print("Dataset: Synthetic/Demo Trace")
        print("Training Transformer (1 epoch for demo)...")
        transformer.train(ref_string, epochs=1)
        
        input_seq = [1, 2]
        preds_trans = transformer.predict(input_seq)
        print(f"Input Sequence: {input_seq}")
        print("Predictor: Transformer")
        print(f"Predicted next page candidates: {preds_trans}")
    else:
        print("Skipping Transformer: Dependencies not met (PyTorch is required).")

    print_header("4. MEMORY PRESSURE FORECASTING")
    # Synthetic snapshots
    snapshots = [
        {"ram_percent": 50.0},
        {"ram_percent": 55.0},
        {"ram_percent": 60.0},
        {"ram_percent": 65.0},
        {"ram_percent": 70.0},
        {"ram_percent": 75.0}
    ]
    forecaster = MemoryPressureForecaster(window_size=5)
    forecaster.train(snapshots)
    
    recent_snapshots = snapshots[-5:]
    pressure_pred = forecaster.predict(recent_snapshots)
    print("Dataset: Synthetic/Demo Trace (Not live psutil telemetry)")
    print(f"Input Window RAM %: {[s['ram_percent'] for s in recent_snapshots]}")
    print(f"Predicted next RAM %: {pressure_pred['predicted_ram_percent']:.2f}%")

    print_header("5. MEMORY LEAK DETECTION")
    # Synthetic process snapshots
    leak_snapshots = [
        {"proc_pid": 123, "proc_rss_bytes": 1000},
        {"proc_pid": 123, "proc_rss_bytes": 2100},
        {"proc_pid": 123, "proc_rss_bytes": 3200},
        {"proc_pid": 123, "proc_rss_bytes": 4150}
    ]
    detector = MemoryLeakDetector(window_size=4, min_growth_bytes_per_step=1000.0, min_r2=0.75)
    leak_res = detector.detect(leak_snapshots)
    print("Dataset: Synthetic/Demo Trace")
    print(f"PID: {leak_res['proc_pid']}")
    print(f"RSS Trend (Bytes/step): {leak_res['growth_rate_bytes_per_step']:.2f}")
    print(f"Trend R^2 Score: {leak_res['trend_r2_score']:.2f}")
    print(f"Leak Detected: {leak_res['is_leak_detected']}")

    print_header("6. CP4.2 ACTION (Page Prediction)")
    page_action_layer = PagePredictionActionLayer()
    memory_state = [1, 2, 5]
    print(f"Predicted Output (from Markov): {preds_markov}")
    print(f"Current Memory State (frames): {memory_state}")
    
    # Demonstrate PROTECT
    action_res_42 = page_action_layer.evaluate_and_act(preds_markov, memory_state, "Markov")
    print(f"Selected Action: {action_res_42['action']}")
    print(f"Reason: {action_res_42['reason']}")

    # Demonstrate PREFETCH
    print("\nDemonstrating PREFETCH (Page 4 not in memory):")
    action_res_42_pre = page_action_layer.evaluate_and_act([{"page": 4, "probability": 0.9}], memory_state, "Markov")
    print(f"Selected Action: {action_res_42_pre['action']}")
    print(f"Reason: {action_res_42_pre['reason']}")

    print_header("7. CP4.3 ACTION (Memory Health)")
    health_action_layer = MemoryHealthActionLayer(pressure_threshold=70.0)
    print("Input: Simulated Leak Detection AND High Pressure")
    print(f"Threshold: {health_action_layer.pressure_threshold}%")
    
    # Priority 1: Leak Detected
    action_res_43 = health_action_layer.evaluate_and_act(leak_res, pressure_pred)
    print(f"Predicted RAM: {action_res_43['predicted_ram_percent']:.2f}%")
    print(f"Leak Status for PID {action_res_43['pid']}: {leak_res['is_leak_detected']}")
    print(f"Selected Action: {action_res_43['action']}")
    print(f"Reason: {action_res_43['reason']}")
    
    # Priority 2: High Pressure (no leak)
    print("\nDemonstrating ALERT_HIGH_PRESSURE (No leak):")
    no_leak = dict(leak_res)
    no_leak["is_leak_detected"] = False
    action_res_43_alert = health_action_layer.evaluate_and_act(no_leak, pressure_pred)
    print(f"Selected Action: {action_res_43_alert['action']}")
    print(f"Reason: {action_res_43_alert['reason']}")

    print_header("8. FINAL PIPELINE SUMMARY")
    print("PREDICT -> DECIDE -> ACT -> MEASURE\n")
    print("PREDICT:")
    print("  - ml.memory.markov_predictor.MarkovPagePredictor")
    print("  - ml.memory.transformer_predictor.TransformerPagePredictor")
    print("  - ml.memory.memory_pressure_forecaster.MemoryPressureForecaster")
    print("  - ml.memory.memory_leak_detector.MemoryLeakDetector")
    print("\nDECIDE & ACT (Simulated):")
    print("  - modules.memory.page_prediction_action.PagePredictionActionLayer")
    print("  - modules.memory.memory_health_action.MemoryHealthActionLayer")
    print("\nMEASURE:")
    print("  - Internal action_log in Action Layers")
    print("  - modules.memory.page_replacement hit/miss ratios")
    print(f"{'='*60}\n")
    
if __name__ == "__main__":
    main()
