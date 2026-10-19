"""CP7.2 consolidated memory evaluation. Run: python -m scripts.evaluate_memory_module"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from ml.memory.memory_pressure_forecaster import MemoryPressureForecaster
from ml.memory.memory_leak_detector import MemoryLeakDetector

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'memory'
OUT = DATA / 'evaluation_cp72'


def read_csv(path):
    with path.open(newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def load_pressure_data():
    rows = read_csv(DATA / 'live_logs' / 'baseline.csv')
    snapshots = []
    for row in rows:
        try:
            v = float(row['ram_percent'])
            if 0 <= v <= 100:
                snapshots.append({'ram_percent': v})
        except (KeyError, ValueError, TypeError):
            continue
    return snapshots


def pressure_metrics():
    snapshots = load_pressure_data()
    window = 5
    split = int(len(snapshots) * 0.7)
    if split < window + 1 or len(snapshots) - split < 1:
        raise ValueError('Insufficient baseline snapshots for chronological pressure evaluation')
    model = MemoryPressureForecaster(window_size=window)
    model.train(snapshots[:split])
    # Include preceding context so predictions start at the first held-out observation.
    test_with_context = snapshots[split-window:]
    result = model.evaluate(test_with_context)
    result.update({'source': 'live_logs/baseline.csv', 'split': 'chronological_70_30',
                   'train_snapshots': split, 'test_snapshots': len(snapshots)-split,
                   'window_size': window,
                   'caution': 'Short single-session baseline; not a broad generalization benchmark.'})
    return result


def leak_metrics():
    detector = MemoryLeakDetector(window_size=10, min_growth_bytes_per_step=1024.0, min_r2=0.75)
    tp = tn = fp = fn = 0
    cases = []
    # Fixed, labeled synthetic traces; not independently verified real-world leaks.
    for i in range(20):
        expected = i < 10
        pid = 5000 + i
        base = 20_000_000 + i * 100_000
        values = ([base + t * (3000 + 200 * (i % 3)) for t in range(10)] if expected
                  else [base + ((t * 7 + i * 3) % 5) * 120 for t in range(10)])
        snaps = [{'proc_pid': pid, 'proc_rss_bytes': x} for x in values]
        pred = detector.detect(snaps)['is_leak_detected']
        cases.append({'case': i, 'expected': expected, 'predicted': pred})
        if expected and pred: tp += 1
        elif not expected and not pred: tn += 1
        elif not expected and pred: fp += 1
        else: fn += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return ({'samples': len(cases), 'true_positive': tp, 'true_negative': tn,
             'false_positive': fp, 'false_negative': fn,
             'accuracy': (tp + tn) / len(cases), 'precision': precision,
             'recall': recall, 'f1': 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
             'source': 'fixed labeled synthetic RSS traces',
             'caution': 'Synthetic classification performance does not establish real-world leak detection accuracy.'}, cases)


def save_plot(path, names, series, ylabel, title):
    fig, ax = plt.subplots(figsize=(9, 4.8))
    width = 0.8 / len(series)
    for j, (label, vals) in enumerate(series):
        xs = [i - 0.4 + width * (j + 0.5) for i in range(len(names))]
        ax.bar(xs, vals, width=width, label=label)
    ax.set_xticks(range(len(names)), names)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    comparison = json.loads((DATA / 'evaluation_cp34' / 'comparison.json').read_text(encoding='utf-8'))
    page_rows = read_csv(DATA / 'evaluation_cp42' / 'page_action_impact.csv')
    pressure = pressure_metrics()
    leak, cases = leak_metrics()
    names = ['Markov', 'Transformer']
    top1 = [100 * comparison[m]['metrics']['overall']['top1_accuracy'] for m in ('markov', 'transformer')]
    top3 = [100 * comparison[m]['metrics']['overall']['top3_accuracy'] for m in ('markov', 'transformer')]
    save_plot(OUT / 'predictor_accuracy.png', names, [('Top-1', top1), ('Top-3', top3)],
              'Accuracy (%)', 'Held-out synthetic page prediction')
    workload_names = [r['case'] for r in page_rows]
    save_plot(OUT / 'page_faults.png', workload_names,
              [('Baseline faults', [int(r['baseline_faults']) for r in page_rows]),
               ('Guided demand faults', [int(r['guided_demand_faults']) for r in page_rows])],
              'Demand faults', 'LRU versus simulated Markov-guided prefetch')
    save_plot(OUT / 'page_loads.png', workload_names,
              [('Baseline loads', [int(r['baseline_faults']) for r in page_rows]),
               ('Guided total loads', [int(r['guided_total_loads']) for r in page_rows])],
              'Page loads (demand + prefetch)', 'Prefetch cost comparison')
    results = {'page_prediction': {'test_windows': comparison['split_rows']['test'],
                                    'markov_top1': top1[0] / 100, 'transformer_top1': top1[1] / 100,
                                    'markov_top3': top3[0] / 100, 'transformer_top3': top3[1] / 100},
               'page_actions': page_rows, 'pressure_forecasting': pressure,
               'leak_detection': leak}
    (OUT / 'summary.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    with (OUT / 'leak_cases.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['case', 'expected', 'predicted'])
        writer.writeheader(); writer.writerows(cases)
    lines = [
        '# CP7.2 — Module 2 Memory Evaluation', '',
        '## Page prediction (held-out synthetic data)',
        f'- Markov Top-1: {top1[0]:.2f}%; Top-3: {top3[0]:.2f}%.',
        f'- Transformer Top-1: {top1[1]:.2f}%; Top-3: {top3[1]:.2f}%.',
        '- Sequential traces repeat by design; 100% sequential accuracy should not be interpreted as generalization.',
        '', '![Prediction accuracy](predictor_accuracy.png)', '',
        '## Simulated prefetch evaluation',
        '| Workload | Baseline demand faults | Guided demand faults | Guided prefetch loads | Guided total loads |',
        '|---|---:|---:|---:|---:|',
    ]
    for r in page_rows:
        lines.append(f"| {r['case']} | {r['baseline_faults']} | {r['guided_demand_faults']} | {r['guided_prefetch_loads']} | {r['guided_total_loads']} |")
    lines += ['', '![Demand faults](page_faults.png)', '', '![Total page loads](page_loads.png)', '',
              '- Prefetching is simulated; actual OS-level prefetch and page protection are not implemented.',
              '- Demand-fault reductions are not equivalent to reductions in total page-loading work.', '',
              '## Memory pressure forecasting',
              f"- Source: {pressure['source']}; chronological 70/30 split; {pressure['test_snapshots']} held-out observations.",
              f"- MAE: {pressure['MAE']:.4f} percentage points; MSE: {pressure['MSE']:.4f} squared percentage points.",
              '- This is a short single-session log and is not a broad validation dataset.', '',
              '## Suspected memory leak detection',
              f"- Fixed labeled synthetic traces: {leak['samples']}; TP={leak['true_positive']}, TN={leak['true_negative']}, FP={leak['false_positive']}, FN={leak['false_negative']}.",
              f"- Accuracy: {leak['accuracy']:.2%}; precision: {leak['precision']:.2%}; recall: {leak['recall']:.2%}; F1: {leak['f1']:.2%}.",
              '- Sustained RSS growth is a leak-risk signal, not proof of a genuine memory leak.', '',
              '## Conclusions and limitations',
              '- The action layers issue simulated directives; RESTART_PROCESS does not restart an OS process.',
              '- The predictive prefetch experiment does not enforce PROTECT decisions.',
              '- Synthetic workloads and short live telemetry limit generalization; larger independent datasets are future work.', '']
    (OUT / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
    print(f"Pressure: MAE={pressure['MAE']:.4f}, MSE={pressure['MSE']:.4f} ({pressure['total_predictions']} held-out predictions)")
    print(f"Synthetic leak cases: accuracy={leak['accuracy']:.2%}, precision={leak['precision']:.2%}, recall={leak['recall']:.2%}, F1={leak['f1']:.2%}")
    print('Saved:', OUT)


if __name__ == '__main__':
    main()
