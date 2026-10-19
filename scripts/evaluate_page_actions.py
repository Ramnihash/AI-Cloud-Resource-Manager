"""CP4.2: compare demand-only LRU with simulated Markov-guided prefetch LRU.

Run: python -m scripts.evaluate_page_actions
Prefetches are physical page loads, not free; demand faults and total loads
are reported separately. No real OS memory is modified.
"""
from __future__ import annotations

import argparse
import csv
from collections import OrderedDict
from pathlib import Path

from ml.memory.markov_predictor import MarkovPagePredictor
from modules.memory.page_prediction_action import PagePredictionActionLayer
from modules.memory.page_replacement import LRUPageReplacement


def simulate(reference, frames, predictor=None, threshold=0.0):
    """Demand accesses first, then predict/prefetch for the *next* access."""
    cache = OrderedDict()
    action_layer = PagePredictionActionLayer()
    stats = dict(demand_faults=0, demand_hits=0, prefetch_loads=0,
                 useful_prefetches=0, wasted_prefetches=0,
                 protect_decisions=0, prefetch_decisions=0, no_action_decisions=0)
    # Prefetched pages are marked until their first demand use or eviction.
    pending = set()
    for index, page in enumerate(reference):
        if page in cache:
            stats['demand_hits'] += 1
            cache.move_to_end(page)
            if page in pending:
                pending.remove(page)
                stats['useful_prefetches'] += 1
        else:
            stats['demand_faults'] += 1
            if len(cache) == frames:
                victim, _ = cache.popitem(last=False)
                if victim in pending:
                    pending.remove(victim)
                    stats['wasted_prefetches'] += 1
            cache[page] = None
        if predictor is None or index == len(reference) - 1:
            continue
        candidates = predictor.predict(page, top_k=1)
        if not candidates or candidates[0]['probability'] < threshold:
            candidates = []
        decision = action_layer.evaluate_and_act(candidates, list(cache), 'Markov')
        action = decision['action']
        if action == 'PREFETCH':
            stats['prefetch_decisions'] += 1
            predicted = decision['predicted_page']
            if len(cache) == frames:
                victim, _ = cache.popitem(last=False)
                if victim in pending:
                    pending.remove(victim)
                    stats['wasted_prefetches'] += 1
            cache[predicted] = None
            pending.add(predicted)
            stats['prefetch_loads'] += 1
        elif action == 'PROTECT':
            # The existing action layer emits a directive only. This simulator
            # records PROTECT but does not pretend to implement pinned frames.
            stats['protect_decisions'] += 1
        else:
            stats['no_action_decisions'] += 1
    stats['wasted_prefetches'] += len(pending)
    stats['total_loads'] = stats['demand_faults'] + stats['prefetch_loads']
    stats['references'] = len(reference)
    return stats


def run_case(name, train, test, frames, threshold):
    predictor = MarkovPagePredictor()
    predictor.train(train)
    baseline = LRUPageReplacement(frames).run(test)
    plain = simulate(test, frames)
    assert plain['demand_faults'] == baseline['page_faults']
    guided = simulate(test, frames, predictor, threshold)
    return {
        'case': name, 'frames': frames, 'references': len(test),
        'baseline_faults': baseline['page_faults'],
        'guided_demand_faults': guided['demand_faults'],
        'guided_prefetch_loads': guided['prefetch_loads'],
        'guided_total_loads': guided['total_loads'],
        'fault_change': guided['demand_faults'] - baseline['page_faults'],
        'load_change': guided['total_loads'] - baseline['page_faults'],
        'useful_prefetches': guided['useful_prefetches'],
        'wasted_prefetches': guided['wasted_prefetches'],
        'protect_decisions': guided['protect_decisions'],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--frames', type=int, default=3)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--output', type=Path, default=Path('data/memory/evaluation_cp42/page_action_impact.csv'))
    args = parser.parse_args()
    if args.frames < 1 or not 0 <= args.threshold <= 1:
        parser.error('frames must be positive and threshold in [0,1]')
    cases = [
        ('cyclic', [0, 1, 2, 3] * 30, [0, 1, 2, 3] * 10),
        ('locality', [0, 1, 0, 1, 2, 0, 1, 3] * 30,
         [0, 1, 0, 1, 2, 0, 1, 3] * 10),
        ('classic', [1, 2, 3, 4, 1, 2, 5, 1, 2, 3, 4, 5] * 20,
         [1, 2, 3, 4, 1, 2, 5, 1, 2, 3, 4, 5] * 3),
    ]
    rows = [run_case(name, train, test, args.frames, args.threshold)
            for name, train, test in cases]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for r in rows:
        print(f"{r['case']}: baseline faults={r['baseline_faults']}, "
              f"guided demand faults={r['guided_demand_faults']}, "
              f"prefetch loads={r['guided_prefetch_loads']}, "
              f"total loads={r['guided_total_loads']}, "
              f"useful={r['useful_prefetches']}, wasted={r['wasted_prefetches']}")
    print('Saved:', args.output)
    print('NOTE: PROTECT is logged only; this is a simulated prefetch experiment, not real OS paging.')


if __name__ == '__main__':
    main()
