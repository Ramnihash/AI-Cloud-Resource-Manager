"""CP2.2: reproducible page-reference datasets with trace-level split isolation.
Run: python -m scripts.prepare_memory_dataset
"""
import csv
import json
import random
from pathlib import Path

from ml.memory.page_reference_generator import PageReferenceGenerator

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data' / 'memory' / 'page_sequences_v1'
PATTERNS = ('uniform', 'sequential', 'looping', 'mixed')
SPLITS = {'train': 24, 'validation': 8, 'test': 8}
NUM_PAGES = 32
TRACE_LENGTH = 256
WINDOW_SIZE = 8
BASE_SEED = 2026


def make_trace(generator, pattern, seed):
    if pattern in ('uniform', 'sequential'):
        return generator.generate(NUM_PAGES, TRACE_LENGTH, pattern, seed=seed)['sequence']
    rng = random.Random(seed)
    if pattern == 'looping':
        loop = rng.sample(range(NUM_PAGES), 5)
        return [loop[i % len(loop)] for i in range(TRACE_LENGTH)]
    # Mixed: concatenate independently generated segments of different behaviors.
    # Segment boundaries are part of the trace, not dataset split boundaries.
    sequence = []
    for block in range(0, TRACE_LENGTH, 32):
        length = min(32, TRACE_LENGTH - block)
        kind = ('uniform', 'locality', 'sequential')[((block // 32) + seed) % 3]
        part = generator.generate(NUM_PAGES, length, kind, seed=seed + block)['sequence']
        if kind == 'sequential':
            offset = rng.randrange(NUM_PAGES)
            part = [(page + offset) % NUM_PAGES for page in part]
        sequence.extend(part)
    return sequence


def build_dataset():
    OUT.mkdir(parents=True, exist_ok=True)
    generator = PageReferenceGenerator()
    manifest = []
    seen_trace_ids = set()
    for split_index, (split, count) in enumerate(SPLITS.items()):
        path = OUT / f'{split}.csv'
        with path.open('w', newline='', encoding='utf-8') as handle:
            writer = csv.writer(handle)
            writer.writerow(['trace_id', 'pattern', 'position', 'input_pages', 'target_page'])
            for pattern_index, pattern in enumerate(PATTERNS):
                for trace_number in range(count):
                    seed = BASE_SEED + split_index * 100000 + pattern_index * 10000 + trace_number
                    trace_id = f'{split}_{pattern}_{trace_number:03d}'
                    assert trace_id not in seen_trace_ids
                    seen_trace_ids.add(trace_id)
                    sequence = make_trace(generator, pattern, seed)
                    assert len(sequence) == TRACE_LENGTH
                    assert all(isinstance(p, int) and 0 <= p < NUM_PAGES for p in sequence)
                    for target_pos in range(WINDOW_SIZE, len(sequence)):
                        writer.writerow([trace_id, pattern, target_pos,
                                         json.dumps(sequence[target_pos - WINDOW_SIZE:target_pos]),
                                         sequence[target_pos]])
                    manifest.append({'trace_id': trace_id, 'split': split, 'pattern': pattern,
                                     'seed': seed, 'length': TRACE_LENGTH})
        print(f'{split}: {count * len(PATTERNS)} traces, '
              f'{count * len(PATTERNS) * (TRACE_LENGTH - WINDOW_SIZE)} windows -> {path}')
    metadata = {'num_pages': NUM_PAGES, 'trace_length': TRACE_LENGTH,
                'window_size': WINDOW_SIZE, 'base_seed': BASE_SEED,
                'patterns': PATTERNS, 'traces_per_pattern': SPLITS,
                'note': 'Split by independent trace ID before window creation. Sequential traces are identical by design; evaluate memorization/generalization separately.',
                'traces': manifest}
    (OUT / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    # Verify saved artifacts and split isolation.
    split_ids = {}
    for split in SPLITS:
        with (OUT / f'{split}.csv').open(newline='', encoding='utf-8') as handle:
            reader = csv.DictReader(handle)
            ids = set()
            rows = 0
            for row in reader:
                ids.add(row['trace_id'])
                window = json.loads(row['input_pages'])
                assert len(window) == WINDOW_SIZE
                assert all(0 <= int(p) < NUM_PAGES for p in window)
                assert 0 <= int(row['target_page']) < NUM_PAGES
                rows += 1
            assert rows == SPLITS[split] * len(PATTERNS) * (TRACE_LENGTH - WINDOW_SIZE)
            split_ids[split] = ids
    assert split_ids['train'].isdisjoint(split_ids['validation'])
    assert split_ids['train'].isdisjoint(split_ids['test'])
    assert split_ids['validation'].isdisjoint(split_ids['test'])
    print('CP2.2 dataset checks passed (shape, page range, and trace ID isolation).')


if __name__ == '__main__':
    build_dataset()
