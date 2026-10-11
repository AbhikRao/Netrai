#!/usr/bin/env python3
"""Flag possible transformed-image overlap across source datasets/splits.

64-bit DCT perceptual hashes are a triage tool: a close hash is not proof of the
same retina and a distant hash cannot exclude cropped/rotated copies. No
candidate is removed automatically or treated as a confirmed leak.
"""

import argparse
import heapq
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
SUFFIXES = {'.jpeg', '.jpg', '.png', '.tif', '.tiff'}


def perceptual_hash(path):
    image = cv2.imread(str(path), cv2.IMREAD_REDUCED_GRAYSCALE_8)
    if image is None:
        raise ValueError(f'Cannot decode {path}')
    mask = image > 8
    if mask.any():
        ys, xs = mask.nonzero()
        image = image[ys.min():ys.max()+1, xs.min():xs.max()+1]
    thumbnail = cv2.resize(image, (32, 32), interpolation=cv2.INTER_AREA)
    low_frequency = cv2.dct(thumbnail.astype(np.float32))[:8, :8]
    threshold = np.median(low_frequency.flatten()[1:])
    bits = (low_frequency > threshold).flatten()
    bits[0] = False  # brightness-independent DC coefficient
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return path, value


def scan_candidates(left, right, maximum_distance=4, max_stored=None):
    """Count all close hashes; keep at most the closest max_stored pairs."""
    if not 0 <= maximum_distance <= 7:
        raise ValueError('maximum_distance must be in [0, 7] for this exact bucket search')
    if max_stored is not None and max_stored < 1:
        raise ValueError('max_stored must be positive when specified')
    block_count = maximum_distance + 1
    widths = [64 // block_count + (block < 64 % block_count)
              for block in range(block_count)]
    offsets = [sum(widths[:block]) for block in range(block_count)]
    blocks = [(offset, (1 << width)-1) for offset, width in zip(offsets, widths)]
    index = defaultdict(list)
    for path, value in left:
        for block, (offset, mask) in enumerate(blocks):
            index[(block, (value >> offset) & mask)].append((path, value))
    found = []
    total = 0
    distance_counts = {distance: 0 for distance in range(maximum_distance+1)}
    for path, value in right:
        possible = {}
        for block, (offset, mask) in enumerate(blocks):
            for earlier_path, earlier_value in index[(block, (value >> offset) & mask)]:
                possible[earlier_path] = earlier_value
        for earlier_path, earlier_value in possible.items():
            distance = (earlier_value ^ value).bit_count()
            if distance <= maximum_distance:
                total += 1
                distance_counts[distance] += 1
                pair = (str(earlier_path), str(path), distance)
                if max_stored is None:
                    found.append(pair)
                elif len(found) < max_stored:
                    heapq.heappush(found, (-distance, pair))
                elif distance < -found[0][0]:
                    heapq.heapreplace(found, (-distance, pair))
    if max_stored is not None:
        found = [pair for _, pair in found]
    return {'total': total, 'by_distance': distance_counts,
            'pairs': sorted(found, key=lambda pair: (pair[2], pair[0], pair[1]))}


def candidates(left, right, maximum_distance=4):
    """Return all pairs for small diagnostic inputs; production uses a cap."""
    return scan_candidates(left, right, maximum_distance)['pairs']


def images(folder):
    if not folder.is_dir():
        raise FileNotFoundError(folder)
    return sorted(path for path in folder.rglob('*')
                  if path.is_file() and path.suffix.lower() in SUFFIXES)


def source_hashes(name, paths, output, workers, progress):
    """Reuse hashes only when every source path, size, and mtime still matches."""
    cache = output/'cache'/f'{name}_phash.csv'
    signatures = [(str(path), path.stat().st_size, path.stat().st_mtime_ns)
                  for path in paths]
    if cache.is_file():
        previous = pd.read_csv(cache, dtype={'hash_hex': str})
        if (list(zip(previous.path, previous.bytes, previous.mtime_ns)) == signatures):
            return [(Path(path), int(value, 16))
                    for path, value in zip(previous.path, previous.hash_hex)]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        values = list(tqdm(pool.map(perceptual_hash, paths), total=len(paths),
                           desc=f'hash {name}', disable=not progress))
    cache.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{'path': path, 'bytes': size, 'mtime_ns': mtime,
                   'hash_hex': f'{value:016x}'}
                  for (path, size, mtime), (_, value) in zip(signatures, values)]).to_csv(
                      cache, index=False)
    return values


def run(args):
    cv2.setNumThreads(1)
    roots = {'eyeq_train': args.eyeq/'images/train',
             'eyeq_test': args.eyeq/'images/test',
             'aptos_train': args.aptos/'train_images',
             'idrid_grading': args.idrid/'B. Disease Grading/1. Original Images'}
    sources = {name: images(folder) for name, folder in roots.items()}
    hashes = {}
    for name, paths in sources.items():
        hashes[name] = source_hashes(name, paths, args.output, args.workers, args.progress)
    comparisons = [('eyeq_train', 'eyeq_test'),
                   ('eyeq_train', 'aptos_train'), ('eyeq_test', 'aptos_train'),
                   ('eyeq_train', 'idrid_grading'), ('eyeq_test', 'idrid_grading'),
                   ('aptos_train', 'idrid_grading')]
    rows, counts, counts_by_distance, retained = [], {}, {}, {}
    for first, second in comparisons:
        result = scan_candidates(hashes[first], hashes[second], args.max_distance,
                                 args.max_stored_per_comparison)
        found = result['pairs']
        key = f'{first}__{second}'
        counts[key] = result['total']
        counts_by_distance[key] = result['by_distance']
        retained[key] = len(found)
        rows.extend({'source_a': first, 'source_b': second,
                     'image_a': Path(left).relative_to(roots[first]).as_posix(),
                     'image_b': Path(right).relative_to(roots[second]).as_posix(),
                     'hamming_distance': distance}
                    for left, right, distance in found)
    args.output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=['source_a', 'source_b', 'image_a', 'image_b',
                                'hamming_distance']).to_csv(
        args.output/'perceptual_overlap_candidates.csv', index=False)
    summary = {'schema_version': 2, 'method': 'foreground-cropped grayscale DCT pHash 64-bit',
               'maximum_hamming_distance': args.max_distance,
               'source_image_counts': {name: len(paths) for name, paths in sources.items()},
               'candidate_pair_counts': counts,
               'candidate_counts_by_hamming_distance': counts_by_distance,
               'stored_pair_counts': retained,
               'max_stored_per_comparison': args.max_stored_per_comparison,
               'status': 'candidate_review_required_not_confirmed_leakage',
               'limitations': ['Near hashes can be different retinas; human visual review is required.',
                               'CSV retains only the closest matches up to the declared cap; summary counts include all candidates.',
                               'Crop, rotation, mirror, or strong color changes can evade this hash.',
                               'Filename patient identifiers are not cross-dataset identities.']}
    (args.output/'perceptual_overlap_summary.json').write_text(
        json.dumps(summary, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--eyeq', type=Path, default=ROOT/'data/eyeq')
    parser.add_argument('--aptos', type=Path, default=ROOT/'data/aptos2019')
    parser.add_argument('--idrid', type=Path, default=ROOT/'data/idrid')
    parser.add_argument('--output', type=Path,
                        default=ROOT/'results/verification_2026-09-24/perceptual_overlap')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--max-distance', type=int, default=4)
    parser.add_argument('--max-stored-per-comparison', type=int, default=4000)
    parser.add_argument('--progress', action='store_true')
    run(parser.parse_args())
