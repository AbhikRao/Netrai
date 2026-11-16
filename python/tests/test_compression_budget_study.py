"""X16 upload-compression study (compression_budget_study.py): the pre-registered recommendation rule on a synthetic score table, and
re-encoding."""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

PYTHON_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PYTHON_DIR))

import compression_budget_study as x16  # noqa: E402

CUT, N = 1.0, 100                       # at most 1 flip of 100 is allowed (FLIP_LIMIT 1%)


def table(changes):
    """One row per image and condition; ``changes`` maps a condition to (bytes, flipped images, extra guard flag, extra rejection)."""
    stored = np.where(np.arange(N) < 50, 2.0, 0.5)
    stored[0] = 1.05                    # referred, next to the cut-off
    rows = []
    for name, _, _ in x16.CONDITIONS:
        size, flipped, flag, reject = changes.get(name, (400 * 1024, [], False, False))
        for k in range(N):
            score = stored[k] if k not in flipped else (0.98 if k == 0 else 2.0 - stored[k] + 0.5)
            rows.append({'k': k, 'source': 'odir', 'grade': 3 if k < 50 else 0, 'condition': name, 'bytes': size,
                         'e768_score': score, 'e768_refer': score >= CUT, 'legacy_score': score, 'legacy_refer': score >= CUT,
                         'iqs': 0.2 if reject and k == 60 else 0.8, 'guard_flag': bool(flag and k == 70)})
    return pd.DataFrame(rows)


SAMPLE = pd.DataFrame({'source': ['odir'] * N, 'path': [f'{k}.jpg' for k in range(N)]})
MANY = list(range(40, 46))


class RecommendationRule(unittest.TestCase):
    def test_smallest_eligible_setting_wins_and_ties_go_to_higher_quality(self):
        changes = {'q90': (300 * 1024, [0], False, False),              # eligible (1 flip)
                   'q75': (200 * 1024, [], True, False),                # no flip, but an extra guard flag
                   'q50': (150 * 1024, [], False, True),                # no flip, but an extra quality rejection
                   'q30': (100 * 1024, MANY, False, False),             # too many flips
                   '1024px_q90': (250 * 1024, [], False, False),        # eligible, smallest
                   '1024px_q75': (250 * 1024, [], False, False),        # same size, lower quality
                   '768px_q90': (260 * 1024, [3], False, False)}        # eligible, larger
        for name in ('1024px_q50', '768px_q75', '768px_q50'):
            changes[name] = (50 * 1024, MANY, False, False)
        summary = x16.summarize(table(changes), SAMPLE, CUT)
        self.assertEqual(summary['recommended'], '1024px_q90')
        self.assertEqual(summary['conditions']['q75']['extra_guard_flags'], 1)
        self.assertEqual(summary['conditions']['q50']['extra_quality_rejections'], 1)
        self.assertEqual(summary['conditions']['stored']['e768_flips'], 0)
        self.assertEqual(summary['capacity_what_if'], 'run for the recommended size only')
        q90 = summary['post_hoc']['q90']
        self.assertEqual((q90['referrals_lost'], q90['referrals_gained'], q90['flips_within_near_band']), (1, 0, 1))
        self.assertEqual(q90['max_stored_distance_from_cut'], 0.05)

    def test_no_recommendation_when_every_setting_changes_too_many_decisions(self):
        changes = {name: (100 * 1024, MANY, False, False) for name, _, _ in x16.CONDITIONS if name != 'stored'}
        summary = x16.summarize(table(changes), SAMPLE, CUT)
        self.assertIsNone(summary['recommended'])
        self.assertTrue(summary['capacity_what_if'].startswith('not run'))
        self.assertEqual(summary['post_hoc']['q90']['referrals_lost'], 6)


class Encoding(unittest.TestCase):
    def test_downscaling_keeps_the_aspect_ratio_and_skips_smaller_images(self):
        bgr = np.random.default_rng(0).integers(0, 255, (600, 900, 3), dtype=np.uint8)
        rgb, size = x16.encode(bgr, 768, 75)
        self.assertEqual(rgb.shape, (512, 768, 3))
        self.assertGreater(size, 0)
        same, _ = x16.encode(bgr, 1024, 90)
        self.assertEqual(same.shape, bgr.shape)


if __name__ == '__main__':
    unittest.main()
