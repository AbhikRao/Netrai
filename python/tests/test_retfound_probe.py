"""Tests for the RETFound probe analysis (synthetic features) and preprocessing/encoder plumbing."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import retfound_probe as rp


def synthetic(signal: float, seed=0, dim=32, n=250):
    rng = np.random.default_rng(seed)
    frames, grading, feats = [], [], []
    for site in 'abcd':
        grade = rng.integers(0, 5, n)
        positive = grade >= 2
        g = grade + rng.normal(0, 2.0, n)
        f = rng.normal(0, 1, (n, dim))
        f[:, :4] += positive[:, None] * signal
        frames.append(pd.DataFrame({'image_id': [f'{site}{i}' for i in range(n)], 'source': site, 'grade': grade,
                                    'group_id': [f'{site}{i}' for i in range(n)]}))
        grading.append(g); feats.append(f)
    return pd.concat(frames, ignore_index=True), np.concatenate(grading), np.vstack(feats)


class ProbeTests(unittest.TestCase):
    def test_informative_features_beat_a_noisy_grading_score(self):
        frame, g, f = synthetic(1.0)
        r = rp.analyze_sites(frame, g, f, c=0.05, replicates=50)
        self.assertTrue(r['rule_probe_alone']['success'])
        self.assertTrue(r['rule_stack']['success'])

    def test_uninformative_features_fail_both_rules(self):
        frame, g, f = synthetic(0.0)
        r = rp.analyze_sites(frame, g, f, c=0.05, replicates=50)
        self.assertFalse(r['rule_probe_alone']['success'])
        self.assertFalse(r['rule_stack']['success'])

    def test_target_site_labels_never_used_to_score_itself(self):
        frame, g, f = synthetic(1.0)
        # if the target site's rows leaked into its own probe, flipping its labels would not change its score
        flipped = frame.copy()
        flipped.loc[flipped.source == 'a', 'grade'] = 4 - flipped.loc[flipped.source == 'a', 'grade']
        a = rp.analyze_sites(frame, g, f, c=0.05, replicates=10)
        b = rp.analyze_sites(flipped, g, f, c=0.05, replicates=10)
        # site b's probe is fitted on a,c,d: flipping a's labels must change b's probe AUC
        self.assertNotAlmostEqual(a['sites']['b']['probe']['auc'], b['sites']['b']['probe']['auc'], places=3)

    def test_preprocess_shape_and_range(self):
        t = rp.preprocess(np.random.default_rng(0).integers(0, 255, (300, 280, 3), dtype=np.uint8))
        self.assertEqual(tuple(t.shape), (3, 224, 224))
        self.assertLess(float(t.abs().max()), 3.0)

    def test_encoder_loads_matching_checkpoint_and_rejects_mismatch(self):
        small = 'vit_tiny_patch16_224'
        reference = rp.build_encoder(None, small)
        with tempfile.TemporaryDirectory() as d:
            good = Path(d) / 'good.pth'
            torch.save({'model': reference.state_dict()}, good)
            rp.build_encoder(good, small)  # loads
            bad = Path(d) / 'bad.pth'
            torch.save({'model': {'blocks.0.attn.qkv.weight': torch.zeros(3, 3)}}, bad)
            with self.assertRaises(Exception):
                rp.build_encoder(bad, small)


if __name__ == '__main__':
    unittest.main()
