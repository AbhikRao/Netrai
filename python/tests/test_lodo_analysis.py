"""Tests for the leave-one-dataset-out analysis of train_v6 run directories."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import lodo_analysis as la
from netrai_eval import report as er


def fake_run(base: Path, name: str, seed: int, shift_heldout: float = 0.0, noise: float = 1.0):
    """A finished-run directory with selection rows from two sources and a held-out source 'h'."""
    rng = np.random.default_rng(seed)
    run = base / name
    run.mkdir()

    def make(sources, n_each, offset):
        rows, outputs = [], []
        for source in sources:
            grades = rng.choice(5, n_each, p=[.45, .15, .2, .1, .1])
            logits = rng.normal(0, noise, (n_each, 5))
            logits[np.arange(n_each), grades] += 2.5
            logits[:, 2:] += offset  # raise the referable-grade logits only (a uniform shift would cancel in the softmax)
            referral = (grades >= 2) * 3.0 - 1.5 + rng.normal(0, noise, n_each) + offset
            outputs.append(np.concatenate([logits, referral[:, None]], 1))
            rows.append(pd.DataFrame({'image_id': [f'{source}{k}' for k in range(n_each)], 'grade': grades,
                                      'source': source, 'group_id': [f'{source}g{k // 2}' for k in range(n_each)]}))
        return pd.concat(rows, ignore_index=True), np.concatenate(outputs)

    selection, out_s = make(['a', 'b'], 600, 0.0)
    heldout, out_h = make(['h'], 400, shift_heldout)
    selection.to_csv(run / 'validation_rows.csv', index=False)
    heldout.to_csv(run / 'heldout_rows.csv', index=False)
    np.save(run / 'best_validation_outputs.npy', out_s)
    np.save(run / 'best_heldout_outputs.npy', out_h)
    (run / 'run_info.json').write_text(json.dumps({'config': {'output': name}}))
    return run


class AnalyzeTests(unittest.TestCase):
    def test_report_is_tier_TL_and_policy_comes_from_selection_rows_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = fake_run(Path(tmp), 'run1', seed=1)
            report = la.analyze_run(run, replicates=30, seed=1)
            er.validate_report(report)
            self.assertEqual(report['tier'], 'TL')
            self.assertEqual(report['cohort'], 'h')
            metrics = report['metrics']
            self.assertEqual(len(metrics['policy_fitted_on_selection_rows']['thresholds']), 4)
            held = metrics['per_source']['h']
            self.assertEqual(held['n_images'], 400)
            self.assertIn('auc', held['intervals'])
            self.assertIn('a', metrics['selection_auc_by_source_TD'])
            self.assertNotIn('h', metrics['selection_auc_by_source_TD'])

    def test_policy_does_not_change_when_only_the_heldout_outputs_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = fake_run(Path(tmp), 'run1', seed=2)
            before = la.analyze_run(run, replicates=10, seed=1)['metrics']['policy_fitted_on_selection_rows']
            np.save(run / 'best_heldout_outputs.npy', np.load(run / 'best_heldout_outputs.npy') + 5.0)
            after = la.analyze_run(run, replicates=10, seed=1)['metrics']['policy_fitted_on_selection_rows']
            self.assertEqual(before, after)

    def test_a_shifted_heldout_source_shows_threshold_transfer_loss(self):
        with tempfile.TemporaryDirectory() as tmp:
            same = la.analyze_run(fake_run(Path(tmp), 'same', 3, shift_heldout=0.0), 10, 1)['metrics']['per_source']['h']
            shifted = la.analyze_run(fake_run(Path(tmp), 'shifted', 3, shift_heldout=1.0), 10, 1)['metrics']['per_source']['h']
            drop_auc = same['ranking']['auc'] - shifted['ranking']['auc']
            drop_spec = (same['referral_grade_policy']['specificity']
                         - shifted['referral_grade_policy']['specificity'])
            # a score offset of the held-out source moves the fixed operating point much more than the ranking
            self.assertGreater(drop_spec, 0.05)
            self.assertGreater(drop_spec, 2 * max(drop_auc, 0.0))

    def test_unfinished_or_overlapping_runs_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            run = fake_run(base, 'run1', 1)
            (run / 'best_heldout_outputs.npy').unlink()
            with self.assertRaises(FileNotFoundError):
                la.load_run(run)
            run2 = fake_run(base, 'run2', 1)
            rows = pd.read_csv(run2 / 'heldout_rows.csv')
            rows.loc[0, 'image_id'] = pd.read_csv(run2 / 'validation_rows.csv').image_id[0]
            rows.to_csv(run2 / 'heldout_rows.csv', index=False)
            with self.assertRaisesRegex(ValueError, 'overlap'):
                la.load_run(run2)
            run3 = fake_run(base, 'run3', 1)
            np.save(run3 / 'best_heldout_outputs.npy', np.zeros((5, 6)))
            with self.assertRaisesRegex(ValueError, 'do not match'):
                la.load_run(run3)


class CompareTests(unittest.TestCase):
    def test_paired_comparison_detects_a_better_run_and_is_zero_for_identical_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            good = fake_run(base, 'good', seed=4, noise=0.6)
            bad = fake_run(base, 'bad', seed=4, noise=0.6)
            rng = np.random.default_rng(0)
            noisy = np.load(bad / 'best_heldout_outputs.npy')
            noisy[:, :5] += rng.normal(0, 2.0, noisy[:, :5].shape)
            np.save(bad / 'best_heldout_outputs.npy', noisy)
            result = la.compare(good, bad, replicates=60, seed=2)
            self.assertGreater(result['auc']['difference'], 0.03)
            same = la.compare(good, good, replicates=20, seed=2)
            self.assertEqual(same['auc']['difference'], 0.0)
            self.assertEqual(same['auc']['difference_ci95'], [0.0, 0.0])

    def test_runs_must_hold_out_the_same_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            a = fake_run(base, 'a', 1)
            b = fake_run(base, 'b', 1)
            rows = pd.read_csv(b / 'heldout_rows.csv').iloc[::-1].reset_index(drop=True)
            rows.to_csv(b / 'heldout_rows.csv', index=False)
            with self.assertRaisesRegex(ValueError, 'same images'):
                la.compare(a, b)


if __name__ == '__main__':
    unittest.main()
