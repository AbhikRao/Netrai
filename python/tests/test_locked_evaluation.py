"""Tests for the locked-evaluation code that produces every published fresh-test number.

Covers the label parsers, the metric functions and the freeze/evaluate guards on a
small synthetic cohort. Nothing here reads real labels.
"""
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import exposure
import locked_evaluation as le

ROOT = le.ROOT


class OdirGradeTests(unittest.TestCase):
    CASES = [
        ('normal fundus', 0),
        ('mild nonproliferative retinopathy', 1),
        ('mild non-proliferative retinopathy', 1),
        ('moderate non proliferative retinopathy', 2),
        ('severe nonproliferative retinopathy', 3),
        ('proliferative diabetic retinopathy', 4),
        ('diabetic retinopathy', 'unspecified'),
        ('cataract', 'other_disease_only'),
        ('glaucoma', 'other_disease_only'),
        ('hypertensive retinopathy', 'other_disease_only'),
        ('low image quality', 'ungradable'),
        ('lens dust', 'ungradable'),
        ('optic disk photographically invisible', 'ungradable'),
        ('no fundus image', 'ungradable'),
    ]

    def test_single_keywords(self):
        for keyword, expected in self.CASES:
            with self.subTest(keyword=keyword):
                self.assertEqual(le.odir_grade(keyword), expected)

    def test_combinations(self):
        # DR keyword wins over a co-listed normal/other finding; highest grade is kept.
        self.assertEqual(le.odir_grade('moderate non proliferative retinopathy，cataract'), 2)
        self.assertEqual(le.odir_grade('mild nonproliferative retinopathy, severe nonproliferative retinopathy'), 3)
        # Quality terms dominate every other keyword.
        self.assertEqual(le.odir_grade('moderate non proliferative retinopathy, lens dust'), 'ungradable')
        # Hypertensive terms are skipped, so a co-listed normal fundus still maps to 0.
        self.assertEqual(le.odir_grade('hypertensive retinopathy, normal fundus'), 0)
        # 'diabetic retinopathy' alone (no stage) makes the eye unusable for the graded endpoint.
        self.assertEqual(le.odir_grade('diabetic retinopathy, normal fundus'), 'unspecified')

    def test_non_proliferative_is_not_grade_four(self):
        self.assertEqual(le.odir_grade('non-proliferative something'), 'other_disease_only')


class Fdr26GradeTests(unittest.TestCase):
    def test_valid_spellings(self):
        for value, expected in [('DR0 - Pas de RD', 0), ('DR1 - Minime', 1), ('DR2 - Moderee', 2),
                                ('DR3 - Severe', 3), ('DR4 - Proliferante', 4), ('DR3', 3), ('dr 1', 1),
                                (2, 2), (2.0, 2), ('2', 2)]:
            with self.subTest(value=value):
                self.assertEqual(le.fdr26_grade(value), expected)

    def test_invalid_spellings(self):
        for value in ['', None, 'DR5 - x', '5', 'DR2-3', 'DR 12', 'x', float('nan')]:
            with self.subTest(value=value):
                self.assertIsNone(le.fdr26_grade(value))


def one_hot_logits(grades, referable=None):
    """Logits whose softmax is (almost) one-hot, so the expected-grade score equals the grade."""
    grades = np.asarray(grades)
    logits = np.full((len(grades), 6), -20.0)
    logits[np.arange(len(grades)), grades] = 20.0
    referable = grades >= 2 if referable is None else np.asarray(referable)
    logits[:, 5] = np.where(referable, 10.0, -10.0)
    return logits


class MetricsTests(unittest.TestCase):
    THRESHOLDS = np.array([0.5, 1.5, 2.5, 3.5])

    def test_perfect_predictions(self):
        grades = np.array([0, 0, 1, 2, 3, 4, 2, 0])
        result, positive = le.metrics(one_hot_logits(grades), grades, self.THRESHOLDS, 0.5)
        self.assertEqual((result['tp'], result['fn'], result['tn'], result['fp']), (4, 0, 4, 0))
        self.assertAlmostEqual(result['sensitivity'], 1.0)
        self.assertAlmostEqual(result['specificity'], 1.0)
        self.assertAlmostEqual(result['qwk'], 1.0)
        self.assertEqual(result['per_grade_support'], [3, 1, 2, 1, 1])
        self.assertEqual(result['per_grade_recall'], [1.0] * 5)
        self.assertEqual(result['head_disagreement_rate'], 0.0)
        self.assertEqual(positive.tolist(), (grades >= 2).tolist())

    def test_known_confusion_counts_and_auc(self):
        truth = np.array([0, 0, 0, 0, 1, 2, 2, 2, 3, 4])
        predicted = np.array([0, 0, 2, 0, 0, 2, 1, 3, 3, 4])  # one FP (index 2), one FN (index 6)
        result, _ = le.metrics(one_hot_logits(predicted), truth, self.THRESHOLDS, 0.5)
        self.assertEqual((result['tp'], result['fn'], result['tn'], result['fp']), (4, 1, 4, 1))
        self.assertAlmostEqual(result['sensitivity'], 4 / 5)
        self.assertAlmostEqual(result['specificity'], 4 / 5)
        score = np.array([float(p) for p in predicted])
        self.assertAlmostEqual(result['auc_expected_grade_score'], roc_auc_score(truth >= 2, score), places=6)
        self.assertEqual(result['per_grade_support'], [4, 1, 3, 1, 1])
        self.assertAlmostEqual(result['per_grade_recall'][1], 0.0)  # the only grade-1 eye predicted 0

    def test_sensitivity_at_85_specificity_uses_ranking_only(self):
        rng = np.random.default_rng(0)
        grades = rng.integers(0, 5, 400)
        logits = one_hot_logits(grades)
        logits[:, :5] += rng.normal(0, 4, (400, 5))  # noisy scores
        high, _ = le.metrics(logits, grades, np.array([0.1, 0.2, 0.3, 0.4]), 0.5)
        low, _ = le.metrics(logits, grades, np.array([0.5, 1.5, 2.5, 3.5]), 0.5)
        self.assertEqual(high['sensitivity_at_85_specificity'], low['sensitivity_at_85_specificity'])
        self.assertEqual(high['auc_expected_grade_score'], low['auc_expected_grade_score'])

    def test_head_disagreement_rate(self):
        grades = np.array([0, 0, 2, 2])
        referable = np.array([False, True, True, False])  # two disagreements with the grade policy
        with np.errstate(all='ignore'):  # grades 1, 3, 4 are absent: per-grade recall is NaN by design
            result, _ = le.metrics(one_hot_logits(grades, referable), grades, self.THRESHOLDS, 0.5)
        self.assertAlmostEqual(result['head_disagreement_rate'], 0.5)

    def test_thresholds_for_default_and_policy(self):
        thresholds, head = le.thresholds_for({'id': 'A_legacy'})
        self.assertEqual(len(thresholds), 4)
        self.assertEqual(head, 0.5)
        with tempfile.TemporaryDirectory(dir=ROOT / 'results') as tmp:
            policy = Path(tmp) / 'policy.json'
            policy.write_text(json.dumps({'thresholds': [0.6, 1.1, 2.1, 3.1], 'head_threshold': 0.2}))
            thresholds, head = le.thresholds_for({'id': 'x', 'policy': str(policy.relative_to(ROOT))})
        self.assertEqual(thresholds.tolist(), [0.6, 1.1, 2.1, 3.1])
        self.assertEqual(head, 0.2)


class FreezeEvaluateGuardTests(unittest.TestCase):
    """Freeze/evaluate on a tiny synthetic cohort with the label reader patched out."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT / 'results')
        self.addCleanup(self.temp.cleanup)
        base = Path(self.temp.name)
        self.images = base / 'images'
        self.images.mkdir()
        self.names = [f'img{i:03d}.png' for i in range(24)]
        for name in self.names:
            (self.images / name).write_bytes(name.encode() * 8)
        self.labels_file = base / 'labels.txt'
        self.labels_file.write_text('placeholder label file')
        self.protocol = base / 'protocol.md'
        self.protocol.write_text('protocol')
        self.variants = base / 'variants.json'
        self.variants.write_text(json.dumps({
            'primary': 'A_legacy', 'reference': 'A_legacy', 'bootstrap_seed': 11, 'bootstrap_replicates': 50,
            'variants': [{'id': 'A_legacy', 'kind': 'legacy'}]}))
        self.output = base / 'lock'
        self.rng = np.random.default_rng(5)
        self.grades = np.array([0, 1, 2, 3, 4, 0] * 4)
        self.logits = one_hot_logits(self.grades) + self.rng.normal(0, 0.01, (24, 6))
        self.frame = pd.DataFrame({'image_id': self.names, 'exam_id': [f'e{i // 2}' for i in range(24)]})
        self.setup = {'cohorts': {'synth': {'images': self.images, 'labels': self.labels_file, 'cluster': 'exam'}},
                      'labels': [self.labels_file], 'protocol': self.protocol, 'tests': {'synth': 'synthetic'},
                      'exposure': 'synthetic'}
        self.registry = base / 'exposure_registry.json'
        self.registry.write_text(json.dumps({'schema_version': 1, 'updated': '2026-01-01', 'cohorts': {
            'synth': {'kind': 'grading', 'status': 'reserved'}}}))
        patches = [mock.patch.object(exposure, 'REGISTRY_PATH', self.registry),
                   mock.patch.dict(le.ROUNDS, {9: self.setup}),
                   mock.patch.object(le, 'inventory', side_effect=lambda cohort: self.frame.copy()),
                   mock.patch.object(le, 'logits_for', side_effect=self.fake_logits),
                   mock.patch.object(le, 'cohort_labels', side_effect=self.fake_labels)]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)

    def fake_logits(self, spec, frame, folder=None, **kwargs):
        index = {name: i for i, name in enumerate(self.names)}
        return self.logits[[index[n] for n in frame.image_id]].copy()

    def fake_labels(self, cohort, frame):
        return frame, np.ones(len(frame), bool), self.grades.astype(float)

    def args(self, round_number=9, development=False):
        return mock.Mock(variants=self.variants, output=self.output, protocol=self.protocol, threads=1,
                         round=round_number, development=development)

    def test_freeze_records_no_labels_and_binds_artifacts(self):
        le.freeze(self.args())
        lock = json.loads((self.output / 'lock.json').read_text())
        self.assertIs(lock['labels_read'], False)
        self.assertEqual(lock['round'], 9)
        bound = set(lock['bindings'])
        self.assertTrue(any(p.endswith('labels.txt') for p in bound))
        self.assertTrue(any(p.endswith('locked_evaluation.py') for p in bound))
        self.assertTrue(any(p.endswith('logits_synth_A_legacy.npy') for p in bound))

    def test_freeze_refuses_to_overwrite(self):
        le.freeze(self.args())
        with self.assertRaises(FileExistsError):
            le.freeze(self.args())

    def test_evaluate_runs_once_then_refuses(self):
        le.freeze(self.args())
        le.evaluate(self.args())
        report = json.loads((self.output / 'report.json').read_text())
        self.assertEqual(report['schema_version'], 2)
        self.assertIn('synth', report['cohorts'])
        with self.assertRaises(FileExistsError):
            le.evaluate(self.args())

    def test_evaluate_detects_a_one_byte_change_in_any_bound_file(self):
        le.freeze(self.args())
        with open(self.labels_file, 'ab') as handle:
            handle.write(b'!')
        with self.assertRaisesRegex(ValueError, 'Locked artifact changed'):
            le.evaluate(self.args())
        self.assertFalse((self.output / 'report.json').exists())

    def test_evaluate_detects_modified_logits(self):
        le.freeze(self.args())
        path = self.output / 'logits_synth_A_legacy.npy'
        values = np.load(path)
        values[0, 0] += 1e-9
        np.save(path, values)
        with self.assertRaisesRegex(ValueError, 'Locked artifact changed'):
            le.evaluate(self.args())

    def test_precomputed_logits_must_pass_the_cpu_spot_check(self):
        self.output.mkdir()
        np.save(self.output / 'logits_synth_A_legacy.npy', self.logits + 1e-2)  # beyond 5e-3
        with self.assertRaisesRegex(ValueError, 'spot check'):
            le.freeze(self.args())

    def test_precomputed_logits_within_tolerance_are_accepted_and_recorded(self):
        self.output.mkdir()
        np.save(self.output / 'logits_synth_A_legacy.npy', self.logits + 1e-3)
        le.freeze(self.args())
        lock = json.loads((self.output / 'lock.json').read_text())
        self.assertAlmostEqual(lock['precomputed_logits_cpu_spot_check_max_abs']['logits_synth_A_legacy.npy'], 1e-3, places=6)

    def test_precomputed_logits_wrong_shape_rejected(self):
        self.output.mkdir()
        np.save(self.output / 'logits_synth_A_legacy.npy', self.logits[:10])
        with self.assertRaisesRegex(ValueError, 'shape'):
            le.freeze(self.args())

    def test_spot_check_covers_the_last_row(self):
        self.output.mkdir()
        corrupted = self.logits.copy()
        corrupted[-1] += 0.5  # only the final image differs
        np.save(self.output / 'logits_synth_A_legacy.npy', corrupted)
        with self.assertRaisesRegex(ValueError, 'spot check'):
            le.freeze(self.args())

    def test_bootstrap_is_deterministic_for_a_seed(self):
        le.freeze(self.args())
        args = self.args()
        lock = json.loads((self.output / 'lock.json').read_text())
        config = json.loads(self.variants.read_text())
        first = le.evaluate_cohort('synth', args, config, lock, self.setup['cohorts'])
        second = le.evaluate_cohort('synth', args, config, lock, self.setup['cohorts'])
        self.assertEqual(first, second)
        sensitivity = first['results']['A_legacy']['sensitivity']
        low, high = first['results']['A_legacy']['sensitivity_ci95']
        self.assertLessEqual(low, sensitivity + 1e-12)
        self.assertGreaterEqual(high, sensitivity - 1e-12)


if __name__ == '__main__':
    unittest.main()
