"""Local-only regression: recompute every locked report from its stored logits and labels.

For each published lock (rounds 2, 3 and 4) the whole cohort block of ``report.json`` --
metrics, bootstrap intervals and paired comparisons -- is recomputed with the current code and
compared with the stored report. Any difference means either the evaluation code changed
behaviour or a published number is wrong. The test is skipped when the lock directories are not
present (CI), and it only reads labels of cohorts that are already exposed.
"""
import json
import unittest
from pathlib import Path
from unittest import mock

import locked_evaluation as le

ROOT = le.ROOT
MI = ROOT / 'results/model_improvement'
LOCKS = [
    ('round2', MI / 'round2_eval', MI / 'round2_eval/variants.json', 2),
    ('round3', MI / 'round3_lock', MI / 'round3_eval/variants.json', 3),
    ('round4_v3', MI / 'round4_lock_v3', MI / 'round4_eval/variants.json', 4),
]


# Documented, benign post-lock changes of bound data artefacts (name -> reason). Anything else that
# differs from its lock-time hash fails the test.
KNOWN_POST_LOCK_CHANGES = {
    'round2': {
        'matlab/variant_R2_K3_freeze4_ddr_s17/netrai_bundle.json':
            'bundle manifest regenerated 2026-10-12 12:00 (commit 5f298c2), after the round-2 evaluation, '
            'when the MATLAB grading source was generalised to 256-1024 px inputs; the ONNX weights, '
            'config, policy and logits are byte-identical',
    },
}


def compare(test, expected, actual, path='root'):
    if isinstance(expected, dict):
        test.assertEqual(set(expected), set(actual), f'{path}: keys differ')
        for key in expected:
            compare(test, expected[key], actual[key], f'{path}.{key}')
    elif isinstance(expected, (list, tuple)):
        test.assertEqual(len(expected), len(actual), f'{path}: length differs')
        for index, (a, b) in enumerate(zip(expected, actual)):
            compare(test, a, b, f'{path}[{index}]')
    elif isinstance(expected, float):
        if expected != expected:  # NaN
            test.assertTrue(actual != actual, f'{path}: expected NaN')
        else:
            test.assertAlmostEqual(expected, actual, places=9, msg=path)
    else:
        test.assertEqual(expected, actual, path)


class PublishedReportRegression(unittest.TestCase):
    def check(self, name, lock_dir, variants_path, round_number):
        if not (lock_dir / 'report.json').is_file() or not variants_path.is_file():
            self.skipTest(f'{name}: lock artifacts are not available on this machine')
        lock = json.loads((lock_dir / 'lock.json').read_text())
        report = json.loads((lock_dir / 'report.json').read_text())
        bound = [p for p in lock['bindings'] if p.endswith('variants.json')]
        self.assertEqual(len(bound), 1)
        self.assertEqual(le.sha(ROOT / bound[0]), lock['bindings'][bound[0]],
                         f'{name}: variants file changed since the lock')
        # Data artefacts (logits, inventories, labels, policies, checkpoints, ONNX, variants) must still be
        # byte-identical. Code and plan files legitimately evolve after a lock (later rounds edit the
        # evaluator; post-lock addenda are appended to plans), so they are reported, not asserted.
        evolving = {'.py', '.md'}
        self.changed_code = []
        for path, digest in lock['bindings'].items():
            if (ROOT / path).suffix in evolving:
                if le.sha(ROOT / path) != digest:
                    self.changed_code.append(path)
                continue
            if path in KNOWN_POST_LOCK_CHANGES.get(name, {}):
                continue
            self.assertEqual(le.sha(ROOT / path), digest, f'{name}: {path} changed since the lock')
        config = json.loads(variants_path.read_text())
        setup = le.ROUNDS[round_number]
        args = mock.Mock(output=lock_dir)
        for cohort in setup['cohorts']:
            with self.subTest(round=name, cohort=cohort):
                recomputed = le.evaluate_cohort(cohort, args, config, lock, setup['cohorts'])
                compare(self, report['cohorts'][cohort], recomputed, f'{name}.{cohort}')

    def test_round2(self):
        self.check(*LOCKS[0])

    def test_round3(self):
        self.check(*LOCKS[1])

    def test_round4_v3(self):
        self.check(*LOCKS[2])


if __name__ == '__main__':
    unittest.main()
