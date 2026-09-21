import sys
import unittest
from pathlib import Path

import numpy as np


PYTHON_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PYTHON_DIR))

from compare_quality_baselines import build_models, metrics


class QualityBaselineTests(unittest.TestCase):
    def test_metrics_cover_all_three_classes(self):
        summary, matrix = metrics(
            np.array([0, 1, 2, 0, 1, 2]),
            np.array([0, 1, 2, 0, 1, 2]))
        self.assertEqual(summary['macro_f1'], 1.0)
        self.assertEqual(summary['per_class_recall']['reject'], 1.0)
        self.assertEqual(matrix.shape, (3, 3))

    def test_baseline_models_are_reproducibly_named(self):
        self.assertEqual(set(build_models(42)), {
            'balanced_logistic_regression',
            'balanced_random_forest',
            'balanced_extra_trees',
            'balanced_hist_gradient_boosting',
        })


if __name__ == '__main__':
    unittest.main()
