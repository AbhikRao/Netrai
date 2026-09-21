import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


PYTHON_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PYTHON_DIR))

from models.quality_model import quality_decisions, quality_input
from train_quality_model import (
    fit_temperature, grouped_split, patient_bootstrap, patient_id,
    retention_metrics, select_reject_threshold, wilson_lower_bound,
)


class LearnedQualityTests(unittest.TestCase):
    def test_preprocess_constant_rgb_has_expected_shape_and_values(self):
        image = np.full((17, 29, 3), [10, 20, 30], dtype=np.uint8)
        result = quality_input(image)
        self.assertEqual(result.shape, (3, 224, 224))
        self.assertEqual(result.dtype, np.float32)
        expected = ((np.array([10, 20, 30], np.float32) / 255
                     - np.array([.485, .456, .406], np.float32))
                    / np.array([.229, .224, .225], np.float32))
        np.testing.assert_allclose(result[:, 100, 100], expected, atol=1e-6)

    def test_quality_decision_threshold_overrides_good_usable_choice(self):
        probabilities = np.array([[.7, .2, .1], [.2, .5, .3], [.4, .1, .5]])
        np.testing.assert_array_equal(
            quality_decisions(probabilities, .4), [0, 1, 2])

    def test_patient_identity_and_grouped_split_prevent_eye_leakage(self):
        self.assertEqual(patient_id('123_left.jpeg'), '123')
        train = pd.DataFrame({
            'image': [f'{i}_{eye}.jpeg' for i in range(15) for eye in ('left', 'right')],
            'quality': [i % 3 for i in range(15) for _ in range(2)],
        })
        train['patient_id'] = train.image.map(patient_id)
        test = pd.DataFrame({'image': ['99_left.jpeg'], 'quality': [0], 'patient_id': ['99']})
        fit, selection, calibration = grouped_split(train, test, 42)
        partitions = [set(train.iloc[index].patient_id)
                      for index in (fit, selection, calibration)]
        self.assertFalse(partitions[0] & partitions[1])
        self.assertFalse(partitions[0] & partitions[2])
        self.assertFalse(partitions[1] & partitions[2])

    def test_temperature_and_reject_threshold_are_fitted_separately(self):
        logits = np.array([[4, 0, 0], [0, 4, 0], [0, 0, 4]] * 4, dtype=float)
        labels = np.array([0, 1, 2] * 4)
        temperature = fit_temperature(logits, labels)
        self.assertGreater(temperature, 0)
        probabilities = np.exp(logits / temperature)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        selected, trials = select_reject_threshold(probabilities, labels)
        self.assertTrue(0 < selected['reject_threshold'] < 1)
        self.assertIn('reject_recall_one_sided_95_wilson_lower', selected)
        self.assertEqual(len(trials), 29)

    def test_wilson_lower_bound_is_conservative(self):
        self.assertLess(wilson_lower_bound(80, 100), .80)
        self.assertGreater(wilson_lower_bound(90, 100), .80)

    def test_patient_bootstrap_and_retention_expose_safety_endpoints(self):
        frame = pd.DataFrame({
            'quality': [0, 1, 2, 2], 'patient_id': ['a', 'b', 'c', 'd'],
            'DR_grade': [0, 1, 2, 4],
        })
        predictions = np.array([0, 2, 2, 1])
        retention = retention_metrics(frame, predictions)
        self.assertEqual(retention['false_reject_rate_among_good_or_usable'], .5)
        intervals = patient_bootstrap(
            frame.quality, predictions, frame.patient_id, seed=2, replicates=20)
        self.assertEqual(intervals['replicates'], 20)
        self.assertEqual(len(intervals['macro_f1_95_ci']), 2)


if __name__ == '__main__':
    unittest.main()
