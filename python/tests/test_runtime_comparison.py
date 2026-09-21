import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


PYTHON_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PYTHON_DIR))

from compare_runtime_outputs import compare_runtime_outputs, write_comparison


def _python_row(image_id='case', grade=2, score=2.56):
    row = {
        'id_code': image_id,
        'status': 'success',
        'true_grade': 3,
        'predicted_grade': grade,
        'continuous_score': score,
        'confidence': 0.80,
        'referable_probability': 0.90,
        'calibrated_referable_probability': 0.92,
        'referable_head_probability': 0.91,
        'referable_head_prediction': True,
        'head_disagreement': False,
        'triage_action': 'refer',
        'iqs': 0.50,
    }
    for class_index in range(5):
        row[f'prob_grade_{class_index}'] = 0.2
        row[f'calibrated_prob_grade_{class_index}'] = 0.2
    return row


def _matlab_row(image_id='case', grade=2, score=2.56):
    row = {
        'id_code': image_id,
        'status': 'success',
        'true_grade': 3,
        'predicted_grade': grade,
        'continuous_score': score,
        'confidence': 0.80,
        'referable_probability': 0.90,
        'calibrated_referable_probability': 0.92,
        'referable_head_probability': 0.91,
        'referable_head_prediction': 1,
        'head_disagreement': 0,
        'triage_action': 'refer',
        'iqs': 0.50,
    }
    for class_index in range(5):
        row[f'raw_prob_grade_{class_index}'] = 0.2
        row[f'prob_grade_{class_index}'] = 0.2
    return row


class RuntimeComparisonTests(unittest.TestCase):
    def _write_frames(self, python_rows, matlab_rows, directory):
        python_path = Path(directory) / 'python.csv'
        matlab_path = Path(directory) / 'matlab.csv'
        pd.DataFrame(python_rows).to_csv(python_path, index=False)
        pd.DataFrame(matlab_rows).to_csv(matlab_path, index=False)
        return python_path, matlab_path

    def test_exact_outputs_pass_all_parity_gates(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self._write_frames([_python_row()], [_matlab_row()], directory)
            summary, details = compare_runtime_outputs(
                *paths, thresholds=np.array([0.5, 1.5, 2.5, 3.5]))
            self.assertTrue(summary['gates']['screening_decision_parity_pass'])
            self.assertTrue(summary['gates']['strict_five_grade_parity_pass'])
            self.assertTrue(summary['gates']['probability_tolerance_pass'])
            self.assertFalse(details.loc[0, 'threshold_edge_grade_mismatch'])

    def test_threshold_edge_grade_difference_preserves_screening_parity(self):
        python_row = _python_row(grade=3, score=2.56)
        matlab_row = _matlab_row(grade=2, score=2.53)
        # Confidence names the selected class, so it is not a like-for-like
        # probability when the selected grades differ.
        python_row['confidence'] = 0.99
        matlab_row['confidence'] = 0.10
        with tempfile.TemporaryDirectory() as directory:
            paths = self._write_frames([python_row], [matlab_row], directory)
            summary, details = compare_runtime_outputs(
                *paths, thresholds=np.array([0.5, 1.5, 2.55, 3.5]),
                threshold_edge_tolerance=0.05)
            self.assertTrue(summary['gates']['screening_decision_parity_pass'])
            self.assertFalse(summary['gates']['strict_five_grade_parity_pass'])
            self.assertTrue(summary['gates']['probability_tolerance_pass'])
            self.assertEqual(summary['threshold_edge_grade_mismatches'], 1)
            self.assertTrue(details.loc[0, 'threshold_edge_grade_mismatch'])

    def test_missing_matlab_id_fails_screening_parity(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self._write_frames(
                [_python_row('case')], [_matlab_row('missing')], directory)
            with self.assertRaisesRegex(ValueError, 'no image ids in common'):
                compare_runtime_outputs(
                    *paths, thresholds=np.array([0.5, 1.5, 2.5, 3.5]))

    def test_outputs_are_written_as_csv_and_strict_json(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self._write_frames([_python_row()], [_matlab_row()], directory)
            summary, details = compare_runtime_outputs(
                *paths, thresholds=np.array([0.5, 1.5, 2.5, 3.5]))
            csv_path, json_path = write_comparison(
                summary, details, Path(directory) / 'comparison')
            self.assertTrue(csv_path.is_file())
            self.assertIn('"screening_decision_parity_pass": true',
                          json_path.read_text(encoding='utf-8'))


if __name__ == '__main__':
    unittest.main()
