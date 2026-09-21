import sys
import tempfile
import unittest
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


PYTHON_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PYTHON_DIR))

from calibrate_quality import _write_cache_atomic, extract_features, factor_diagnostics
from audit_quality_robustness import apply_corruption, summarize
from utils.quality_calibration import load_iqs_thresholds


class QualityCalibrationRunnerTests(unittest.TestCase):
    def _fixture(self, directory):
        root = Path(directory)
        image_dir = root / 'images'
        image_dir.mkdir()
        labels = pd.DataFrame({
            'image': ['a.jpeg', 'b.jpeg'],
            'quality': [0, 2],
        })
        labels_path = root / 'labels.csv'
        labels.to_csv(labels_path, index=False)
        cv2.imwrite(str(image_dir / 'a.jpeg'), np.full((24, 32, 3), 120, np.uint8))
        cv2.imwrite(str(image_dir / 'b.jpeg'), np.full((24, 32, 3), 60, np.uint8))
        return labels_path, image_dir, root / 'features.csv'

    def test_feature_cache_is_complete_and_label_ordered(self):
        with tempfile.TemporaryDirectory() as directory:
            labels_path, image_dir, cache_path = self._fixture(directory)
            result = extract_features(
                labels_path, image_dir, cache_path,
                workers=1, checkpoint_every=1)
            self.assertEqual(result['image'].tolist(), ['a.jpeg', 'b.jpeg'])
            self.assertEqual(result['status'].tolist(), ['success', 'success'])
            self.assertEqual(len(pd.read_csv(cache_path)), 2)

    def test_partial_cache_is_resumed_instead_of_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            labels_path, image_dir, cache_path = self._fixture(directory)
            first = extract_features(
                labels_path, image_dir, cache_path,
                workers=1, checkpoint_every=1)
            first.iloc[:1].to_csv(cache_path, index=False)
            resumed = extract_features(
                labels_path, image_dir, cache_path,
                workers=1, checkpoint_every=1)
            self.assertEqual(len(resumed), 2)
            self.assertEqual(resumed['image'].tolist(), ['a.jpeg', 'b.jpeg'])

    def test_factor_diagnostics_state_global_label_limitation(self):
        rows = []
        for quality, base in ((0, 0.9), (1, 0.5), (2, 0.1)):
            for offset in (0.0, 0.01):
                rows.append({
                    'quality': quality, 'status': 'success',
                    'iqs': base + offset, 'focus': base + offset,
                    'illumination': base + offset, 'fov': base + offset,
                })
        diagnostics = factor_diagnostics(pd.DataFrame(rows))
        self.assertIn('no factor-specific', diagnostics['scope'])
        self.assertEqual(
            diagnostics['features']['focus']['reject_vs_rest_auc_low_score'], 1.0)

    def test_quality_threshold_artifact_validation(self):
        artifact = {
            'schema_version': 1,
            'status': 'eyeq_calibrated_research_thresholds',
            'deployment_status': 'deployed',
            'reject_threshold': 0.2,
            'good_threshold': 0.7,
            'train_metrics': {},
            'test_metrics': {},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'quality.json'
            path.write_text(json.dumps(artifact), encoding='utf-8')
            reject, good, loaded = load_iqs_thresholds(path)
            self.assertEqual((reject, good), (0.2, 0.7))
            self.assertEqual(loaded['schema_version'], 1)

            artifact['deployment_status'] = 'not_deployed'
            path.write_text(json.dumps(artifact), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'evidence-only'):
                load_iqs_thresholds(path)
            reject, good, _ = load_iqs_thresholds(
                path, allow_evidence_only=True)
            self.assertEqual((reject, good), (0.2, 0.7))

            artifact['good_threshold'] = 0.1
            path.write_text(json.dumps(artifact), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, '0 <= reject'):
                load_iqs_thresholds(path, allow_evidence_only=True)

    def test_quality_corruptions_preserve_shape_and_audit_pairs(self):
        image = np.tile(np.arange(64, dtype=np.uint8), (48, 1))
        image = np.dstack([image, image, image])
        rows = []
        for name in ('baseline', 'defocus_blur', 'low_light',
                     'uneven_illumination', 'fov_misalignment', 'jpeg_q25'):
            result = apply_corruption(image, name)
            self.assertEqual(result.shape, image.shape)
            rows.append({
                'image': 'fixture', 'corruption': name,
                'iqs': 0.8 if name == 'baseline' else 0.5,
                'focus': 0.8 if name == 'baseline' else 0.5,
                'illumination': 0.8 if name == 'baseline' else 0.5,
                'fov': 0.8 if name == 'baseline' else 0.5,
                'predicted_quality': 'good' if name == 'baseline' else 'usable',
                'recapture_feedback': '',
            })
        summary = summarize(pd.DataFrame(rows))
        self.assertAlmostEqual(
            summary['defocus_blur']['median_iqs_change_from_baseline'], -0.3)

    def test_atomic_cache_writer_leaves_no_temporary_file(self):
        with tempfile.TemporaryDirectory() as directory:
            cache_path = Path(directory) / 'features.csv'
            frame = pd.DataFrame({'image': ['a'], 'quality': [0]})
            _write_cache_atomic(frame, cache_path)
            self.assertEqual(pd.read_csv(cache_path).to_dict('records'), [
                {'image': 'a', 'quality': 0},
            ])
            self.assertFalse((Path(directory) / '.features.csv.tmp').exists())


if __name__ == '__main__':
    unittest.main()
