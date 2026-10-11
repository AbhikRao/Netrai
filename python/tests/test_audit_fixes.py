import sys
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.losses import FocalLoss
from modules.quality import assess_image_quality
from modules.explainability import generate_evidence_checklist, assess_gradcam_quality
from modules.segmentation import extract_clinical_features, localize_fovea
from utils.preprocessing import preprocess_for_model, IMAGENET_MEAN, IMAGENET_STD
from utils.validation import grades, probabilities
from evaluate_ma_froc import match_candidates
from evaluate_lesion_masks import load_union_mask, restore_native_mask, metrics
import batch_validate
import train_grading
from utils.artifacts import sha256_file


class AuditFixTests(unittest.TestCase):
    def test_weighted_smoothing_changes_loss_and_gradient(self):
        logits = torch.zeros(2, 5, requires_grad=True)
        labels = torch.tensor([0, 4])
        loss_a = FocalLoss(torch.ones(5), label_smoothing=0.1)(logits, labels)
        grad_a = torch.autograd.grad(loss_a, logits, retain_graph=True)[0]
        loss_b = FocalLoss(torch.tensor([1., 1., 1., 1., 10.]), label_smoothing=0.1)(logits, labels)
        grad_b = torch.autograd.grad(loss_b, logits)[0]
        self.assertGreater(loss_b.item(), loss_a.item())
        self.assertFalse(torch.allclose(grad_a, grad_b))

    def test_gamma_zero_matches_weighted_smoothed_ce(self):
        logits = torch.tensor([[1., 0., -1., 2., 0.]])
        target = torch.tensor([3])
        weight = torch.tensor([1., 2., 3., 4., 5.])
        actual = FocalLoss(weight, gamma=0, label_smoothing=0.1)(logits, target)
        expected = torch.nn.functional.cross_entropy(logits, target, weight=weight,
                    label_smoothing=0.1, reduction='none').mean()
        torch.testing.assert_close(actual, expected)

    def test_linear_contract_matches_training_operations(self):
        image = np.random.default_rng(4).integers(8, 256, (89, 107, 3), dtype=np.uint8)
        resized = cv2.resize(image, (512, 512))
        enhanced = cv2.addWeighted(resized, 4, cv2.GaussianBlur(resized, (0, 0), 10), -4, 128)
        expected = (enhanced.astype(np.float32)/255 - IMAGENET_MEAN)/IMAGENET_STD
        np.testing.assert_array_equal(expected, preprocess_for_model(image, version='training-linear-v2'))

    def test_invalid_evidence_rejected(self):
        for labels in ([0.1], [5], [np.nan]):
            with self.assertRaises(ValueError):
                grades(labels)
        with self.assertRaises(ValueError):
            probabilities([[0.1] * 5])

    def test_constant_image_rejected(self):
        iqs, metrics = assess_image_quality(np.full((64, 64, 3), 127, np.uint8))
        self.assertEqual(iqs, 0)
        self.assertTrue(metrics['invalid_input'])

    def test_missing_anatomy_never_means_absent_or_clear(self):
        rows = generate_evidence_checklist({}, 0, np.ones(5)/5)
        self.assertTrue(all(row['status'] in ('UNKNOWN', 'UNVERIFIED') for row in rows))
        self.assertNotIn('clear', str(rows).lower())
        self.assertTrue(assess_gradcam_quality(np.zeros((32, 32)))['shortcut_flag'])

    def test_hemorrhage_quadrants_are_not_copies(self):
        mask = np.zeros((20, 20), np.uint8)
        mask[10:, 10:] = 255
        vector = extract_clinical_features({'hem_mask': mask})
        np.testing.assert_array_equal(vector[4:8], [1, 0, 0, 0])

    def test_fovea_does_not_localize_in_black_padding(self):
        self.assertIsNone(localize_fovea(np.zeros((64, 64), np.uint8), (10, 32), 5))

    def test_crowded_matching_is_label_order_invariant(self):
        truth = np.zeros((32, 32), np.uint8)
        truth[10, 10] = truth[10, 14] = 1
        candidates = [(12, 10), (10, 10)]
        self.assertEqual(match_candidates(candidates, [2, 1], truth, 0, tolerance=2), (2, 0, 2))
        count, labels = cv2.connectedComponents(truth, 8)
        swapped = np.where(labels > 0, 3-labels, 0)
        self.assertEqual(match_candidates(candidates, [2, 1], truth, 0, tolerance=2,
                                         labeled=(count, swapped)), (2, 0, 2))

    def test_native_annotation_retains_single_pixel_lesion(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            truth = np.zeros((1024, 2048), np.uint8)
            truth[301, 901] = 255
            cv2.imwrite(str(base/'mask.png'), truth)
            loaded = load_union_mask(base, {'ma_mask': 'mask.png'}, ('ma_mask',))
            self.assertEqual(loaded.shape, truth.shape)
            self.assertEqual(loaded.sum(), 1)
            restored = restore_native_mask(np.zeros((512, 512), bool), truth.shape)
            result = metrics(restored, loaded)
            self.assertEqual(result['lesion_fn'], 1)
            self.assertTrue(np.isnan(result['lesion_precision']))

    def test_batch_isolates_failure_and_resumes_without_rerunning(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            for index in range(3):
                cv2.imwrite(str(base/f'{index}.png'), np.full((32, 32, 3), index+10, np.uint8))
            pd.DataFrame({'id_code': ['0', '1', '2'], 'diagnosis': [0, 1, 2]}).to_csv(base/'labels.csv', index=False)
            args = batch_validate.build_parser().parse_args([
                '--csv', str(base/'labels.csv'), '--image-dir', str(base),
                '--output', str(base/'output'), '--batch-size', '3',
                '--quality-threshold', '0', '--skip-segmentation', '--no-enhancement'])

            def predict(images, **kwargs):
                if len(images) > 1 or images[0][0, 0, 0] == 11:
                    raise ValueError('deliberate item/batch failure')
                return {'grades': np.array([0]), 'grade_probabilities': np.array([[.8, .05, .05, .05, .05]]),
                        'continuous_scores': np.array([.5]), 'referable_head_probabilities': np.array([.1]),
                        'referable_head_predictions': np.array([False]), 'head_disagreements': np.array([False]),
                        'triage_actions': np.array(['auto_clear'])}

            with patch.object(batch_validate, 'predict_batch_with_details', side_effect=predict), contextlib.redirect_stdout(io.StringIO()):
                first = batch_validate.run_batch(args)
            self.assertEqual(first['counts']['successful'], 2)
            self.assertEqual(first['counts']['errors'], 1)
            args.resume = True
            with patch.object(batch_validate, 'predict_batch_with_details') as mocked, contextlib.redirect_stdout(io.StringIO()):
                second = batch_validate.run_batch(args)
            mocked.assert_not_called()
            self.assertEqual(first['metrics'], second['metrics'])
            cv2.imwrite(str(base/'0.png'), np.full((32, 32, 3), 22, np.uint8))
            with self.assertRaisesRegex(ValueError, 'identity changed'):
                batch_validate.run_batch(args)

    def test_training_resume_matches_uninterrupted_training(self):
        class TinyModel(torch.nn.Module):
            def __init__(self, **kwargs):
                super().__init__()
                self.head = torch.nn.Sequential(torch.nn.Dropout(.3), torch.nn.Linear(3, 6))

            def forward(self, image):
                output = self.head(image.mean(dim=(2, 3)))
                return output[:, :5], output[:, 5]

        with tempfile.TemporaryDirectory() as directory:
            base, rows = Path(directory), []
            for split in ('fit', 'selection', 'calibration'):
                for grade in range(5):
                    name = f'{split}_{grade}'
                    image = np.random.default_rng(len(rows)).integers(10, 255, (32, 32, 3), dtype=np.uint8)
                    cv2.imwrite(str(base/f'{name}.png'), image)
                    rows.append({'image_id': name, 'image_path': f'{name}.png', 'grade': grade,
                                 'group_id': name, 'split': split, 'sha256': sha256_file(base/f'{name}.png')})
            pd.DataFrame(rows).to_csv(base/'manifest.csv', index=False)
            common = ['--manifest', str(base/'manifest.csv'), '--epochs', '2', '--batch-size', '2']
            original_save = train_grading.save_checkpoint

            def stop_after_epoch(path, state):
                original_save(path, state)
                if path.name == 'last.pt' and state['epoch'] == 0:
                    raise InterruptedError('simulated interruption after durable epoch')

            with patch.object(train_grading, 'NetrAIModel', TinyModel), contextlib.redirect_stdout(io.StringIO()):
                train_grading.main(common + ['--output', str(base/'full')])
                with patch.object(train_grading, 'save_checkpoint', side_effect=stop_after_epoch):
                    with self.assertRaises(InterruptedError):
                        train_grading.main(common + ['--output', str(base/'resumed')])
                self.assertTrue((base/'resumed/best.pt').exists())
                train_grading.main(common + ['--output', str(base/'resumed'), '--resume', str(base/'resumed/last.pt')])
            full = torch.load(base/'full/last.pt', weights_only=True)
            resumed = torch.load(base/'resumed/last.pt', weights_only=True)
            self.assertEqual(full['history'], resumed['history'])
            for name, tensor in full['model_state_dict'].items():
                torch.testing.assert_close(tensor, resumed['model_state_dict'][name], rtol=0, atol=0)
