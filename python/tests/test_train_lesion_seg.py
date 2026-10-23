"""Tests for the lesion trainer: tiled inference, positive weights and a tiny end-to-end CPU run."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import lesion_data as ld
import train_lesion_seg as tl
from models.lesion_unet import LesionNet
from tests.test_lesion_data import DataTestCase


class PredictTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.model = LesionNet('unet', 'efficientnet_b0', pretrained=False).eval()
        rng = np.random.default_rng(0)
        self.image = (rng.random((100, 150, 3)) * 255).astype(np.uint8)

    def test_whole_image_window_equals_a_direct_forward_pass(self):
        probs = tl.predict_full(self.model, self.image, torch.device('cpu'), tile=256, overlap=32, enhance='none')
        self.assertEqual(probs.shape, (100, 150, 4))
        padded, _, _ = tl.pad_to_32(self.image)
        with torch.no_grad():
            direct = torch.sigmoid(self.model(ld.to_tensor(padded)[None]))[0, :, :100, :150].permute(1, 2, 0).numpy()
        np.testing.assert_allclose(probs, direct, atol=1e-5)

    def test_tiled_inference_covers_every_pixel_with_valid_probabilities(self):
        probs = tl.predict_full(self.model, self.image, torch.device('cpu'), tile=64, overlap=32, enhance='ben')
        self.assertEqual(probs.shape, (100, 150, 4))
        self.assertTrue(np.isfinite(probs).all())
        self.assertTrue(((probs >= 0) & (probs <= 1)).all())
        self.assertGreater(float(probs.min()), 0.0)  # every pixel was written (sigmoid output never exactly 0)

    def test_padding_makes_dimensions_divisible_by_32(self):
        padded, ph, pw = tl.pad_to_32(self.image)
        self.assertEqual((padded.shape[0] % 32, padded.shape[1] % 32), (0, 0))
        self.assertEqual((ph, pw), (28, 10))


class WeightTests(unittest.TestCase):
    def frame(self):
        rows = []
        for i in range(4):
            row = {'height': 100, 'width': 100}
            for lesion in ld.LESIONS:
                row[f'annotated_{lesion}'] = True
                row[f'pixels_{lesion}'] = {'EX': 100, 'HE': 400, 'MA': 4, 'SE': 0}[lesion]
            rows.append(row)
        rows[0]['annotated_HE'] = False
        return pd.DataFrame(rows)

    def test_rarer_lesions_get_larger_weights_and_absent_lesions_get_one(self):
        w = tl.positive_weights(self.frame(), 'auto')
        values = dict(zip(ld.LESIONS, w.tolist()))
        self.assertGreater(values['MA'], values['EX'])
        self.assertGreater(values['EX'], values['HE'])
        self.assertEqual(values['SE'], 1.0)
        self.assertTrue(all(1.0 <= v <= 100.0 for v in values.values()))
        self.assertEqual(tl.positive_weights(self.frame(), 'none').tolist(), [1.0] * 4)

    def test_selection_value_ignores_lesions_without_positives(self):
        summary = {'EX': {'aupr': 0.5}, 'HE': {'aupr': float('nan')}, 'MA': {'aupr': 0.1}, 'SE': {'aupr': 0.3}}
        self.assertAlmostEqual(tl.selection_value(summary), 0.3)
        self.assertTrue(np.isnan(tl.selection_value({'EX': {'aupr': float('nan')}})))


class EndToEndTests(DataTestCase):
    def build_export(self):
        clusters = [(str(r.image_path), i) for i, r in enumerate(ld.ddr_records('train') + ld.ddr_records('valid') + ld.idrid_train_records())]
        csv = self.base / 'clusters.csv'
        pd.DataFrame(clusters, columns=['image', 'cluster']).to_csv(csv, index=False)
        out = self.base / 'export'
        ld.export_dataset(out, long_side=256, valid_fraction=0.3, seed=2, clusters_csv=csv, workers=1)
        return out

    def args(self, dataset, output, *extra):
        return tl.parse_args(['--dataset', str(dataset), '--output', str(output), '--encoder', 'efficientnet_b0', '--no-pretrained',
                              '--tile', '64', '--eval-tile', '128', '--overlap', '32', '--epochs', '1', '--samples-per-epoch', '8',
                              '--batch-size', '2', '--workers', '0', '--seed', '1', *extra])

    def test_tiny_run_writes_outputs_and_never_reads_test_labels(self):
        dataset = self.build_export()
        output = self.base / 'run'
        result = tl.run(self.args(dataset, output))
        for name in ('best.pt', 'last.pt', 'history.json', 'valid_histograms_best.npz', 'valid_summary_best.json',
                     'run_info.json', 'compute.json'):
            self.assertTrue((output / name).is_file(), name)
        info = json.loads((output / 'run_info.json').read_text())
        self.assertIs(info['test_labels_read'], False)
        self.assertEqual(info['evidence_tiers']['valid'], 'TD (cluster-split development data)')
        record = result['history'][0]
        self.assertEqual(set(record['valid']), set(ld.LESIONS))
        histograms = np.load(output / 'valid_histograms_best.npz')['histograms']
        self.assertEqual(histograms.shape[1:], (4, 2, 1000))
        summary = json.loads((output / 'valid_summary_best.json').read_text())
        self.assertEqual(set(summary['thresholds_chosen_on_valid']), set(ld.LESIONS))

    def test_resume_and_wall_clock_guard(self):
        dataset = self.build_export()
        output = self.base / 'run2'
        tl.run(self.args(dataset, output))
        resumed = tl.run(self.args(dataset, output, '--epochs', '2'))
        self.assertEqual([r['epoch'] for r in resumed['history']], [0, 1])
        guarded = tl.run(self.args(dataset, self.base / 'run3', '--max-hours', '0', '--epochs', '3'))
        self.assertTrue(guarded['truncated'])
        self.assertEqual(len(guarded['history']), 1)
        self.assertTrue(json.loads((self.base / 'run3' / 'compute.json').read_text())['truncated_by_wall_clock'])

    def test_a_cluster_spanning_train_and_valid_is_refused(self):
        dataset = self.build_export()
        manifest = pd.read_csv(dataset / 'manifest.csv')
        manifest.loc[manifest.split == 'valid', 'cluster'] = manifest.loc[manifest.split == 'train', 'cluster'].iloc[0]
        manifest.to_csv(dataset / 'manifest.csv', index=False)
        with self.assertRaises(AssertionError):
            tl.run(self.args(dataset, self.base / 'run4'))


if __name__ == '__main__':
    unittest.main()
