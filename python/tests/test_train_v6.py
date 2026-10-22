"""Tests for the manifest-v6 trainer: data splitting, EMA, deterministic augmentation, and a tiny CPU run."""
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

import train_v6 as t6


def fundus(seed, size=96):
    rng = np.random.default_rng(seed)
    image = np.zeros((size, size, 3), np.uint8)
    yy, xx = np.mgrid[:size, :size]
    inside = (yy - size / 2) ** 2 + (xx - size / 2) ** 2 < (size * 0.46) ** 2
    base = rng.integers(80, 160, 3)
    for channel in range(3):
        image[:, :, channel] = np.clip(base[channel] + rng.normal(0, 12, (size, size)), 0, 255).astype(np.uint8)
    image[~inside] = 0
    return image


def toy_manifest(directory: Path):
    rows = []
    layout = [('a', 'train', 12), ('b', 'train', 12), ('a', 'td', 6), ('b', 'validation', 6), ('c', 'train', 6),
              ('c', 'validation', 4), ('c', 'td', 4)]
    counter = 0
    for source, split, count in layout:
        for k in range(count):
            image_id = f'{source}_{split}_{k}'
            cv2.imwrite(str(directory / f'{image_id}.png'), fundus(counter))
            rows.append({'image_id': image_id, 'grade': (counter + k) % 5, 'group_id': f'g_{image_id}',
                         'source': source, 'split': split})
            counter += 1
    return pd.DataFrame(rows)


class SplitTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.manifest = toy_manifest(Path(self.temp.name))

    def test_default_split_uses_train_for_fitting_and_td_plus_validation_for_selection(self):
        train, selection, heldout = t6.split_frames(self.manifest)
        self.assertEqual(set(train.split), {'train'})
        self.assertEqual(set(selection.split), {'td', 'validation'})
        self.assertEqual(len(heldout), 0)

    def test_select_on_choices(self):
        _, td, _ = t6.split_frames(self.manifest, select_on='td')
        self.assertEqual(set(td.split), {'td'})
        _, validation, _ = t6.split_frames(self.manifest, select_on='validation')
        self.assertEqual(set(validation.split), {'validation'})
        with self.assertRaises(ValueError):
            t6.split_frames(self.manifest, select_on='all')

    def test_excluded_source_is_in_neither_training_nor_selection(self):
        train, selection, heldout = t6.split_frames(self.manifest, ['c'])
        self.assertEqual(set(heldout.source), {'c'})
        self.assertEqual(len(heldout), 14)  # every split of the held-out source
        self.assertNotIn('c', set(train.source))
        self.assertNotIn('c', set(selection.source))

    def test_selection_exclusion_keeps_source_in_training_only(self):
        train, selection, _ = t6.split_frames(self.manifest, selection_exclude=['b'])
        self.assertIn('b', set(train.source))
        self.assertNotIn('b', set(selection.source))
        self.assertIn('a', set(selection.source))

    def test_checkpoint_init_excludes_aptos_from_selection_by_default(self):
        self.assertEqual(t6.resolve_selection_exclusions('legacy.pt', ['messidor']), ['aptos'])
        self.assertEqual(t6.resolve_selection_exclusions('imagenet', ['messidor']), [])
        self.assertEqual(t6.resolve_selection_exclusions('legacy.pt', ['messidor'], ['aptos', 'idrid']), ['aptos', 'idrid'])

    def test_aptos_fold_requires_imagenet_init(self):
        with self.assertRaises(ValueError):
            t6.resolve_selection_exclusions('legacy.pt', ['aptos'])
        self.assertEqual(t6.resolve_selection_exclusions('imagenet', ['aptos']), [])

    def test_unknown_excluded_source_is_rejected(self):
        with self.assertRaises(ValueError):
            t6.split_frames(self.manifest, ['zzz'])

    def test_manifest_excluded_rows_are_dropped(self):
        manifest = self.manifest.copy()
        manifest.loc[0, 'split'] = 'excluded'
        train, selection, _ = t6.split_frames(manifest)
        self.assertNotIn(manifest.image_id[0], set(train.image_id) | set(selection.image_id))

    def test_overlap_between_train_and_selection_is_detected(self):
        manifest = self.manifest.copy()
        manifest.loc[manifest.split == 'td', 'group_id'] = manifest.loc[manifest.split == 'train'].group_id.iloc[0]
        with self.assertRaises(AssertionError):
            t6.split_frames(manifest)

    def test_source_multipliers(self):
        train, _, _ = t6.split_frames(self.manifest)
        base = np.full(len(train), 1.0 / len(train))
        scaled = t6.apply_source_multipliers(base, train, {'a': 0.0})
        self.assertEqual(float(scaled[(train.source == 'a').to_numpy()].sum()), 0.0)
        self.assertAlmostEqual(float(scaled.sum()), 1.0)
        with self.assertRaises(ValueError):
            t6.apply_source_multipliers(base, train, {'nope': 2.0})


class PrefetchTests(unittest.TestCase):
    def test_prefetch_reads_every_unique_file_once(self):
        with tempfile.TemporaryDirectory() as d:
            paths = []
            for i in range(10):
                path = Path(d) / f'f{i}.bin'
                path.write_bytes(b'x' * (i + 1))
                paths.append(path)
            result = t6.prefetch_files(paths + paths[:3], threads=4)
            self.assertEqual(result['files'], 10)
            self.assertEqual(result['files'], 10)
            self.assertGreaterEqual(result['files_per_second'], 0)

    def test_prefetch_reports_missing_files(self):
        with self.assertRaises(FileNotFoundError):
            t6.prefetch_files(['/nonexistent/file.png'], threads=2)


class ComponentTests(unittest.TestCase):
    def test_ema_follows_the_decay_formula_and_copies_integer_buffers(self):
        model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.BatchNorm1d(2))
        ema = t6.EMA(model, 0.9)
        before = ema.shadow[0].weight.clone()
        with torch.no_grad():
            model[0].weight.add_(1.0)
        ema.update(model)
        decay = min(0.9, (1 + 1) / (10 + 1))  # warm-up: the first update trusts the model strongly
        expected = before * decay + model[0].weight.detach() * (1 - decay)
        torch.testing.assert_close(ema.shadow[0].weight, expected)
        model[1].num_batches_tracked.fill_(7)
        ema.update(model)
        self.assertEqual(int(ema.shadow[1].num_batches_tracked), 7)
        for _ in range(500):
            ema.update(model)
        torch.testing.assert_close(ema.shadow[0].weight, model[0].weight.detach(), atol=1e-4, rtol=1e-4)

    def test_ema_rejects_bad_decay(self):
        for decay in (0.0, 1.0, -0.5):
            with self.assertRaises(ValueError):
                t6.EMA(torch.nn.Linear(1, 1), decay)

    def test_wall_clock_guard(self):
        self.assertFalse(t6.past_deadline(1000.0, 1.0, now=1000.0 + 3599))
        self.assertTrue(t6.past_deadline(1000.0, 1.0, now=1000.0 + 3601))

    def test_freezing_blocks_stops_gradients_on_the_early_backbone(self):
        model, frozen = t6.build_model('random', freeze_blocks=4)
        self.assertEqual(len(frozen), 6)
        self.assertFalse(any(p.requires_grad for p in model.backbone.conv_stem.parameters()))
        self.assertFalse(any(p.requires_grad for p in model.backbone.blocks[3].parameters()))
        self.assertTrue(all(p.requires_grad for p in model.backbone.blocks[5].parameters()))
        self.assertTrue(all(p.requires_grad for p in model.grade_head.parameters()))


class RowsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.manifest = toy_manifest(self.directory)
        self.frame = self.manifest[self.manifest.split == 'train'].reset_index(drop=True)

    def make(self, train, size=None, glare=0.0):
        return t6.RowsV6(self.frame, [self.directory], train, seed=5, glare_prob=glare, input_size=size)

    def test_input_size_downsizes_and_keeps_the_deployed_tensor_layout(self):
        x, y = self.make(False, size=64)[0]
        self.assertEqual(tuple(x.shape), (3, 64, 64))
        self.assertEqual(x.dtype, torch.float32)
        self.assertEqual(y, int(self.frame.grade[0]))
        x_native, _ = self.make(False)[0]
        self.assertEqual(tuple(x_native.shape), (3, 96, 96))

    def test_evaluation_inputs_are_deterministic_and_unaugmented(self):
        a, _ = self.make(False, 64)[3]
        b, _ = self.make(False, 64)[3]
        torch.testing.assert_close(a, b)

    def test_training_augmentation_is_seeded_by_position_not_worker(self):
        first, second = self.make(True, 64), self.make(True, 64)
        for dataset in (first, second):
            dataset.set_order([2, 2, 0, 5], epoch=1)
        torch.testing.assert_close(first[0][0], second[0][0])  # same position, same augmentation
        self.assertFalse(torch.equal(first[0][0], first[1][0]))  # same row drawn twice is augmented differently
        self.assertEqual(len(first), 4)
        self.assertEqual(first[0][1], first[1][1])  # ... but keeps its label
        second.set_order([2, 2, 0, 5], epoch=2)
        self.assertFalse(torch.equal(first[0][0], second[0][0]))  # a new epoch re-augments

    def test_missing_image_raises(self):
        frame = self.frame.copy()
        frame.loc[0, 'image_id'] = 'does_not_exist'
        with self.assertRaises(FileNotFoundError):
            t6.RowsV6(frame, [self.directory], False, 1)[0]


class EndToEndTests(unittest.TestCase):
    """A few real optimisation steps on tiny synthetic images (CPU)."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.images = self.base / 'images'
        self.images.mkdir()
        toy_manifest(self.images).to_csv(self.base / 'manifest.csv', index=False)

    def args(self, output, *extra):
        return t6.parse_args(['--manifest', str(self.base / 'manifest.csv'), '--roots', str(self.images),
                              '--output', str(output), '--init', 'random', '--freeze-blocks', '4', '--input-size', '64',
                              '--batch-size', '4', '--workers', '0', '--epochs', '1', '--seed', '3', *extra])

    def test_run_writes_all_outputs_and_scores_the_held_out_source(self):
        output = self.base / 'run'
        result = t6.run(self.args(output, '--exclude-source', 'c', '--prefetch-threads', '4'))
        for name in ('history.json', 'best.pt', 'last.pt', 'validation_rows.csv', 'best_validation_outputs.npy',
                     'heldout_rows.csv', 'best_heldout_outputs.npy', 'selection_outputs_epoch0.npy',
                     'heldout_outputs_epoch0.npy', 'run_info.json', 'compute.json'):
            self.assertTrue((output / name).is_file(), name)
        info = json.loads((output / 'run_info.json').read_text())
        self.assertEqual(info['counts'], {'train': 24, 'selection': 12, 'heldout': 14})
        self.assertEqual(info['evidence_tiers'], {'selection': 'TD', 'heldout': 'TL'})
        self.assertEqual(set(pd.read_csv(output / 'heldout_rows.csv').source), {'c'})
        self.assertNotIn('c', set(pd.read_csv(output / 'validation_rows.csv').source))
        outputs = np.load(output / 'best_validation_outputs.npy')
        self.assertEqual(outputs.shape, (12, 6))
        self.assertTrue(np.isfinite(outputs).all())
        self.assertEqual(np.load(output / 'best_heldout_outputs.npy').shape, (14, 6))
        record = result['history'][0]
        self.assertIn('heldout_per_source', record)
        self.assertFalse(record['truncated'])
        self.assertTrue(np.isfinite(record['loss']))

    def test_outputs_feed_the_policy_fitter_format(self):
        output = self.base / 'run2'
        t6.run(self.args(output))
        rows = pd.read_csv(output / 'validation_rows.csv')
        self.assertEqual(list(rows.columns), ['image_id', 'grade', 'source', 'group_id'])
        self.assertEqual(len(rows), len(np.load(output / 'best_validation_outputs.npy')))

    def test_resume_continues_after_the_last_completed_epoch(self):
        output = self.base / 'run3'
        t6.run(self.args(output))
        resumed = t6.run(self.args(output, '--epochs', '2'))
        self.assertEqual([r['epoch'] for r in resumed['history']], [0, 1])
        self.assertEqual(len(json.loads((output / 'history.json').read_text())), 2)

    def test_ema_run_evaluates_the_averaged_weights(self):
        output = self.base / 'run4'
        result = t6.run(self.args(output, '--ema', '0.9', '--loss', 'emd'))
        self.assertTrue(result['history'][0]['ema'])
        state = torch.load(output / 'last.pt', map_location='cpu', weights_only=False)
        self.assertIsNotNone(state['ema'])
        self.assertGreater(state['ema_updates'], 0)

    def test_wall_clock_guard_stops_cleanly_and_is_recorded(self):
        output = self.base / 'run5'
        result = t6.run(self.args(output, '--max-hours', '0', '--epochs', '3'))
        self.assertTrue(result['truncated'])
        self.assertEqual(len(result['history']), 1)  # stopped after the first epoch's early break
        compute = json.loads((output / 'compute.json').read_text())
        self.assertTrue(compute['truncated_by_wall_clock'])
        self.assertTrue((output / 'last.pt').is_file())

    def test_heldout_images_are_never_trained_on(self):
        output = self.base / 'run6'
        t6.run(self.args(output, '--exclude-source', 'a', '--source-weight', 'b=2.0'))
        info = json.loads((output / 'run_info.json').read_text())
        self.assertNotIn('a', info['train_by_source'])
        self.assertNotIn('a', info['selection_by_source'])
        self.assertEqual(info['source_multipliers'], {'b': 2.0})


if __name__ == '__main__':
    unittest.main()
