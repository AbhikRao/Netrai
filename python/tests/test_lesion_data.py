"""Tests for lesion_data on a synthetic DDR/IDRiD-like tree (no real labels are read)."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np
import pandas as pd

import exposure
import lesion_data as ld


def fundus(h, w, seed):
    rng = np.random.default_rng(seed)
    image = np.zeros((h, w, 3), np.uint8)
    yy, xx = np.mgrid[:h, :w]
    inside = ((yy - h / 2) / (h * 0.45)) ** 2 + ((xx - w / 2) / (w * 0.45)) ** 2 < 1
    image[inside] = np.clip(rng.normal(120, 10, (inside.sum(), 3)), 0, 255).astype(np.uint8)
    return image


def blob_mask(h, w, spots, radius=3):
    mask = np.zeros((h, w), np.uint8)
    for y, x in spots:
        cv2.circle(mask, (x, y), radius, 255, -1)
    return mask


class Tree:
    """DDR train/valid/test + IDRiD tree under a temporary root."""

    def __init__(self, base: Path):
        self.base = base
        self.ddr, self.idrid = base / 'ddr', base / 'idrid'
        h, w = 300, 400
        for split, count in (('train', 6), ('valid', 3), ('test', 2)):
            label = 'segmentation label' if split == 'valid' else 'label'
            (self.ddr / split / 'image').mkdir(parents=True)
            for lesion in ld.LESIONS:
                (self.ddr / split / label / lesion).mkdir(parents=True)
            for k in range(count):
                stem = f'007-{split}{k}-100'
                cv2.imwrite(str(self.ddr / split / 'image' / f'{stem}.jpg'), fundus(h, w, hash(stem) % 1000))
                for index, lesion in enumerate(ld.LESIONS):
                    spots = [(120 + 10 * index, 150 + 15 * k)] if lesion != 'SE' or k % 2 == 0 else []
                    cv2.imwrite(str(self.ddr / split / label / lesion / f'{stem}.tif'), blob_mask(h, w, spots))
        images = self.idrid / '1. Original Images' / 'a. Training Set'
        truth = self.idrid / '2. All Segmentation Groundtruths' / 'a. Training Set'
        images.mkdir(parents=True)
        for lesion, name in ld.IDRID_MASK_DIR.items():
            (truth / name).mkdir(parents=True)
        for k in range(2):
            stem = f'IDRiD_0{k + 1}'
            cv2.imwrite(str(images / f'{stem}.jpg'), fundus(h, w, 50 + k))
            for lesion in ('MA', 'EX'):
                cv2.imwrite(str(truth / ld.IDRID_MASK_DIR[lesion] / f'{stem}_{lesion}.tif'), blob_mask(h, w, [(100, 120)]))
            if k == 0:  # image 2 has no HE/SE masks: not annotated
                for lesion in ('HE', 'SE'):
                    cv2.imwrite(str(truth / ld.IDRID_MASK_DIR[lesion] / f'{stem}_{lesion}.tif'), blob_mask(h, w, [(150, 200)]))

    def patches(self, registry: Path):
        return [mock.patch.object(ld, 'DDR_ROOT', self.ddr), mock.patch.object(ld, 'IDRID_ROOT', self.idrid),
                mock.patch.object(exposure, 'REGISTRY_PATH', registry)]


class DataTestCase(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.tree = Tree(self.base)
        self.registry = self.base / 'registry.json'
        self.registry.write_text(json.dumps({'schema_version': 1, 'updated': '2026-01-01', 'cohorts': {
            'ddr_lesion_seg_test': {'kind': 'segmentation', 'status': 'reserved',
                                    'label_paths': [str(self.tree.ddr / 'test' / 'label')]}}}))
        for patch in self.tree.patches(self.registry):
            patch.start()
            self.addCleanup(patch.stop)


class RecordTests(DataTestCase):
    def test_ddr_train_and_valid_records_find_all_masks(self):
        train, valid = ld.ddr_records('train'), ld.ddr_records('valid')
        self.assertEqual((len(train), len(valid)), (6, 3))
        self.assertTrue(all(all(p is not None for p in r.mask_paths.values()) for r in train + valid))
        self.assertIn('segmentation label', str(valid[0].mask_paths['MA']))
        self.assertTrue(train[0].image_id.startswith('ddr_007-'))

    def test_test_split_needs_a_valid_lock(self):
        with self.assertRaises(exposure.ExposureError):
            ld.ddr_records('test')
        with self.assertRaises(exposure.ExposureError):
            ld.ddr_records('test', self.base / 'no_lock')
        lock = self.base / 'lock'
        lock.mkdir()
        (lock / 'lock.json').write_text(json.dumps({'tests': {'ddr_lesion_seg_test': 'x'}, 'labels_read': False}))
        records = ld.ddr_records('test', lock)
        self.assertEqual(len(records), 2)

    def test_reserved_masks_cannot_be_read_without_explicit_permission(self):
        lock = self.base / 'lock'
        lock.mkdir()
        (lock / 'lock.json').write_text(json.dumps({'tests': {'ddr_lesion_seg_test': 'x'}, 'labels_read': False}))
        record = ld.ddr_records('test', lock)[0]
        with self.assertRaises(exposure.ExposureError):
            ld.read_masks(record, (300, 400))
        masks, annotated = ld.read_masks(record, (300, 400), allow_reserved=True)
        self.assertTrue(annotated.all())
        self.assertGreater(int((masks > 0).sum()), 0)

    def test_unknown_split_rejected(self):
        with self.assertRaises(ValueError):
            ld.ddr_records('dev')

    def test_idrid_missing_masks_are_unannotated(self):
        records = ld.idrid_train_records()
        self.assertEqual(len(records), 2)
        first, second = records
        self.assertTrue(all(first.mask_paths[l] is not None for l in ld.LESIONS))
        self.assertIsNone(second.mask_paths['HE'])
        self.assertIsNone(second.mask_paths['SE'])
        _, annotated = ld.read_masks(second, (300, 400))
        self.assertEqual(annotated.tolist(), [True, False, True, False])  # EX, HE, MA, SE


class MaskAndPrepareTests(DataTestCase):
    def test_masks_are_binary_and_size_checked(self):
        record = ld.ddr_records('train')[0]
        masks, annotated = ld.read_masks(record, (300, 400))
        self.assertEqual(set(np.unique(masks)), {0, 255})
        self.assertTrue(annotated.all())
        with self.assertRaisesRegex(ValueError, 'size'):
            ld.read_masks(record, (100, 100))

    def test_prepare_crops_the_field_and_downsizes_without_upsizing(self):
        image = fundus(300, 400, 1)
        masks = np.zeros((300, 400, 4), np.uint8)
        masks[150, 200, 2] = 255
        masks[140:150, 190:200, 0] = 255
        small, m_small, scale = ld.prepare(image, masks, long_side=100)
        self.assertLess(max(small.shape[:2]), 101)
        self.assertLess(scale, 1.0)
        self.assertEqual(small.shape[:2], m_small.shape[:2])
        self.assertGreater(int((m_small[:, :, 0] > 0).sum()), 0)  # the larger lesion survives
        self.assertEqual(set(np.unique(m_small)), {0, 255})
        same, _, unit = ld.prepare(image, masks, long_side=4096)
        self.assertEqual(unit, 1.0)
        self.assertLessEqual(max(same.shape[:2]), 400)  # cropped, never upsized

    def test_foreground_box_ignores_black_margins(self):
        image = np.zeros((100, 100, 3), np.uint8)
        image[20:60, 30:80] = 100
        self.assertEqual(ld.foreground_box(image), (20, 60, 30, 80))


class ClusterSplitTests(unittest.TestCase):
    def frame(self):
        rows = [{'image_id': f'ddr_s{i}', 'dataset': 'ddr'} for i in range(40)]
        rows += [{'image_id': f'idrid_i{i}', 'dataset': 'idrid'} for i in range(4)]
        return pd.DataFrame(rows)

    def clusters(self):
        paths, labels = [], []
        for i in range(40):
            paths.append(f'/x/s{i}.jpg')
            labels.append(0 if i < 12 else i)  # one big cluster of 12, singletons otherwise
        for i in range(4):
            paths.append(f'/y/i{i}.jpg')
            labels.append(100 + i)
        return pd.DataFrame({'image': paths, 'cluster': labels})

    def test_clusters_stay_together_the_largest_goes_to_train_and_idrid_is_train_only(self):
        result = ld.cluster_split(self.frame(), self.clusters(), 0.25, seed=1)
        self.assertTrue((result.groupby('cluster').split.nunique() == 1).all())
        self.assertEqual(set(result[result.cluster == 0].split), {'train'})
        self.assertEqual(set(result[result.dataset == 'idrid'].split), {'train'})
        ddr_valid = (result[(result.dataset == 'ddr')].split == 'valid').sum()
        self.assertGreaterEqual(ddr_valid, 10)
        self.assertLessEqual(ddr_valid, 11)

    def test_missing_cluster_is_an_error_and_split_is_deterministic(self):
        clusters = self.clusters().iloc[:-2]
        with self.assertRaisesRegex(ValueError, 'no cluster'):
            ld.cluster_split(self.frame(), clusters)
        a = ld.cluster_split(self.frame(), self.clusters(), 0.25, 3)
        b = ld.cluster_split(self.frame(), self.clusters(), 0.25, 3)
        pd.testing.assert_frame_equal(a, b)


class ExportTests(DataTestCase):
    def test_export_writes_train_and_valid_images_and_never_touches_the_test_labels(self):
        clusters = []
        for r in ld.ddr_records('train') + ld.ddr_records('valid') + ld.idrid_train_records():
            clusters.append((str(r.image_path), len(clusters)))
        csv = self.base / 'clusters.csv'
        pd.DataFrame(clusters, columns=['image', 'cluster']).to_csv(csv, index=False)
        # make the reserved test labels unreadable garbage: any read attempt would raise or fail
        for path in (self.tree.ddr / 'test' / 'label').rglob('*.tif'):
            path.write_bytes(b'garbage')
        out = self.base / 'export'
        frame = ld.export_dataset(out, long_side=256, valid_fraction=0.3, seed=2, clusters_csv=csv, workers=1)
        self.assertEqual(len(frame), 6 + 3 + 2)
        self.assertEqual(set(frame.split), {'train', 'valid'})
        self.assertTrue(frame.image_id.str.startswith(('ddr_007-train', 'ddr_007-valid', 'idrid_')).all())
        self.assertFalse(frame.image_id.str.contains('007-test').any())  # no reserved test image was exported
        for image_id in frame.image_id:
            self.assertTrue((out / 'images' / f'{image_id}.png').is_file())
            masks = cv2.imread(str(out / 'masks' / f'{image_id}.png'), cv2.IMREAD_UNCHANGED)
            self.assertEqual(masks.shape[2], 4)
        self.assertFalse(any(Path(p).name.startswith('ddr_007-test') for p in (out / 'images').iterdir()))
        summary = json.loads((out / 'export_summary.json').read_text())
        self.assertIn('test masks were not read', summary['reserved'])
        # DDR: every lesion type has a (possibly empty) mask file; IDRiD image 2 lacks HE and SE masks
        self.assertEqual(summary['annotated_images'], {'EX': 11, 'HE': 10, 'MA': 11, 'SE': 10})
        with self.assertRaises(FileExistsError):
            ld.export_dataset(out, 256, 0.3, 2, csv, 1)


class TileTests(DataTestCase):
    def build_export(self):
        clusters = [(str(r.image_path), i) for i, r in enumerate(ld.ddr_records('train') + ld.ddr_records('valid') + ld.idrid_train_records())]
        csv = self.base / 'clusters.csv'
        pd.DataFrame(clusters, columns=['image', 'cluster']).to_csv(csv, index=False)
        out = self.base / 'export'
        frame = ld.export_dataset(out, long_side=1024, valid_fraction=0.3, seed=2, clusters_csv=csv, workers=1)
        return out, frame[frame.split == 'train']

    def test_tiles_have_the_right_shapes_and_are_deterministic(self):
        out, train = self.build_export()
        dataset = ld.LesionTiles(train, out, tile=64, seed=3, length=8)
        image, mask, annotated = dataset[0]
        self.assertEqual(tuple(image.shape), (3, 64, 64))
        self.assertEqual(tuple(mask.shape), (4, 64, 64))
        self.assertEqual(annotated.dtype.is_floating_point, False)
        again = ld.LesionTiles(train, out, tile=64, seed=3, length=8)[0]
        self.assertTrue(np.array_equal(image.numpy(), again[0].numpy()))
        self.assertFalse(np.array_equal(image.numpy(), dataset[1][0].numpy()))
        dataset.epoch = 1
        self.assertFalse(np.array_equal(image.numpy(), dataset[0][0].numpy()))

    def test_lesion_centred_sampling_finds_lesions_far_more_often_than_random(self):
        out, train = self.build_export()
        centred = ld.LesionTiles(train, out, tile=64, seed=1, lesion_prob=1.0, length=60)
        uniform = ld.LesionTiles(train, out, tile=64, seed=1, lesion_prob=0.0, length=60)
        hit = lambda ds: np.mean([float(ds[i][1].sum() > 0) for i in range(len(ds))])
        self.assertGreater(hit(centred), 0.9)
        self.assertGreater(hit(centred), hit(uniform) + 0.2)

    def test_unannotated_channels_are_flagged_per_image(self):
        out, train = self.build_export()
        idrid = train[train.image_id == 'idrid_IDRiD_02'].reset_index(drop=True)
        _, _, annotated = ld.LesionTiles(idrid, out, tile=64, seed=1, length=2)[0]
        self.assertEqual(annotated.tolist(), [True, False, True, False])

    def test_tile_must_be_a_multiple_of_32(self):
        with self.assertRaises(ValueError):
            ld.LesionTiles(pd.DataFrame(), Path('.'), tile=50)


if __name__ == '__main__':
    unittest.main()


class MultiChannelMaskTests(unittest.TestCase):
    def test_multichannel_masks_use_colour_channels_and_ignore_alpha(self):
        import tempfile
        import cv2
        import numpy as np
        with tempfile.TemporaryDirectory() as d:
            raw = np.zeros((20, 30, 4), np.uint8)
            raw[:, :, 3] = 255                     # alpha set everywhere must not mark lesions
            raw[5:8, 10:14, :3] = 255
            path = Path(d) / 'm.tif'
            cv2.imwrite(str(path), raw)
            record = ld.Record('x', 'idrid', Path(d) / 'img.jpg', {'EX': path, 'HE': None, 'MA': None, 'SE': None})
            masks, annotated = ld.read_masks(record, (20, 30))
        self.assertTrue(annotated[0])
        self.assertEqual(int((masks[..., 0] > 0).sum()), 12)
