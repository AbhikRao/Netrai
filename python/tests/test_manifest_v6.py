"""Tests for manifest v6 construction (patient/cluster-level development holdouts)."""
import unittest

import numpy as np
import pandas as pd

import build_manifest_v6 as bm


def toy_manifest(seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for source, groups, per_group in (('a', 200, 2), ('b', 150, 1), ('c', 60, 4)):
        for g in range(groups):
            for k in range(per_group):
                rows.append({'image_id': f'{source}_{g}_{k}', 'grade': int(rng.choice(5, p=[.4, .15, .25, .1, .1])),
                             'group_id': f'{source}_g{g}', 'source': source, 'split': 'train'})
    for g in range(40):  # an existing validation source
        rows.append({'image_id': f'v_{g}', 'grade': int(rng.integers(0, 5)), 'group_id': f'v_g{g}', 'source': 'v',
                     'split': 'validation'})
    return pd.DataFrame(rows)


def with_hashes(frame, duplicates=()):
    """Unique PNG hashes, except that each (i, j) pair of row positions shares one."""
    frame = frame.copy()
    frame['png_sha256'] = [f'h{i}' for i in range(len(frame))]
    for i, j in duplicates:
        frame.iloc[j, frame.columns.get_loc('png_sha256')] = frame.iloc[i]['png_sha256']
    return frame


class CarveTests(unittest.TestCase):
    def setUp(self):
        self.base = toy_manifest()
        self.carved = bm.carve_td(self.base, 0.10, seed=1)

    def test_groups_never_cross_splits(self):
        bm.check_manifest(self.carved)

    def test_validation_rows_are_untouched(self):
        pd.testing.assert_series_equal(self.carved.loc[self.base.split == 'validation', 'split'],
                                       self.base.loc[self.base.split == 'validation', 'split'])

    def test_only_training_rows_become_td_and_everything_else_is_unchanged(self):
        changed = self.carved.split != self.base.split
        self.assertTrue((self.base.loc[changed, 'split'] == 'train').all())
        self.assertTrue((self.carved.loc[changed, 'split'] == 'td').all())
        pd.testing.assert_frame_equal(self.carved.drop(columns='split'), self.base.drop(columns='split'))

    def test_td_share_is_about_the_requested_fraction_per_source(self):
        for source in ('a', 'b', 'c'):
            part = self.carved[self.carved.source == source]
            groups = part.group_id.nunique()
            td_groups = part.loc[part.split == 'td', 'group_id'].nunique()
            self.assertAlmostEqual(td_groups / groups, 0.10, delta=0.02)

    def test_deterministic_for_a_seed_and_different_across_seeds(self):
        again = bm.carve_td(self.base, 0.10, seed=1)
        pd.testing.assert_frame_equal(self.carved, again)
        other = bm.carve_td(self.base, 0.10, seed=2)
        self.assertFalse(self.carved.split.equals(other.split))

    def test_a_source_does_not_change_when_another_is_added(self):
        subset = self.base[self.base.source != 'c']
        again = bm.carve_td(subset, 0.10, seed=1)
        a = self.carved[self.carved.source == 'a'].split.to_numpy()
        np.testing.assert_array_equal(a, again[again.source == 'a'].split.to_numpy())

    def test_bad_fractions_rejected(self):
        for fraction in (0.0, 0.5, -0.1):
            with self.assertRaises(ValueError):
                bm.carve_td(self.base, fraction)

    def test_coverage_check_flags_thin_holdouts(self):
        thin = self.carved.copy()
        thin.loc[(thin.split == 'td') & (thin.source == 'a'), 'grade'] = 0  # no referable TD images
        with self.assertRaisesRegex(ValueError, 'too few'):
            bm.check_td_coverage(thin, minimum=5)
        self.assertEqual(set(bm.check_td_coverage(self.carved, minimum=5)), {'a', 'b', 'c'})


class DuplicateHandlingTests(unittest.TestCase):
    def setUp(self):
        base = toy_manifest()
        # row 0 and row 450 (a different source) are identical photos; so are rows 3 and 9 in source a
        self.pairs = [(3, 9), (10, 300)]
        self.base = with_hashes(base, self.pairs)

    def test_clusters_merge_groups_and_identical_photographs(self):
        clustered = bm.add_clusters(self.base)
        for i, j in self.pairs:
            self.assertEqual(clustered.cluster_id.iloc[i], clustered.cluster_id.iloc[j])
        # rows that share a group id share a cluster
        same_group = clustered[clustered.group_id == 'c_g0']
        self.assertEqual(same_group.cluster_id.nunique(), 1)
        self.assertGreater(clustered.cluster_id.nunique(), 50)

    def test_duplicates_never_straddle_td_and_train(self):
        carved = bm.carve_td(bm.add_clusters(self.base), 0.2, seed=3)
        bm.check_manifest(carved)  # also checks cluster_id and png_sha256 across splits
        for i, j in self.pairs:
            self.assertEqual(carved.split.iloc[i], carved.split.iloc[j])

    def test_td_carving_without_clusters_can_split_duplicates(self):
        # documents why clusters are needed: with group_id alone some seed separates a duplicate pair
        split_apart = False
        for seed in range(40):
            carved = bm.carve_td(self.base, 0.2, seed=seed)
            if any(carved.split.iloc[i] != carved.split.iloc[j] for i, j in self.pairs):
                split_apart = True
                break
        self.assertTrue(split_apart)

    def test_conflicting_duplicate_groups_are_excluded(self):
        frame = self.base.copy()
        frame.loc[frame.index[3], 'grade'] = 0
        frame.loc[frame.index[9], 'grade'] = 4
        cleaned, counts = bm.apply_exclusions(frame)
        self.assertEqual(cleaned.split.iloc[3], 'excluded')
        self.assertEqual(cleaned.split.iloc[9], 'excluded')
        self.assertEqual(counts['conflicting_duplicate_rows'], 2)

    def test_training_duplicates_of_validation_images_are_excluded(self):
        validation_position = int(np.flatnonzero(self.base.split == 'validation')[0])
        frame = with_hashes(toy_manifest(), [(validation_position, 5)])
        frame.iloc[5, frame.columns.get_loc('grade')] = frame.iloc[validation_position]['grade']  # same label
        cleaned, counts = bm.apply_exclusions(frame)
        self.assertEqual(cleaned.split.iloc[5], 'excluded')
        self.assertEqual(cleaned.split.iloc[validation_position], 'validation')
        self.assertEqual(counts['train_duplicates_of_validation'], 1)

    def test_check_manifest_detects_a_cross_split_duplicate_hash(self):
        frame = bm.add_clusters(self.base)
        frame = bm.carve_td(frame, 0.2, seed=3)
        i, j = self.pairs[0]
        frame.iloc[j, frame.columns.get_loc('split')] = 'validation' if frame.split.iloc[i] != 'validation' else 'train'
        with self.assertRaisesRegex(ValueError, 'cross'):
            bm.check_manifest(frame)

    def test_excluded_rows_do_not_count_as_crossing(self):
        frame = bm.add_clusters(self.base)
        cleaned, _ = bm.apply_exclusions(frame.assign(grade=lambda d: np.where(d.index == 3, 0, np.where(d.index == 9, 4, d.grade))))
        bm.check_manifest(bm.carve_td(cleaned, 0.2, seed=3))


class CheckManifestTests(unittest.TestCase):
    def test_rejects_duplicate_ids(self):
        frame = toy_manifest()
        frame.loc[1, 'image_id'] = frame.loc[0, 'image_id']
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            bm.check_manifest(frame)

    def test_rejects_groups_crossing_splits(self):
        frame = toy_manifest()
        frame.loc[0, 'split'] = 'validation'  # group a_g0 has two images
        with self.assertRaisesRegex(ValueError, 'cross'):
            bm.check_manifest(frame)

    def test_rejects_unknown_split_and_bad_grade_and_missing_columns(self):
        frame = toy_manifest()
        with self.assertRaisesRegex(ValueError, 'Unknown splits'):
            bm.check_manifest(frame.assign(split='dev'))
        with self.assertRaisesRegex(ValueError, 'Grades'):
            bm.check_manifest(frame.assign(grade=5))
        with self.assertRaisesRegex(ValueError, 'columns'):
            bm.check_manifest(frame.drop(columns='group_id'))

    def test_rejects_a_group_spanning_sources(self):
        frame = toy_manifest()
        frame.loc[frame.group_id == 'a_g0', 'source'] = ['a', 'b']
        with self.assertRaisesRegex(ValueError, 'several sources'):
            bm.check_manifest(frame)


if __name__ == '__main__':
    unittest.main()
