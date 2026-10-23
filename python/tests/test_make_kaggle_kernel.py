"""Tests for the Kaggle kernel stager (local files only)."""
import ast
import json
import tempfile
import unittest
from pathlib import Path

import make_kaggle_kernel as mk


class KernelTests(unittest.TestCase):
    def build(self, directory, **overrides):
        args = dict(out=Path(directory) / 'k', kernel_id='netrai-x2-messidor-s1', title='netrai-x2-messidor-s1',
                    dataset_sources=['abhikrao/netrai-train-768', 'abhikrao/netrai-code-v6'],
                    train_args=['--init', 'LEGACY', '--exclude-source', 'messidor', '--epochs', '1'])
        args.update(overrides)
        return mk.build_kernel(**args)

    def test_metadata_is_private_gpu_internet_and_names_the_datasets(self):
        with tempfile.TemporaryDirectory() as tmp:
            metadata = self.build(tmp)
            on_disk = json.loads((Path(tmp) / 'k' / 'kernel-metadata.json').read_text())
            self.assertEqual(metadata, on_disk)
            self.assertEqual(on_disk['id'], 'abhikrao/netrai-x2-messidor-s1')
            self.assertTrue(on_disk['is_private'] and on_disk['enable_gpu'] and on_disk['enable_internet'])
            self.assertEqual(on_disk['dataset_sources'], ['abhikrao/netrai-train-768', 'abhikrao/netrai-code-v6'])
            self.assertEqual(on_disk['code_file'], 'run_training.py')

    def test_generated_script_is_valid_python_and_carries_the_arguments(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.build(tmp)
            text = (Path(tmp) / 'k' / 'run_training.py').read_text()
            ast.parse(text)
            self.assertIn("'--exclude-source', 'messidor'", text)
            self.assertIn("'LEGACY'", text)
            self.assertIn('train_v6.py', text)
            self.assertIn('timm==1.0.29', text)
            self.assertNotIn('KAGGLE_KEY', text)

    def test_template_owned_arguments_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            for forbidden in ('--manifest', '--roots', '--output'):
                with self.assertRaises(ValueError):
                    self.build(tmp, train_args=[forbidden, 'x'], out=Path(tmp) / forbidden.strip('-'))

    def test_staging_is_immutable_and_slugs_are_validated(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.build(tmp)
            with self.assertRaises(FileExistsError):
                self.build(tmp)
            with self.assertRaises(ValueError):
                self.build(tmp, out=Path(tmp) / 'other', kernel_id='bad slug')
            with self.assertRaises(ValueError):
                self.build(tmp, out=Path(tmp) / 'other2', kernel_id='user/slug')


class LesionKernelTests(unittest.TestCase):
    def test_lesion_kernel_runs_the_lesion_trainer_with_its_own_template(self):
        with tempfile.TemporaryDirectory() as tmp:
            metadata = mk.build_lesion_kernel(Path(tmp) / 'k', 'netrai-x8-pilot-a', 'netrai-x8-pilot-a',
                                              ['abhikrao/netrai-lesion-2048', 'abhikrao/netrai-code-lesion'],
                                              ['--arch', 'unet', '--epochs', '12'])
            self.assertEqual(metadata['id'], 'abhikrao/netrai-x8-pilot-a')
            self.assertTrue(metadata['enable_gpu'] and metadata['is_private'])
            text = (Path(tmp) / 'k' / 'run_training.py').read_text()
            ast.parse(text)
            self.assertIn('train_lesion_seg.py', text)
            self.assertIn("'--arch', 'unet'", text)
            self.assertNotIn('train_v6', text)
            with self.assertRaises(ValueError):
                mk.build_lesion_kernel(Path(tmp) / 'k2', 'x', 'x', [], ['--dataset', 'y'])
            with self.assertRaises(FileExistsError):
                mk.build_lesion_kernel(Path(tmp) / 'k', 'netrai-x8-pilot-a', 't', [], [])


if __name__ == '__main__':
    unittest.main()
