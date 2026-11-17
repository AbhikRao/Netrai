"""Tests for blind extraction: zip, tar.gz, directory and nested-archive releases; labels sealed, names never printed, unsafe paths refused."""
import contextlib
import csv
import io
import tarfile
import tempfile
import unittest
import zipfile
from pathlib import Path

import blind_extract as be

GRADE_TREE = {'release/0/a.jpg': b'retina-a', 'release/2/b.JPG': b'retina-b', 'release/2/c.png': b'retina-a',   # c duplicates a
              'release/labels.csv': b'image,grade\na.jpg,0\nb.JPG,2\n', 'release/README.txt': b'grades in folders'}


def write_zip(path: Path, files: dict) -> None:
    with zipfile.ZipFile(path, 'w') as zf:
        for name, data in files.items():
            zf.writestr(name, data)


def write_tar(path: Path, files: dict) -> None:
    with tarfile.open(path, 'w:gz') as tf:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))


class BlindExtractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)

    def extract(self, source, name='set'):
        out = self.base / name
        printed = io.StringIO()
        with contextlib.redirect_stdout(printed):
            be.main(['--archive', str(source), '--out', str(out), '--sealed', str(out / 'sealed'), '--prefix', 'x_'])
        return out, printed.getvalue()

    def check(self, out, printed, prefix=''):
        images = sorted(p.name for p in (out / 'images').iterdir())
        self.assertEqual(len(images), 2)                                         # a and b; c is a byte-identical copy of a
        self.assertTrue(all(name.startswith('x_') for name in images))
        self.assertIn("'byte_identical_duplicates': 1", printed)
        for secret in ('release', 'a.jpg', 'b.JPG', 'labels', '/0/', '/2/'):     # counts only: no path, name or grade folder
            self.assertNotIn(secret, printed)
        with open(out / 'sealed/mapping.csv') as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(sorted(r['archive_path'] for r in rows), sorted(prefix + k for k in GRADE_TREE if not k.endswith(('.csv', '.txt'))))
        self.assertEqual(len({r['image_id'] for r in rows}), 2)
        sealed = sorted(p.relative_to(out / 'sealed').as_posix() for p in (out / 'sealed').rglob('*') if p.is_file())
        self.assertIn('mapping.csv', sealed)
        self.assertTrue(any(s.endswith('release/labels.csv') for s in sealed))
        self.assertFalse(any(s.endswith(('.jpg', '.JPG', '.png')) for s in sealed))

    def test_zip_release(self):
        write_zip(self.base / 'r.zip', GRADE_TREE)
        self.check(*self.extract(self.base / 'r.zip'))

    def test_tar_gz_release(self):
        write_tar(self.base / 'r.tar.gz', GRADE_TREE)
        self.check(*self.extract(self.base / 'r.tar.gz'))

    def test_directory_release(self):
        for name, data in GRADE_TREE.items():
            (self.base / 'dl' / name).parent.mkdir(parents=True, exist_ok=True)
            (self.base / 'dl' / name).write_bytes(data)
        self.check(*self.extract(self.base / 'dl'))

    def test_directory_with_a_nested_archive(self):
        (self.base / 'dl').mkdir()
        write_zip(self.base / 'dl/images.zip', GRADE_TREE)
        out, printed = self.extract(self.base / 'dl')
        self.check(out, printed, prefix='images.zip!/')

    def test_unsafe_member_paths_are_refused(self):
        write_zip(self.base / 'evil.zip', {'../escape.csv': b'x', 'ok/a.jpg': b'retina'})
        with self.assertRaises(ValueError):
            self.extract(self.base / 'evil.zip')
        self.assertFalse((self.base / 'escape.csv').exists())
        self.assertFalse((self.base / 'set/escape.csv').exists())

    def test_extraction_happens_once(self):
        write_zip(self.base / 'r.zip', GRADE_TREE)
        self.extract(self.base / 'r.zip')
        with self.assertRaises(FileExistsError):
            self.extract(self.base / 'r.zip')

    def test_other_inputs_are_refused(self):
        (self.base / 'labels.csv').write_text('x')
        with self.assertRaises(ValueError):
            self.extract(self.base / 'labels.csv')


if __name__ == '__main__':
    unittest.main()
