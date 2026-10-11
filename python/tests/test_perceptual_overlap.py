import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit_perceptual_overlap import candidates, scan_candidates
from verify_perceptual_candidates import geometric_evidence, resolve_image


class PerceptualOverlapTests(unittest.TestCase):
    def test_flags_close_cross_source_hashes_only(self):
        left = [(Path('a'), 0), (Path('b'), (1 << 64)-1)]
        spread = sum(1 << bit for bit in (0, 13, 26, 39))
        right = [(Path('c'), 0b1111), (Path('d'), 0xAAAAAAAAAAAAAAAA),
                 (Path('e'), spread)]
        self.assertEqual(candidates(left, right, 4),
                         [('a', 'c', 4), ('a', 'e', 4)])
        capped = scan_candidates(left, right, 4, max_stored=1)
        self.assertEqual(capped['total'], 2)
        self.assertEqual(capped['by_distance'][4], 2)
        self.assertEqual(len(capped['pairs']), 1)
        with tempfile.TemporaryDirectory() as directory:
            rng = np.random.default_rng(4)
            first = rng.integers(0, 256, (1024, 1024), dtype=np.uint8)
            unrelated = rng.integers(0, 256, (1024, 1024), dtype=np.uint8)
            paths = [Path(directory)/name for name in ('a.png', 'b.png', 'c.png')]
            for path, pixels in zip(paths, (first, first, unrelated)):
                self.assertTrue(cv2.imwrite(str(path), pixels))
            self.assertTrue(geometric_evidence(paths[0], paths[1])['geometrically_consistent'])
            self.assertFalse(geometric_evidence(paths[0], paths[2])['geometrically_consistent'])
            self.assertEqual(resolve_image({'sample': Path(directory)}, 'sample', 'a.png'),
                             paths[0].resolve())
            with self.assertRaises(FileNotFoundError):
                resolve_image({'sample': Path(directory)}, 'sample', '../outside.png')


if __name__ == '__main__':
    unittest.main()
