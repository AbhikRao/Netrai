import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluate_ma_froc import match_candidates, native_coordinates


class MAFROCTests(unittest.TestCase):
    def test_one_to_one_matches_and_false_alarms(self):
        truth = np.zeros((32, 32), dtype=bool)
        truth[10:13, 10:13] = True
        truth[20:23, 20:23] = True
        centres = [(11, 11), (12, 12), (21, 21), (3, 3)]
        scores = [5, 4, 3, 6]
        self.assertEqual(match_candidates(centres, scores, truth, 3, tolerance=1), (2, 2, 2))
        self.assertEqual(match_candidates(centres, scores, truth, 4.5, tolerance=1), (1, 1, 2))

    def test_native_mapping_inverts_padding_and_resize(self):
        mapped = native_coordinates([(256, 256)], 1024, 512, 512)
        self.assertAlmostEqual(mapped[0][0], 512.5)
        self.assertAlmostEqual(mapped[0][1], 256.5)


if __name__ == '__main__':
    unittest.main()
