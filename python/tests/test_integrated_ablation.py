import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluate_integrated_ablation import ablation_row


class IntegratedAblationTests(unittest.TestCase):
    def test_unresolved_cases_are_not_counted_as_correct(self):
        result = ablation_row([2, 0, 2, 0], [0, 0, 2, 2], [1, 0, 2, 0], 'test')
        self.assertEqual(result['auto_decisions'], 2)
        self.assertEqual(result['unresolved_referable'], 2)
        self.assertEqual(result['auto_tp'], 0)
        self.assertEqual(result['auto_fn'], 0)
        self.assertEqual(result['auto_fp'], 1)
        self.assertEqual(result['recaptures'], 1)
        self.assertEqual(result['human_reviews'], 1)


if __name__ == '__main__':
    unittest.main()
