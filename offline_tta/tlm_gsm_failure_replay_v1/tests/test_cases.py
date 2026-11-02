import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cases import failure_indices, is_stub


class FailureCaseTests(unittest.TestCase):
    def rows(self):
        return [{'source_index':i, 'predict':"Let's think step by step." if i%2==0 else '42'} for i in range(20)]

    def test_selects_fixed_unique_actual_failures_without_scores(self):
        rows=self.rows()
        result=failure_indices(rows, expected_rows=20, expected_failures=10, count=4)
        self.assertEqual(result,failure_indices(rows,expected_rows=20,expected_failures=10,count=4))
        self.assertEqual(result,sorted(set(result)))
        self.assertEqual(len(result),4)
        self.assertTrue(all(i%2==0 for i in result))
        rows[0]['score']=999
        self.assertEqual(result,failure_indices(rows,expected_rows=20,expected_failures=10,count=4))

    def test_rejects_wrong_identity_count_and_index_order(self):
        for rows in [self.rows()[:-1],list(reversed(self.rows())),self.rows()[1:]+self.rows()[:1]]:
            with self.assertRaises(ValueError):failure_indices(rows,expected_rows=20,expected_failures=10,count=4)
        with self.assertRaises(ValueError):failure_indices(self.rows(),expected_rows=20,expected_failures=9,count=4)
        with self.assertRaises(ValueError):failure_indices(self.rows(),expected_rows=20,expected_failures=10,count=11)

    def test_stub_is_separate_from_empty_and_real_reasoning(self):
        self.assertTrue(is_stub("\n### Response: Let's think step by step.\n"))
        self.assertTrue(is_stub('Let’s think step by step'))
        for text in ['', '### Response:', "Let's think step by step.\n2+2=4",'42']:
            self.assertFalse(is_stub(text))


if __name__=='__main__':unittest.main()
