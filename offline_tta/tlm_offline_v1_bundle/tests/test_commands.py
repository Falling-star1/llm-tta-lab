import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
try:
    import run
except ImportError:
    run=None


class CommandTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(run,'Commands are not implemented')

    def test_formal_requires_matching_successful_preflight(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'receipt.json'
            expected={'code_id':'code','model':'llama3b','dataset':'geosignal_5k',
                      'input_sha256':'input'}
            row={**expected,'status':'PASS','frozen_base_unchanged':True,
                 'checkpoint_continuation':'EXACT_MATCH','source_sample_match':True}
            path.write_text(json.dumps(row))
            self.assertEqual(run.require_preflight(path,expected),row)
            for field,value in [('status','FAIL'),('code_id','old'),
                                ('checkpoint_continuation','NOT_CHECKED'),
                                ('source_sample_match',False)]:
                broken={**row,field:value}
                path.write_text(json.dumps(broken))
                with self.assertRaises(ValueError):
                    run.require_preflight(path,expected)

    def test_evaluation_journal_recovers_only_incomplete_tail(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'predictions.jsonl'
            first={'source_index':0,'predict':'one'}
            path.write_bytes((json.dumps(first)+'\n{"source_index":1').encode())
            self.assertEqual(run.read_predictions(path),[first])
            self.assertTrue(path.read_bytes().endswith(b'\n'))
            path.write_text(json.dumps({'source_index':1})+'\n')
            with self.assertRaises(ValueError):
                run.read_predictions(path)


if __name__=='__main__':
    unittest.main()
