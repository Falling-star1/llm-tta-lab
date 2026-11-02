import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
try:
    import run
except ImportError:
    run=None


class CommandTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(run,'Commands are not implemented')

    def test_forced_probe_selection_keeps_formal_weight_magnitude(self):
        import torch
        from tlm_loss import sample_weights
        scores=torch.tensor([2.,4.,5.5])
        selected,probe=sample_weights(scores,threshold=0.,lamb=run.PROBE_LAMB)
        _,formal=sample_weights(scores,threshold=3.,lamb=.1)
        self.assertTrue(bool(selected.all()))
        torch.testing.assert_close(probe,formal)

    def test_probe_budget_survives_initial_fp16_scaler_backoff(self):
        import torch
        def train(steps):
            parameter=torch.nn.Parameter(torch.ones(1,1))
            optimizer=torch.optim.AdamW([parameter],lr=1e-3)
            scaler=torch.amp.GradScaler('cpu')
            for _ in range(steps):
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cpu',dtype=torch.float16):
                    prediction=torch.nn.functional.linear(torch.full((1,1),10.),parameter)
                loss=prediction.float().sum()*10.
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            return parameter.detach(),optimizer.state
        short,state=train(4)
        self.assertTrue(torch.equal(short,torch.ones(1,1)))
        self.assertFalse(state)
        final,state=train(run.PROBE_STEPS)
        self.assertFalse(torch.equal(final,torch.ones(1,1)))
        self.assertGreaterEqual(min(float(v['step']) for v in state.values()),2)
        self.assertLess(run.PROBE_INTERRUPT,run.PROBE_STEPS)

    def test_probe_arguments_keep_sufficient_budget_and_formal_recipe(self):
        from types import SimpleNamespace
        args=SimpleNamespace()
        with mock.patch.object(run,'training_arguments',return_value=args) as factory:
            got=run.probe_training_arguments(Path('probe'),42)
        self.assertIs(got,args)
        self.assertEqual(got.max_steps,run.PROBE_STEPS)
        self.assertEqual(got.save_steps,run.PROBE_INTERRUPT)
        factory.assert_called_once_with(Path('probe'),1e-6,42,run.PROBE_STEPS)

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
