"""Runtime identity and local file handling, without a downloaded model."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    import runtime
except ImportError:
    runtime = None


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(runtime, 'Runtime is not implemented')

    def test_input_loader_verifies_hash_count_order_and_no_answers(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rows = [{'source_index':0,'instruction':'Question','input':''}]
            raw = (json.dumps(rows[0]) + '\n').encode()
            (root/'inputs.jsonl').write_bytes(raw)
            manifest = {'record_count':1,'source_sha256':'a'*64,
                        'files':{'inputs.jsonl':hashlib.sha256(raw).hexdigest()}}
            (root/'manifest.json').write_text(json.dumps(manifest))
            got, identity = runtime.read_inputs(root, 'a'*64, expected_count=1)
            self.assertEqual(got, rows)
            (root/'inputs.jsonl').write_bytes(raw+b' ')
            with self.assertRaises(ValueError):
                runtime.read_inputs(root, 'a'*64, expected_count=1)

    def test_atomic_json_rejects_nonfinite_values(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'record.json'
            runtime.atomic_json(path, {'ok':1})
            self.assertEqual(json.loads(path.read_text()), {'ok':1})
            with self.assertRaises(ValueError):
                runtime.atomic_json(path, {'bad':float('nan')})
            self.assertEqual(json.loads(path.read_text()), {'ok':1})

    def test_live_output_lock_prevents_second_owner(self):
        with tempfile.TemporaryDirectory() as temp:
            with runtime.output_lock(Path(temp)):
                with self.assertRaises((BlockingIOError, OSError)):
                    with runtime.output_lock(Path(temp)):
                        self.fail('Second owner entered the same output')

    def test_scoring_matches_fixed_gsm8k_reference_convention(self):
        scorer = runtime.Scorer('exact_match', extractor=lambda text: '123')
        self.assertEqual(scorer.score('answer','123\n'), 1.)
        self.assertEqual(scorer.score('answer','124'), 0.)

    def test_idle_check_waits_out_previous_process_utilization_sample(self):
        with mock.patch('runtime.subprocess.check_output',side_effect=['1, 99\n','1, 0\n']) as query:
            with mock.patch('runtime.time.sleep') as sleep:
                runtime.wait_for_idle_gpu('0',attempts=2)
        self.assertEqual(query.call_count,2)
        sleep.assert_called_once()

    def test_training_disables_reduced_precision_fp16_reduction(self):
        import os
        import torch
        previous = torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
        self.addCleanup(setattr, torch.backends.cuda.matmul, 'allow_fp16_reduced_precision_reduction', previous)
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
        with mock.patch.dict(os.environ, {'CUBLAS_WORKSPACE_CONFIG':':4096:8'}):
            runtime.configure(42)
        self.assertFalse(torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction)

    def test_generation_keeps_source_reduction_mode_and_restores_training_mode(self):
        import torch
        from types import SimpleNamespace
        seen = []
        class Model:
            fail = False
            def gradient_checkpointing_disable(self): pass
            def eval(self): pass
            def generate(self, input_ids, **kwargs):
                seen.append(torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction)
                if self.fail:
                    raise RuntimeError('generation failure')
                return torch.cat([input_ids,torch.tensor([[3,4]])],dim=1)
        loaded = object.__new__(runtime.LoadedModel)
        loaded.model=Model();loaded.device='cpu';loaded.stop_ids=[2];loaded.pad_id=0
        loaded.tokenizer=SimpleNamespace(decode=lambda ids,**kwargs:'text')
        previous = torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
        self.addCleanup(setattr, torch.backends.cuda.matmul, 'allow_fp16_reduced_precision_reduction', previous)
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction=False
        self.assertEqual(loaded.generate([1,2])['generated_ids'],[3,4])
        self.assertEqual(seen,[True])
        self.assertFalse(torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction)
        loaded.model.fail=True
        with self.assertRaisesRegex(RuntimeError,'generation failure'):
            loaded.generate([1,2])
        self.assertFalse(torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction)


if __name__ == '__main__':
    unittest.main()
