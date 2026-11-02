from pathlib import Path
import sys
import unittest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from projection import fp32_head


class ProjectionTests(unittest.TestCase):
    def test_fp32_head_ignores_outer_autocast_and_restores_on_error(self):
        head=torch.nn.Linear(8,11,bias=True).to(dtype=torch.float16)
        head.requires_grad_(False)
        original=head.forward
        weights=head.weight.detach().clone()
        hidden=torch.randn(1,7,8,dtype=torch.float16,requires_grad=True)
        with self.assertRaisesRegex(RuntimeError,'probe'):
            with fp32_head(head), torch.autocast('cpu',dtype=torch.bfloat16):
                output=head(hidden)
                self.assertEqual(output.dtype,torch.float32)
                expected=torch.nn.functional.linear(hidden.float(),head.weight.float(),head.bias.float())
                # Compute independent reference outside the enclosing autocast.
                with torch.autocast('cpu',enabled=False):
                    expected=torch.nn.functional.linear(hidden.float(),head.weight.float(),head.bias.float())
                torch.testing.assert_close(output,expected,atol=0,rtol=0)
                output.square().sum().backward()
                self.assertTrue(torch.isfinite(hidden.grad).all())
                self.assertIsNone(head.weight.grad)
                raise RuntimeError('probe')
        self.assertEqual(head.forward,original)
        self.assertEqual(head.weight.dtype,torch.float16)
        torch.testing.assert_close(head.weight,weights,atol=0,rtol=0)

    def test_rejects_trainable_output_head(self):
        with self.assertRaises(ValueError):
            with fp32_head(torch.nn.Linear(2,3)): pass


if __name__=='__main__': unittest.main()
