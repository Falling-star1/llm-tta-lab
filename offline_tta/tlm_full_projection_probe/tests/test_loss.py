from pathlib import Path
import sys
import unittest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tlm_offline_v4_bundle'))
from full_projection import full_projection_cross_entropy
from full_projection import _ChunkedCE


class LossTests(unittest.TestCase):
    def test_full_projection_shape_and_half_gradient_match_whole_ce(self):
        for dtype in [torch.float16,torch.float32]:
            torch.manual_seed(42)
            head=torch.nn.Linear(8,23,bias=False).to(dtype=dtype).requires_grad_(False)
            ids=torch.randint(0,23,(2,10));mask=torch.ones_like(ids);mask[1,6:]=0
            hidden=torch.randn(2,10,8,dtype=dtype,requires_grad=True)
            logits=head(hidden)
            labels=ids[:,1:].clone();labels[mask[:,1:]==0]=-100
            counts=(labels!=-100).sum(-1)
            losses=torch.nn.functional.cross_entropy(logits[:,:-1].float().reshape(-1,23),labels.reshape(-1),reduction='none').reshape(2,-1)
            weights=torch.tensor([65536*.2,65536*.7])
            reference=losses.sum(-1)/counts
            (reference*weights).sum().backward()
            expected=hidden.grad.clone();hidden.grad=None
            calls=[];hook=head.register_forward_hook(lambda m,a,o:calls.append(a[0].shape))
            class Decoder(torch.nn.Module):
                def forward(self,**kwargs):
                    from types import SimpleNamespace
                    return SimpleNamespace(last_hidden_state=hidden)
            class Model:
                model=Decoder()
                def get_output_embeddings(self):return head
            actual=full_projection_cross_entropy(Model(),ids,mask,chunk_size=4)
            (actual*weights).sum().backward();hook.remove()
            torch.testing.assert_close(actual,reference,atol=0,rtol=0)
            torch.testing.assert_close(hidden.grad,expected,atol=0,rtol=0)
            self.assertEqual(calls,[torch.Size([2,10,8])])
            self.assertIsNone(head.weight.grad)

    def test_rejects_invalid_chunk_size(self):
        with self.assertRaises(ValueError):full_projection_cross_entropy(None,torch.ones(1,2,dtype=torch.long),chunk_size=0)

    def test_saved_logits_are_offloaded_copy_and_direct_gradient_is_exact(self):
        torch.manual_seed(13)
        logits=torch.randn(2,8,17,dtype=torch.float16,requires_grad=True)
        labels=torch.randint(0,17,(2,7));labels[1,4:]=-100
        counts=(labels!=-100).sum(-1)
        expected=torch.nn.functional.cross_entropy(logits[:,:-1].float().reshape(-1,17),labels.reshape(-1),reduction='none').reshape(2,7).sum(-1)/counts
        incoming=torch.tensor([65536*.3,0.])
        refgrad=torch.autograd.grad(expected,logits,grad_outputs=incoming)[0]
        saved=[]
        with torch.autograd.graph.saved_tensors_hooks(lambda x:saved.append(x) or x,lambda x:x):
            actual=_ChunkedCE.apply(logits,labels,counts,3)
        self.assertEqual(saved[0].device.type,'cpu')
        self.assertNotEqual(saved[0].data_ptr(),logits.data_ptr(),'Save an independent CPU tensor, not the live GPU logits')
        grad=torch.autograd.grad(actual,logits,grad_outputs=incoming)[0]
        torch.testing.assert_close(actual,expected,atol=0,rtol=0)
        torch.testing.assert_close(grad,refgrad,atol=0,rtol=0)


if __name__=='__main__':unittest.main()
