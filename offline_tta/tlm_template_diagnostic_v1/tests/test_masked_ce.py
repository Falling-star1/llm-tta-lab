import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(1,str(Path(__file__).resolve().parents[2]/'tlm_offline_v6_bundle'))
import torch
from torch import nn
from masked_ce import masked_input_cross_entropy


class Decoder(nn.Module):
    def __init__(self):
        super().__init__();self.embedding=nn.Embedding(11,5)
    def forward(self,input_ids,**kwargs):return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


class Tiny(nn.Module):
    def __init__(self):
        super().__init__();self.model=Decoder();self.head=nn.Linear(5,11)
    def get_output_embeddings(self):return self.head


class MaskedCETests(unittest.TestCase):
    def test_masked_loss_and_gradient_match_explicit_targets(self):
        torch.manual_seed(42);m=Tiny();ids=torch.tensor([[1,3,4,5,6]])
        mask=torch.tensor([[True,False,True,False,True]])
        actual=masked_input_cross_entropy(m,ids,torch.ones_like(ids),mask,2)
        actual.sum().backward();got=m.head.weight.grad.clone();m.zero_grad()
        logits=m.head(m.model(ids).last_hidden_state)
        targets=ids[:,1:].clone().masked_fill(~mask[:,1:],-100)
        expected=torch.nn.functional.cross_entropy(logits[:,:-1].reshape(-1,11),targets.reshape(-1),reduction='sum')/4
        expected.backward()
        torch.testing.assert_close(actual[0],expected)
        torch.testing.assert_close(got,m.head.weight.grad)

    def test_all_targets_matches_original_full_loss(self):
        from tlm_loss import input_cross_entropy
        torch.manual_seed(42);m=Tiny();ids=torch.tensor([[1,3,4,5,6]])
        got=masked_input_cross_entropy(m,ids,torch.ones_like(ids),torch.ones_like(ids,dtype=torch.bool),2)
        expected=input_cross_entropy(m.head(m.model(ids).last_hidden_state),ids)
        torch.testing.assert_close(got,expected)

    def test_empty_target_mask_rejected(self):
        m=Tiny();ids=torch.tensor([[1,3]])
        with self.assertRaises(ValueError):
            masked_input_cross_entropy(m,ids,torch.ones_like(ids),torch.zeros_like(ids,dtype=torch.bool),2)


if __name__=='__main__':unittest.main()
