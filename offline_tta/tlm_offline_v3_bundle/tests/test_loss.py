"""Independent checks of the frozen TLM objective and memory implementation."""
import math
from pathlib import Path
import sys
import unittest

import torch
from torch import nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import tlm_loss
except ImportError:
    tlm_loss = None


class TinyDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(11, 6)
        self.proj = nn.Linear(6, 6)

    def forward(self, input_ids, attention_mask=None, use_cache=False, return_dict=True):
        # Causal dependence makes gradients flow through earlier positions too.
        hidden = self.proj(self.embed(input_ids).cumsum(dim=1)).tanh()
        from types import SimpleNamespace
        return SimpleNamespace(last_hidden_state=hidden)


class TinyLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = TinyDecoder()
        self.lm_head = nn.Linear(6, 11, bias=False)

    def get_output_embeddings(self):
        return self.lm_head

    def get_base_model(self):
        return self


class LossTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(tlm_loss, 'TLM loss component is not implemented')
        torch.manual_seed(9)

    def test_uniform_logits_use_valid_shifted_token_count(self):
        ids = torch.tensor([[1, 2, 3, 0], [2, 3, 0, 0]])
        mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]])
        got = tlm_loss.input_cross_entropy(torch.zeros(2, 4, 5), ids, mask)
        torch.testing.assert_close(got, torch.full((2,), math.log(5)))

    def test_causal_shift_predicts_next_input_token(self):
        ids = torch.tensor([[1, 2, 0]])
        logits = torch.tensor([[[0., 0., 6.], [5., 0., 0.], [100., 0., 0.]]])
        expected = (F.cross_entropy(logits[:, 0], ids[:, 1]) +
                    F.cross_entropy(logits[:, 1], ids[:, 2])) / 2
        torch.testing.assert_close(tlm_loss.input_cross_entropy(logits, ids)[0], expected)

    def test_real_eos_is_not_removed_when_pad_has_same_id(self):
        ids = torch.tensor([[1, 0, 0]])
        mask = torch.tensor([[1, 1, 0]])
        logits = torch.randn(1, 3, 4, requires_grad=True)
        expected = F.cross_entropy(logits[:, 0], torch.tensor([0]))
        torch.testing.assert_close(tlm_loss.input_cross_entropy(logits, ids, mask)[0], expected)

    def test_ce_matches_official_one_hot_kl_value_and_gradient(self):
        ids = torch.tensor([[1, 2, 3, 0]])
        mask = torch.tensor([[1, 1, 1, 0]])
        logits = torch.randn(1, 4, 5, requires_grad=True)
        actual = tlm_loss.input_cross_entropy(logits, ids, mask).sum()
        positions = logits[0, :2].log_softmax(-1)
        target = F.one_hot(ids[0, 1:3], num_classes=5).float()
        official = F.kl_div(positions, target, reduction='batchmean')
        ga, = torch.autograd.grad(actual, logits, retain_graph=True)
        gb, = torch.autograd.grad(official, logits)
        torch.testing.assert_close(actual, official, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(ga, gb, atol=1e-6, rtol=1e-5)

    def test_strict_threshold_and_detached_base_weights(self):
        base = torch.tensor([2., 3., 4.], requires_grad=True)
        selected, weights = tlm_loss.sample_weights(base)
        self.assertEqual(selected.tolist(), [False, False, True])
        torch.testing.assert_close(weights, .1 * torch.exp(base.detach() - 3))
        current = torch.tensor([1., 2., 3.], requires_grad=True)
        loss = tlm_loss.weighted_input_loss(current, base)
        loss.backward()
        self.assertIsNone(base.grad)
        torch.testing.assert_close(current.grad, torch.tensor([0., 0., .1 * math.e]))

    def test_no_selected_samples_keep_zero_gradient_graph(self):
        current = torch.tensor([2.], requires_grad=True)
        loss = tlm_loss.weighted_input_loss(current, torch.tensor([2.]))
        loss.backward()
        self.assertEqual(loss.item(), 0.)
        self.assertEqual(current.grad.item(), 0.)

    def test_empty_targets_nonfinite_and_bad_shapes_are_rejected(self):
        for ids, mask in [(torch.tensor([[1]]), None),
                          (torch.tensor([[1, 2]]), torch.tensor([[1, 0]]))]:
            with self.assertRaises(ValueError):
                tlm_loss.input_cross_entropy(torch.zeros(1, ids.shape[1], 4), ids, mask)
        with self.assertRaises(ValueError):
            tlm_loss.input_cross_entropy(torch.zeros(1, 3, 4), torch.tensor([[1, 2]]))
        with self.assertRaises(FloatingPointError):
            tlm_loss.sample_weights(torch.tensor([float('nan')]))
        with self.assertRaises(FloatingPointError):
            tlm_loss.sample_weights(torch.tensor([1000.]))
        with self.assertRaises(FloatingPointError):
            tlm_loss.input_cross_entropy(torch.full((1, 2, 4), float('inf')),
                                         torch.tensor([[1, 2]]))

    def test_chunked_projection_matches_full_loss_and_all_gradients(self):
        import copy
        full, chunked = TinyLM(), None
        chunked = copy.deepcopy(full)
        ids = torch.tensor([[1, 2, 3, 0, 0], [2, 4, 1, 5, 3]])
        mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]])
        hidden = full.model(ids).last_hidden_state
        expected = tlm_loss.input_cross_entropy(full.lm_head(hidden), ids, mask)
        got = tlm_loss.model_input_cross_entropy(chunked, ids, mask, chunk_size=2)
        expected.sum().backward()
        got.sum().backward()
        torch.testing.assert_close(got, expected, atol=1e-6, rtol=1e-5)
        for (name, p), (_, q) in zip(full.named_parameters(), chunked.named_parameters()):
            with self.subTest(name=name):
                torch.testing.assert_close(p.grad, q.grad, atol=1e-6, rtol=1e-5)

    def test_chunked_scoring_works_without_gradients(self):
        with torch.no_grad():
            got = tlm_loss.model_input_cross_entropy(TinyLM(), torch.tensor([[1, 2, 3]]), chunk_size=1)
        self.assertEqual(tuple(got.shape), (1,))
        self.assertFalse(got.requires_grad)

    def test_training_rejects_left_padding_and_internal_mask_gaps(self):
        ids = torch.tensor([[0, 0, 1, 2]])
        for mask in [torch.tensor([[0, 0, 1, 1]]), torch.tensor([[1, 0, 1, 0]])]:
            with self.assertRaises(ValueError):
                tlm_loss.input_cross_entropy(torch.zeros(1, 4, 4), ids, mask)

    def test_partial_final_projection_chunk_excludes_last_unused_hidden(self):
        model=TinyLM()
        ids=torch.tensor([[1,2,3,4]])
        expected=tlm_loss.input_cross_entropy(model.lm_head(model.model(ids).last_hidden_state),ids)
        actual=tlm_loss.model_input_cross_entropy(model,ids,chunk_size=2)
        torch.testing.assert_close(actual,expected)


if __name__ == '__main__':
    unittest.main()
