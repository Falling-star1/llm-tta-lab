"""Bounded numerical diagnostics must retain failures and restore model state."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import torch
from transformers import LlamaConfig, LlamaForCausalLM, Qwen2Config, Qwen2ForCausalLM, TrainingArguments

HERE = Path(__file__).resolve().parents[1]
V3 = HERE.parent / 'tlm_offline_v3_bundle'
sys.path.insert(0, str(V3))
from train_core import TLMTrainer, InputCollator, build_lora


def tiny_model(kind):
    torch.manual_seed(101)
    config_type, model_type = ((LlamaConfig, LlamaForCausalLM) if kind == 'llama'
                              else (Qwen2Config, Qwen2ForCausalLM))
    return model_type(config_type(vocab_size=32, hidden_size=16, intermediate_size=32,
                                 num_hidden_layers=1, num_attention_heads=2,
                                 num_key_value_heads=2, max_position_embeddings=32,
                                 pad_token_id=0, attention_dropout=0., tie_word_embeddings=False))


class DiagnosticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        path = HERE / 'diagnose.py'
        self.assertTrue(path.is_file(), 'Standalone precision diagnostic is not implemented')
        spec = importlib.util.spec_from_file_location('tlm_precision_diagnostic', path)
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.feature = {'source_index': 7, 'input_ids': [1, 3, 4, 8, 2, 9, 7, 5]}

    def trainer(self, kind='llama'):
        model = build_lora(tiny_model(kind), 42)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        args = TrainingArguments(output_dir=self.temp.name, max_steps=16,
                                 per_device_train_batch_size=1, use_cpu=True, bf16=True,
                                 report_to=[], logging_strategy='no', disable_tqdm=True,
                                 remove_unused_columns=False, seed=42)
        return TLMTrainer(model=model, args=args, train_dataset=[self.feature],
                          data_collator=InputCollator(0), threshold=0.,
                          lamb=.1 * __import__('math').exp(-3.), chunk_size=2)

    def test_statistics_retain_large_mismatch_zero_reference_and_nonfinite(self):
        stats = self.module.tensor_difference(torch.tensor([1., -2., 0.]),
                                              torch.tensor([1., 2., 0.]))
        self.assertFalse(stats['pointwise_close'])
        self.assertEqual(stats['outside_tolerance'], 1)
        self.assertAlmostEqual(stats['difference_l2'], 4.)
        self.assertLess(stats['cosine'], 0.)
        zero = self.module.tensor_difference(torch.ones(3), torch.zeros(3))
        self.assertIsNone(zero['relative_l2'])
        self.assertFalse(zero['pointwise_close'])
        bad = self.module.tensor_difference(torch.tensor([float('inf')]), torch.zeros(1))
        self.assertFalse(bad['finite'])
        json.dumps([stats, zero, bad], allow_nan=False)

    def test_both_architectures_cover_all_paths_and_leave_weights_rng_and_modes(self):
        for kind in ['llama', 'qwen']:
            with self.subTest(kind=kind):
                trainer = self.trainer(kind)
                model = trainer.model
                model.get_base_model().get_output_embeddings().eval()
                initial = {name: p.detach().clone() for name, p in model.named_parameters()}
                modes = [module.training for module in model.modules()]
                rng = torch.get_rng_state().clone()
                emitted = []
                report = self.module.collect_precision_report(trainer, self.feature, on_progress=emitted.append)
                self.assertEqual(set(report['states']), {'initial_zero_B', 'nonzero_B'})
                for state in report['states'].values():
                    self.assertEqual(set(state['paths']), {'full', 'direct_full_head', 'direct_single',
                                                          'direct_multi', 'trainer_multi'})
                    self.assertEqual(set(state['paths']['trainer_multi']['gradients']),
                                     {name for name, p in model.named_parameters() if p.requires_grad})
                    self.assertTrue(state['paths']['trainer_multi']['all_decoder_calls_under_autocast'])
                    self.assertEqual(state['paths']['direct_multi']['current_projection_sizes'], [2, 2, 2, 1])
                    self.assertEqual(state['paths']['direct_full_head']['current_projection_sizes'], [8])
                    self.assertEqual(state['paths']['direct_single']['current_projection_sizes'], [7])
                    for path in state['paths'].values():
                        self.assertIn('comparisons_with_previous_path', path)
                        self.assertTrue(path['representations']['current']['hidden_gradient']['finite'])
                        from tlm_loss import sample_weights
                        _, actual_weight = sample_weights(torch.tensor([path['base_ce']]),
                                                          trainer.threshold, trainer.lamb)
                        self.assertEqual(path['weight'], float(actual_weight[0]))
                self.assertGreater(len(emitted), 1)
                self.assertTrue(report['not_a_preflight_receipt'])
                self.assertFalse(report['formal_training_authorized_by_this_report'])
                from run import require_preflight
                report_path = Path(self.temp.name) / 'diagnostic.json'
                report_path.write_text(json.dumps(report), encoding='utf-8')
                with self.assertRaises(ValueError):
                    require_preflight(report_path, {})
                self.assertTrue(torch.equal(rng, torch.get_rng_state()))
                self.assertEqual(modes, [module.training for module in model.modules()])
                for name, p in model.named_parameters():
                    self.assertTrue(torch.equal(initial[name], p), name)
                    self.assertIsNone(p.grad)
                self.assertEqual(trainer.history, [])
                self.assertEqual(trainer.base_scores, {})
                json.dumps(report, allow_nan=False)

    def test_injected_gradient_error_collects_every_parameter_and_keeps_failure(self):
        trainer = self.trainer()
        original = trainer.compute_loss
        with mock.patch.object(trainer, 'compute_loss', side_effect=lambda *a, **k: original(*a, **k) * 1.2):
            report = self.module.collect_precision_report(trainer, self.feature)
        self.assertEqual(report['comparison_status'], 'MISMATCH')
        for state in report['states'].values():
            path = state['paths']['trainer_multi']
            self.assertFalse(path['strict_comparison_match'])
            self.assertEqual(len(path['gradients']), 4)
            self.assertTrue(any(not row['pointwise_close'] for row in path['gradients'].values()))

    def test_exception_restores_adapter_and_removes_capture_hooks(self):
        trainer = self.trainer()
        model = trainer.model
        initial = {name: p.detach().clone() for name, p in model.named_parameters() if p.requires_grad}
        decoder = model.get_base_model().model
        head = model.get_base_model().get_output_embeddings()
        hooks = (len(decoder._forward_pre_hooks), len(decoder._forward_hooks), len(head._forward_hooks))
        rng = torch.get_rng_state().clone()
        with mock.patch.object(trainer, 'compute_loss', side_effect=RuntimeError('injected execution failure')):
            with self.assertRaisesRegex(RuntimeError, 'injected execution failure'):
                self.module.collect_precision_report(trainer, self.feature)
        self.assertEqual(hooks, (len(decoder._forward_pre_hooks), len(decoder._forward_hooks),
                                 len(head._forward_hooks)))
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        for name, p in model.named_parameters():
            if name in initial:
                self.assertTrue(torch.equal(initial[name], p), name)
            self.assertIsNone(p.grad)
        self.assertEqual(trainer.history, [])
        self.assertEqual(trainer.base_scores, {})

    def test_nonfinite_reference_gradients_finish_both_states_with_json_safe_failure(self):
        trainer = self.trainer()
        parameter = next(p for name, p in trainer.model.named_parameters() if 'lora_B' in name)
        hook = parameter.register_hook(lambda gradient: torch.full_like(gradient, float('inf')))
        try:
            report = self.module.collect_precision_report(
                trainer, self.feature, on_progress=lambda row: json.dumps(row, allow_nan=False))
        finally:
            hook.remove()
        self.assertEqual(report['comparison_status'], 'MISMATCH')
        self.assertEqual(len(report['states']), 2)
        for state in report['states'].values():
            self.assertEqual(len(state['paths']), 5)
            self.assertIsNone(state['reference_factor_norms']['lora_B'])
            self.assertFalse(state['paths']['full']['strict_comparison_match'])
        json.dumps(report, allow_nan=False)


if __name__ == '__main__':
    unittest.main()
