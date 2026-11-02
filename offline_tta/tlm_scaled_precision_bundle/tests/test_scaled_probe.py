import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import torch
from transformers import LlamaConfig, LlamaForCausalLM, Qwen2Config, Qwen2ForCausalLM, TrainingArguments, TrainerCallback

HERE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(HERE), str(HERE.parent / 'tlm_offline_v3_bundle')]
from train_core import TLMTrainer, InputCollator, build_lora


class ScaledProbeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.assertTrue((HERE / 'scaled_probe.py').is_file(), 'Scaled gradient probe is not implemented')
        import scaled_probe
        import trajectory
        self.probe, self.trajectory = scaled_probe, trajectory
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        config = LlamaConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
                             num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
                             max_position_embeddings=32, pad_token_id=0, attention_dropout=0.)
        torch.manual_seed(101)
        self.model = build_lora(LlamaForCausalLM(config), 42)
        self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        self.features = [{'source_index': i, 'input_ids': [1, 3, 4, 8, 2, 7, 9, 5]}
                         for i in range(4)]

    def create(self, name='carrier', callbacks=None):
        args = TrainingArguments(output_dir=str(Path(self.temp.name) / name), max_steps=4,
                                 per_device_train_batch_size=1, use_cpu=True, bf16=True,
                                 learning_rate=1e-3, warmup_ratio=.1, lr_scheduler_type='cosine',
                                 optim='adamw_torch', weight_decay=0., max_grad_norm=1.,
                                 save_strategy='no', report_to=[], logging_strategy='no',
                                 disable_tqdm=True, seed=42, remove_unused_columns=False)
        trainer = TLMTrainer(model=self.model, args=args, train_dataset=self.features,
                             data_collator=InputCollator(0), threshold=0.,
                             lamb=.1 * __import__('math').exp(-3.), chunk_size=2,
                             callbacks=callbacks or [])
        return trainer

    def test_scaled_backward_retains_fp16_intermediate_gradient_and_unscales(self):
        p = torch.nn.Parameter(torch.tensor([1.]))
        scaler = torch.amp.GradScaler('cpu', init_scale=65536.)
        from accelerate import Accelerator
        accelerator = Accelerator(cpu=True, mixed_precision='bf16')
        accelerator.scaler = scaler
        optimizer = torch.optim.AdamW([p])
        loss = (p.half().float() * 1e-8).sum()
        info = self.probe.scaled_backward(accelerator, optimizer, loss)
        self.assertTrue(info['finite'])
        self.assertGreater(float(p.grad[0]), 0.)
        self.assertLess(float(p.grad[0]), 2e-8)
        self.assertEqual(info['scale_used'], 65536.)

    def test_probe_restores_scaler_weights_modes_rng_and_uses_same_scale(self):
        trainer = self.create()
        trainer.accelerator.scaler = torch.amp.GradScaler('cpu', init_scale=64.)
        original = trainer.accelerator.scaler
        initial = {n: p.detach().clone() for n, p in self.model.named_parameters()}
        rng = torch.get_rng_state().clone()
        report = self.probe.compare_scaled_gradients(trainer, self.features[0])
        self.assertTrue(report['not_a_preflight_receipt'])
        for state in report['states'].values():
            self.assertEqual(state['finite_scale'], 64.)
            self.assertEqual(state['paths']['full']['scale_used'], state['paths']['trainer_multi']['scale_used'])
            self.assertEqual(set(state['paths']), {'full', 'trainer_multi', 'trainer_multi_no_checkpoint'})
            self.assertTrue(state['paths']['trainer_multi']['all_decoder_calls_under_autocast'])
        self.assertEqual(report['states']['initial_zero_B']['reference_factor_norms']['lora_A'], 0.)
        self.assertGreater(report['states']['nonzero_B']['reference_factor_norms']['lora_A'], 0.)
        self.assertGreater(report['states']['nonzero_B']['reference_factor_norms']['lora_B'], 0.)
        self.assertIs(trainer.accelerator.scaler, original)
        self.assertEqual(original.get_scale(), 64.)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        for name, p in self.model.named_parameters():
            self.assertTrue(torch.equal(initial[name], p), name)
            self.assertIsNone(p.grad)
        self.assertEqual(trainer.history, [])
        self.assertEqual(trainer.base_scores, {})
        json.dumps(report, allow_nan=False)

    def test_qwen_uses_actual_scaled_path_with_finite_normalized_activation_gradient(self):
        config = Qwen2Config(vocab_size=32, hidden_size=16, intermediate_size=32,
                             num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
                             max_position_embeddings=32, pad_token_id=0, attention_dropout=0.)
        self.model = build_lora(Qwen2ForCausalLM(config), 42)
        self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        trainer = self.create()
        trainer.accelerator.scaler = torch.amp.GradScaler('cpu', init_scale=64.)
        report = self.probe.compare_scaled_gradients(trainer, self.features[0])
        self.assertEqual(len(report['states']), 2)
        for state in report['states'].values():
            for path in state['paths'].values():
                self.assertTrue(path['representations']['current']['hidden_gradient']['finite'])
                self.assertEqual(path['scale_used'], 64.)

    def test_trajectory_exception_restores_weights_forward_and_full_loss_function(self):
        import train_core
        function = train_core.model_input_cross_entropy
        carrier = self.create()
        forward = self.model.forward
        initial = {name: p.detach().clone() for name, p in self.model.named_parameters()}
        class FailAtOne(TrainerCallback):
            def on_step_end(self, *args, **kwargs):
                raise RuntimeError('trajectory execution failure')
        def create(name, callbacks=None):
            return self.create(name, callbacks=(callbacks or []) + [FailAtOne()])
        with self.assertRaisesRegex(RuntimeError, 'trajectory execution failure'):
            self.trajectory.compare_trajectories(carrier, create)
        self.assertIs(train_core.model_input_cross_entropy, function)
        self.assertEqual(self.model.forward, forward)
        for name, p in self.model.named_parameters():
            self.assertTrue(torch.equal(initial[name], p), name)
            self.assertIsNone(p.grad)

    def test_overflow_backoff_is_joint_and_stops_at_first_finite_scale(self):
        trainer = self.create()
        trainer.accelerator.scaler = torch.amp.GradScaler('cpu', init_scale=64.)
        p = next(p for name, p in self.model.named_parameters() if 'lora_B' in name)
        hook = p.register_hook(lambda grad: torch.full_like(grad, float('inf'))
                               if trainer.accelerator.scaler.get_scale() > 16. else grad)
        try:
            report = self.probe.compare_scaled_gradients(trainer, self.features[0])
        finally:
            hook.remove()
        for state in report['states'].values():
            self.assertEqual([attempt['scale'] for attempt in state['attempts']], [64., 32., 16.])
            self.assertEqual(state['finite_scale'], 16.)
        self.assertEqual(trainer.accelerator.scaler.get_scale(), 64.)
        json.dumps(report, allow_nan=False)

    def test_probe_exception_restores_scaler_and_checkpoint_function(self):
        trainer = self.create()
        trainer.accelerator.scaler = torch.amp.GradScaler('cpu', init_scale=64.)
        original = trainer.accelerator.scaler
        import tlm_loss
        checkpoint = tlm_loss.checkpoint
        initial = {n: p.detach().clone() for n, p in self.model.named_parameters() if p.requires_grad}
        with mock.patch.object(trainer, 'compute_loss', side_effect=RuntimeError('execution failure')):
            with self.assertRaisesRegex(RuntimeError, 'execution failure'):
                self.probe.compare_scaled_gradients(trainer, self.features[0])
        self.assertIs(trainer.accelerator.scaler, original)
        self.assertIs(tlm_loss.checkpoint, checkpoint)
        for name, p in self.model.named_parameters():
            if name in initial:
                self.assertTrue(torch.equal(initial[name], p), name)
            self.assertIsNone(p.grad)

    def test_real_trainer_trajectory_collects_updates_and_restores_forward(self):
        carrier = self.create()
        forward = self.model.forward
        initial = {name: p.detach().clone() for name, p in self.model.named_parameters()}
        report = self.trajectory.compare_trajectories(carrier, self.create)
        self.assertEqual(report['steps_per_path'], 4)
        self.assertEqual(len(report['steps']), 4)
        self.assertTrue(report['nonzero_parameter_updates'])
        self.assertTrue(report['all_parameter_states_finite'])
        self.assertTrue(report['same_step_count'])
        self.assertEqual(report['full_source_order'], report['chunk_source_order'])
        self.assertTrue(report['not_a_preflight_receipt'])
        self.assertEqual(self.model.forward, forward)
        for name, p in self.model.named_parameters():
            self.assertTrue(torch.equal(initial[name], p), name)
            self.assertIsNone(p.grad)
        json.dumps(report, allow_nan=False)


if __name__ == '__main__':
    unittest.main()
