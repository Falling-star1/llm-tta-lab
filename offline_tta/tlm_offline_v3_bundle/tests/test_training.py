"""Real pinned Transformers/PEFT training and exact CPU resume checks."""
import copy
import importlib
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import torch
from transformers import (LlamaConfig, LlamaForCausalLM, Qwen2Config,
                          Qwen2ForCausalLM, TrainerCallback, TrainingArguments)

BUNDLE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BUNDLE))
from tlm_loss import input_cross_entropy, model_input_cross_entropy


def tiny_model(kind='llama'):
    torch.manual_seed(101)
    config_class, model_class = ((LlamaConfig, LlamaForCausalLM) if kind == 'llama'
                                else (Qwen2Config, Qwen2ForCausalLM))
    config = config_class(vocab_size=32, hidden_size=16, intermediate_size=32,
                          num_hidden_layers=1, num_attention_heads=2,
                          num_key_value_heads=2, max_position_embeddings=32,
                          pad_token_id=0, bos_token_id=1, eos_token_id=2,
                          attention_dropout=0., tie_word_embeddings=False)
    return model_class(config)


def adapter_state(model):
    return {name: parameter.detach().clone() for name, parameter in model.named_parameters()
            if parameter.requires_grad}


class StopAtTwo(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == 2:
            control.should_training_stop = True
            control.should_save = True
        return control


class RejectSecond(TrainerCallback):
    def __init__(self):
        self.trainer = None
        self.after_first = None

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step == 1:
            self.after_first = adapter_state(model)
            self.trainer.threshold = 1000.
        return control


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.assertTrue((BUNDLE / 'train_core.py').is_file(), 'TLM Trainer implementation is missing')
        self.module = importlib.import_module('train_core')
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def args(self, name='run', **changes):
        values = dict(output_dir=str(self.root / name), max_steps=4,
                      per_device_train_batch_size=1, gradient_accumulation_steps=1,
                      learning_rate=.01, optim='adamw_torch', weight_decay=0.,
                      lr_scheduler_type='cosine', warmup_ratio=.1, max_grad_norm=1.,
                      save_strategy='steps', save_steps=2, save_total_limit=2,
                      logging_strategy='no', report_to=[], disable_tqdm=True,
                      remove_unused_columns=False, use_cpu=True, fp16=False,
                      seed=42, dataloader_num_workers=0)
        values.update(changes)
        return TrainingArguments(**values)

    def trainer(self, name='run', model=None, callbacks=None, **kwargs):
        args = kwargs.pop('args', self.args(name))
        return self.module.TLMTrainer(
            model=model if model is not None else self.module.build_lora(tiny_model(), 42),
            args=args, train_dataset=[{'source_index': index, 'input_ids': ids}
                                     for index, ids in enumerate([[1, 3, 4, 5], [1, 8, 9],
                                                                  [1, 6, 7, 8, 9], [1, 5, 8]])],
            data_collator=self.module.InputCollator(0), callbacks=callbacks or [],
            threshold=0., chunk_size=2, **kwargs)

    def test_arguments_support_pinned_accelerate_before_1_1(self):
        import transformers.training_args as training_args
        available = training_args.is_accelerate_available
        with mock.patch.object(training_args, 'is_accelerate_available',
                               side_effect=lambda version=None: False if version == '1.1.0'
                               else (available() if version is None else available(version))):
            args = self.args()
        self.assertEqual(args.seed, 42)
        self.assertIsNone(args.data_seed)

    def test_lora_uses_paper_gaussian_initialization_and_freezes_base(self):
        for kind in ['llama', 'qwen']:
            with self.subTest(kind=kind):
                model = self.module.build_lora(tiny_model(kind), 42)
                self.assertEqual(model.peft_config['default'].init_lora_weights, 'gaussian')
                self.assertEqual(model.peft_config['default'].target_modules, {'q_proj', 'v_proj'})
                trainable = [(name, value) for name, value in model.named_parameters() if value.requires_grad]
                self.assertTrue(trainable)
                for name, parameter in model.named_parameters():
                    self.assertEqual(parameter.requires_grad, 'lora_' in name)
                    if parameter.requires_grad:
                        self.assertEqual(parameter.dtype, torch.float32)
                    if 'lora_B' in name:
                        self.assertEqual(torch.count_nonzero(parameter).item(), 0)

    def test_explicit_kaiming_control_remains_available(self):
        model=self.module.build_lora(tiny_model(),42,initialization=True)
        self.assertIs(model.peft_config['default'].init_lora_weights,True)
        with self.assertRaises(ValueError):
            self.module.build_lora(tiny_model(),42,initialization=False)

    def test_trainer_enters_accelerator_autocast_for_direct_decoder(self):
        for kind in ['llama','qwen']:
            with self.subTest(kind=kind):
                model=self.module.build_lora(tiny_model(kind),42)
                trainer=self.trainer(model=model)
                batch=trainer.data_collator([{'source_index':0,'input_ids':[1,3,4,8,2]}])
                ids=batch['input_ids']
                model.eval()
                with torch.autocast('cpu',dtype=torch.bfloat16),torch.no_grad():
                    reference=input_cross_entropy(model(input_ids=ids,use_cache=False).logits,ids)
                seen=[]
                handle=model.get_base_model().model.register_forward_pre_hook(
                    lambda module,inputs:seen.append(torch.is_autocast_enabled('cpu')))
                with mock.patch.object(trainer.accelerator,'autocast',
                    side_effect=lambda:torch.autocast('cpu',dtype=torch.bfloat16)):
                    trainer.compute_loss(model,batch)
                handle.remove()
                self.assertEqual(seen,[True,True],'Both base scoring and current CE must enter AMP')
                self.assertAlmostEqual(trainer.history[0]['base_ce'],float(reference[0]),places=5)
                self.assertAlmostEqual(trainer.history[0]['current_ce'],float(reference[0]),places=5)

    def test_precision_reference_check_covers_gradients_and_leaves_no_state(self):
        from precision import compare_precision_paths
        for kind in ['llama','qwen']:
            with self.subTest(kind=kind):
                trainer=self.trainer(model=self.module.build_lora(tiny_model(kind),42))
                # Nonzero B also exercises A gradients after adaptation.
                with torch.no_grad():
                    for name,p in trainer.model.named_parameters():
                        if 'lora_B' in name:
                            p.normal_(mean=0.,std=.01)
                initial=adapter_state(trainer.model)
                with mock.patch.object(trainer.accelerator,'autocast',
                    side_effect=lambda:torch.autocast('cpu',dtype=torch.bfloat16)):
                    report=compare_precision_paths(trainer,{'source_index':0,'input_ids':[1,3,4,8,2]})
                self.assertEqual(report['status'],'PASS')
                self.assertGreater(report['compared_parameters'],0)
                self.assertEqual(trainer.history,[])
                self.assertEqual(trainer.base_scores,{})
                self.assertTrue(all(p.grad is None for p in trainer.model.parameters()))
                self.assertTrue(all(torch.equal(v,adapter_state(trainer.model)[k]) for k,v in initial.items()))

    def test_collator_keeps_real_eos_masks_right_padding_and_rejects_answers(self):
        collator = self.module.InputCollator(2)
        batch = collator([{'source_index': 0, 'input_ids': [1, 2]},
                          {'source_index': 1, 'input_ids': [1, 5, 2]}])
        self.assertEqual(batch['input_ids'].tolist(), [[1, 2, 2], [1, 5, 2]])
        self.assertEqual(batch['attention_mask'].tolist(), [[1, 1, 0], [1, 1, 1]])
        self.assertEqual(batch['labels'].tolist(), [[1, 2, -100], [1, 5, 2]])
        self.assertEqual(batch['source_index'].tolist(), [0, 1])
        for value in batch.values():
            self.assertEqual(value.dtype, torch.long)
        for bad in [[], [{'source_index': True, 'input_ids': [1, 2]}],
                    [{'source_index': 0, 'input_ids': [1]}],
                    [{'source_index': 0, 'input_ids': [1, False]}],
                    [{'source_index': 0, 'input_ids': [1, -1]}],
                    [{'source_index': 0, 'input_ids': [1, 2], 'output': 'secret'}]]:
            with self.subTest(features=bad), self.assertRaises(ValueError):
                collator(bad)

    def test_chunked_real_peft_loss_and_gradients_match_full_without_base_changes(self):
        ids = torch.tensor([[1, 3, 8, 2, 0]])
        mask = torch.tensor([[1, 1, 1, 1, 0]])
        for kind in ['llama', 'qwen']:
            with self.subTest(kind=kind):
                full = self.module.build_lora(tiny_model(kind), 42).eval()
                chunk = copy.deepcopy(full)
                frozen = {name: value.detach().clone() for name, value in full.named_parameters()
                          if not value.requires_grad}
                expected = input_cross_entropy(full(input_ids=ids, attention_mask=mask).logits, ids, mask)
                actual = model_input_cross_entropy(chunk, ids, mask, chunk_size=2)
                expected.sum().backward()
                actual.sum().backward()
                torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
                for (name, left), (_, right) in zip(full.named_parameters(), chunk.named_parameters()):
                    if left.requires_grad:
                        torch.testing.assert_close(left.grad, right.grad, atol=1e-6, rtol=1e-5)
                    else:
                        self.assertIsNone(left.grad)
                        self.assertIsNone(right.grad)
                        self.assertTrue(torch.equal(left, frozen[name]))

    def test_base_score_cache_rejects_changed_tokens_and_batch_two(self):
        trainer = self.trainer()
        collate = trainer.data_collator
        trainer.compute_loss(trainer.model, collate([{'source_index': 1, 'input_ids': [1, 3, 4]}]))
        with self.assertRaises(ValueError):
            trainer.compute_loss(trainer.model, collate([{'source_index': 1, 'input_ids': [1, 3, 5]}]))
        with self.assertRaises(ValueError):
            trainer.compute_loss(trainer.model, collate([{'source_index': 1, 'input_ids': [1, 3, 4]},
                                                         {'source_index': 2, 'input_ids': [1, 3]}]))

    def test_zero_loss_still_advances_adam_momentum(self):
        callback = RejectSecond()
        trainer = self.trainer(args=self.args(max_steps=2, warmup_ratio=0.), callbacks=[callback])
        callback.trainer = trainer
        trainer.train()
        self.assertTrue(trainer.history[0]['selected'])
        self.assertFalse(trainer.history[1]['selected'])
        self.assertEqual(trainer.history[1]['loss'], 0.)
        after = adapter_state(trainer.model)
        self.assertTrue(any(not torch.equal(value, after[name])
                            for name, value in callback.after_first.items()))
        steps = {int(value['step']) for value in trainer.optimizer.state.values()}
        self.assertEqual(steps, {2})

    def test_explicit_checkpoint_resume_matches_uninterrupted_bitwise(self):
        full = self.trainer('full')
        full.train()
        partial = self.trainer('partial', callbacks=[StopAtTwo()])
        partial.train()
        checkpoint = self.root / 'partial' / 'checkpoint-2'
        history = self.module.verify_checkpoint(checkpoint)
        self.assertEqual(len(history), 2)
        resumed = self.trainer('partial')
        resumed.resume_history(checkpoint)
        resumed.train(resume_from_checkpoint=str(checkpoint))
        self.assertEqual(full.history, resumed.history)
        self.assertEqual([row['visit'] for row in resumed.history], [1, 2, 3, 4])
        for name, expected in adapter_state(full.model).items():
            self.assertTrue(torch.equal(expected, adapter_state(resumed.model)[name]), name)
        self.assertEqual(full.lr_scheduler.state_dict(), resumed.lr_scheduler.state_dict())
        for left, right in zip(full.optimizer.state.values(), resumed.optimizer.state.values()):
            for key in left:
                if torch.is_tensor(left[key]):
                    self.assertTrue(torch.equal(left[key], right[key]), key)
                else:
                    self.assertEqual(left[key], right[key])
        rng_full = torch.load(self.root / 'full' / 'checkpoint-4' / 'rng_state.pth', weights_only=False)
        rng_resume = torch.load(self.root / 'partial' / 'checkpoint-4' / 'rng_state.pth', weights_only=False)
        self.assertEqual(rng_full['python'], rng_resume['python'])
        self.assertTrue(torch.equal(rng_full['cpu'], rng_resume['cpu']))

    def test_checkpoint_rejects_missing_commit_and_changed_files(self):
        trainer = self.trainer(args=self.args(max_steps=2))
        trainer.train()
        checkpoint = self.root / 'run' / 'checkpoint-2'
        self.module.verify_checkpoint(checkpoint)
        commit = checkpoint / 'commit.json'
        payload = commit.read_bytes()
        commit.unlink()
        with self.assertRaises(ValueError):
            self.module.verify_checkpoint(checkpoint)
        commit.write_bytes(payload)
        with (checkpoint / 'history.json').open('ab') as stream:
            stream.write(b' ')
        with self.assertRaises(ValueError):
            self.module.verify_checkpoint(checkpoint)

    def test_checkpoint_rejects_unhashed_nested_commit_file(self):
        trainer = self.trainer(args=self.args(max_steps=2))
        trainer.train()
        checkpoint = self.root / 'run' / 'checkpoint-2'
        self.module.verify_checkpoint(checkpoint)
        nested = checkpoint / 'unexpected'
        nested.mkdir()
        (nested / 'commit.json').write_text('{}')
        with self.assertRaises(ValueError):
            self.module.verify_checkpoint(checkpoint)

    def test_checkpoint_explicitly_saves_and_restores_gradient_scaler(self):
        trainer = self.trainer(args=self.args(max_steps=2))
        trainer.train()
        scaler = torch.amp.GradScaler('cpu')
        state = scaler.state_dict()
        state['scale'] = 256.
        state['_growth_tracker'] = 123
        scaler.load_state_dict(state)
        trainer.accelerator.scaler = scaler
        trainer._save_checkpoint(trainer.model, trial=None)
        checkpoint = self.root / 'run' / 'checkpoint-2'
        self.assertTrue((checkpoint / 'scaler.pt').is_file(),
                        'Pinned Trainer does not save the dynamic gradient scaler')
        self.module.verify_checkpoint(checkpoint)
        restored = self.trainer('restored')
        restored.accelerator.scaler = torch.amp.GradScaler('cpu')
        restored.create_optimizer_and_scheduler(num_training_steps=4)
        restored._load_optimizer_and_scheduler(str(checkpoint))
        self.assertEqual(restored.accelerator.scaler.state_dict(), state)

    def test_milestone_replay_accepts_identical_and_rejects_changed_artifacts(self):
        with mock.patch.object(self.module, 'MILESTONE_STEPS', frozenset({2})):
            trainer = self.trainer(args=self.args(max_steps=2))
            trainer.train()
            milestone = self.root / 'run' / 'milestones' / 'step-2'
            self.assertTrue((milestone / 'adapter_model.safetensors').is_file())
            try:
                trainer._save_checkpoint(trainer.model, trial=None)
            except FileExistsError as error:
                self.fail('Identical milestone replay must be idempotent: ' + str(error))
            with (milestone / 'adapter_config.json').open('ab') as stream:
                stream.write(b' ')
            with self.assertRaises(ValueError):
                trainer._save_checkpoint(trainer.model, trial=None)
            with self.assertRaises(ValueError):
                self.module.verify_checkpoint(self.root / 'run' / 'checkpoint-2')


if __name__ == '__main__':
    unittest.main()
