"""Pinned Trainer integration for batch-one offline input-only TLM."""
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import Trainer, set_seed

from tlm_loss import model_input_cross_entropy, sample_weights, weighted_input_loss


MILESTONE_STEPS = frozenset({100, 500, 1000, 2500, 5000})
HISTORY_KEYS = frozenset({'visit', 'source_index', 'base_ce', 'selected', 'weight',
                          'current_ce', 'loss', 'lr'})


def build_lora(base, seed, rank=8, alpha=16, initialization=True):
    """Use PEFT's official default initialization; B starts at zero."""
    if type(seed) is not int or seed < 0 or type(rank) is not int or rank < 1:
        raise ValueError('seed and rank must be nonnegative/positive integers')
    if not isinstance(alpha, (int, float)) or isinstance(alpha, bool) or not math.isfinite(alpha) or alpha <= 0:
        raise ValueError('alpha must be positive and finite')
    if initialization is not True:
        raise ValueError('Official reproduction requires PEFT default initialization=True')
    if hasattr(base, 'peft_config'):
        raise ValueError('Expected a fresh base model without adapters')
    set_seed(seed)
    base.requires_grad_(False)
    config = LoraConfig(task_type=TaskType.CAUSAL_LM, r=rank, lora_alpha=alpha,
                        target_modules=['q_proj', 'v_proj'], lora_dropout=0.,
                        bias='none', init_lora_weights=True, inference_mode=False)
    model = get_peft_model(base, config)
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            if 'lora_' not in name:
                raise ValueError('Unexpected trainable base parameter: ' + name)
            parameter.data = parameter.data.to(torch.float32)
    model.train()
    return model


class InputCollator:
    """Right-pad inputs; real EOS tokens remain valid even when pad == EOS."""
    def __init__(self, pad_id):
        if type(pad_id) is not int or pad_id < 0:
            raise ValueError('pad_id must be a nonnegative integer')
        self.pad_id = pad_id

    def __call__(self, features):
        if not isinstance(features, (list, tuple)) or not features:
            raise ValueError('Expected nonempty input features')
        for row in features:
            if not isinstance(row, dict) or set(row) != {'input_ids', 'source_index'}:
                raise ValueError('Features require exactly input_ids and source_index')
            if type(row['source_index']) is not int or row['source_index'] < 0:
                raise ValueError('source_index must be a nonnegative integer')
            ids = row['input_ids']
            if (not isinstance(ids, (list, tuple)) or len(ids) < 2
                    or any(type(token) is not int or token < 0 for token in ids)):
                raise ValueError('input_ids must contain at least two nonnegative integer tokens')
        length = max(len(row['input_ids']) for row in features)
        ids = torch.full((len(features), length), self.pad_id, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for index, row in enumerate(features):
            count = len(row['input_ids'])
            ids[index, :count] = torch.tensor(row['input_ids'], dtype=torch.long)
            mask[index, :count] = 1
        labels = ids.clone().masked_fill(mask == 0, -100)
        return {'input_ids': ids, 'attention_mask': mask, 'labels': labels,
                'source_index': torch.tensor([row['source_index'] for row in features], dtype=torch.long)}


def _atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _file_hashes(path, exclude_commit=True):
    files = {}
    for file in sorted(path.rglob('*')):
        if file.is_symlink():
            raise ValueError('Checkpoint symlinks are not permitted')
        if file.is_file() and (not exclude_commit or file != path / 'commit.json'):
            digest = hashlib.sha256()
            with file.open('rb') as stream:
                for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
                    digest.update(chunk)
            files[file.relative_to(path).as_posix()] = digest.hexdigest()
    return files


def verify_checkpoint(path):
    """Verify a committed complete Trainer checkpoint before any torch loading."""
    path = Path(path)
    try:
        commit = json.loads((path / 'commit.json').read_text(encoding='utf-8'))
        if (not isinstance(commit, dict) or commit.get('schema_version') != 1
                or not isinstance(commit.get('files'), dict)):
            raise ValueError('Invalid checkpoint commit')
        files = _file_hashes(path)
        if files != commit['files']:
            raise ValueError('Checkpoint file hashes do not match commit')
        required = {'history.json', 'trainer_state.json', 'training_args.bin',
                    'optimizer.pt', 'scheduler.pt', 'rng_state.pth', 'adapter_config.json'}
        if type(commit.get('has_scaler')) is not bool:
            raise ValueError('Checkpoint must declare gradient scaler presence')
        if commit['has_scaler']:
            required.add('scaler.pt')
        elif 'scaler.pt' in files:
            raise ValueError('Unexpected gradient scaler state')
        if not required.issubset(files) or not ({'adapter_model.safetensors', 'adapter_model.bin'} & set(files)):
            raise ValueError('Checkpoint lacks required full-resume state')
        history = json.loads((path / 'history.json').read_text(encoding='utf-8'))
        state = json.loads((path / 'trainer_state.json').read_text(encoding='utf-8'))
        if not isinstance(history, list) or len(history) != state['global_step']:
            raise ValueError('History does not match checkpoint step')
        if commit.get('global_step') != state['global_step']:
            raise ValueError('Commit and Trainer step differ')
        for visit, row in enumerate(history, start=1):
            if (not isinstance(row, dict) or set(row) != HISTORY_KEYS
                    or type(row['visit']) is not int or row['visit'] != visit
                    or type(row['source_index']) is not int or row['source_index'] < 0
                    or type(row['selected']) is not bool):
                raise ValueError('Invalid checkpoint history row')
            for key in ['base_ce', 'weight', 'current_ce', 'loss', 'lr']:
                if type(row[key]) not in (int, float) or not math.isfinite(row[key]) or row[key] < 0:
                    raise ValueError('Invalid checkpoint history number: ' + key)
        return history
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise ValueError('Incomplete or invalid checkpoint: ' + str(path)) from error


class TLMTrainer(Trainer):
    def __init__(self, *args, threshold=3., lamb=.1, chunk_size=128, **kwargs):
        sample_weights(torch.tensor([0.]), threshold, lamb)
        if type(chunk_size) is not int or chunk_size < 1:
            raise ValueError('chunk_size must be a positive integer')
        self.threshold, self.lamb, self.chunk_size = threshold, lamb, chunk_size
        self.history = []
        self.base_scores = {}
        self._history_checkpoint = None
        super().__init__(*args, **kwargs)
        if (self.args.per_device_train_batch_size != 1 or self.args.gradient_accumulation_steps != 1
                or self.args.world_size != 1):
            raise ValueError('TLM requires batch-one single-process training without accumulation')
        if self.args.save_only_model:
            raise ValueError('Full optimizer/scheduler/RNG checkpoint state is required')
        self.args.remove_unused_columns = False
        self.args.save_total_limit = 2

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if set(inputs) != {'input_ids', 'attention_mask', 'labels', 'source_index'}:
            raise ValueError('Training batch contains unexpected keys')
        ids, mask, labels = inputs['input_ids'], inputs['attention_mask'], inputs['labels']
        indexes = inputs['source_index']
        if (ids.ndim != 2 or ids.shape[0] != 1 or indexes.dtype != torch.long
                or tuple(indexes.shape) != (1,) or int(indexes[0]) < 0):
            raise ValueError('Expected batch-one input IDs and source_index')
        if not torch.equal(labels, ids.clone().masked_fill(mask == 0, -100)):
            raise ValueError('Labels must equal input IDs with padding masked')
        index = int(indexes[0])
        tokens = tuple(ids[0].detach().cpu().tolist())
        cached = self.base_scores.get(index)
        if cached is not None and cached[0] != tokens:
            raise ValueError('Input tokens changed for source_index %d' % index)
        if cached is None:
            was_training = model.training
            try:
                model.eval()
                with torch.no_grad(), model.disable_adapter():
                    base_ce = model_input_cross_entropy(model, ids, mask, self.chunk_size)
            finally:
                model.train(was_training)
            self.base_scores[index] = (tokens, float(base_ce[0]))
        else:
            base_ce = torch.tensor([cached[1]], device=ids.device, dtype=torch.float32)
        current = model_input_cross_entropy(model, ids, mask, self.chunk_size)
        loss = weighted_input_loss(current, base_ce, self.threshold, self.lamb)
        selected, weights = sample_weights(base_ce, self.threshold, self.lamb)
        lr = float(self.optimizer.param_groups[0]['lr']) if self.optimizer is not None else float(self.args.learning_rate)
        self.history.append({'visit': len(self.history) + 1, 'source_index': index,
                             'base_ce': float(base_ce[0]), 'selected': bool(selected[0]),
                             'weight': float(weights[0]), 'current_ce': float(current[0].detach()),
                             'loss': float(loss.detach()), 'lr': lr})
        return (loss, {'loss': loss}) if return_outputs else loss

    def _save_checkpoint(self, model, trial, metrics=None):
        path = Path(self._get_output_dir(trial=trial)) / ('checkpoint-%d' % self.state.global_step)
        # A replay must not leave the previous commit valid while its files are
        # being rewritten. A failure before the final commit is fail-closed.
        if self.args.should_save and (path / 'commit.json').exists():
            (path / 'commit.json').unlink()
        super()._save_checkpoint(model, trial, metrics)
        if not self.args.should_save:
            return
        scaler = getattr(self.accelerator, 'scaler', None)
        if scaler is not None:
            torch.save(scaler.state_dict(), path / 'scaler.pt')
        _atomic_json(path / 'history.json', self.history)
        if self.state.global_step in MILESTONE_STEPS:
            milestone = Path(self.args.output_dir) / 'milestones' / ('step-%d' % self.state.global_step)
            milestone.parent.mkdir(parents=True, exist_ok=True)
            staging = Path(tempfile.mkdtemp(prefix='.' + milestone.name + '-', dir=milestone.parent))
            try:
                # Copy only the adapter export already saved by Trainer, so
                # milestone replay is byte-identical and cannot consume RNG.
                for name in ['adapter_config.json', 'adapter_model.safetensors', 'adapter_model.bin']:
                    if (path / name).is_file():
                        shutil.copyfile(path / name, staging / name)
                if milestone.exists():
                    if _file_hashes(milestone, False) != _file_hashes(staging, False):
                        raise ValueError('Existing milestone differs: ' + str(milestone))
                else:
                    os.rename(staging, milestone)
                    staging = None
            finally:
                if staging is not None:
                    shutil.rmtree(staging)
        _atomic_json(path / 'commit.json', {'schema_version': 1,
                                          'global_step': self.state.global_step,
                                          'has_scaler': scaler is not None,
                                          'files': _file_hashes(path)})

    def _load_optimizer_and_scheduler(self, checkpoint):
        if checkpoint is not None:
            verify_checkpoint(checkpoint)
        super()._load_optimizer_and_scheduler(checkpoint)
        if checkpoint is None:
            return
        scaler = getattr(self.accelerator, 'scaler', None)
        path = Path(checkpoint) / 'scaler.pt'
        if (scaler is not None) != path.is_file():
            raise ValueError('Gradient scaler mode differs from the checkpoint')
        if scaler is not None:
            # Explicit weights_only avoids the changed torch.load default on new
            # local runtimes; GradScaler state contains primitive values only.
            scaler.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))

    def resume_history(self, path):
        self.history = verify_checkpoint(path)
        self.base_scores = {}
        self._history_checkpoint = Path(path).resolve()
        return self.history

    def train(self, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None, **kwargs):
        if isinstance(resume_from_checkpoint, bool):
            if resume_from_checkpoint:
                raise ValueError('Resume requires an explicit checkpoint path')
            resume_from_checkpoint = None
        if resume_from_checkpoint is not None:
            verified = verify_checkpoint(resume_from_checkpoint)
            if (self._history_checkpoint != Path(resume_from_checkpoint).resolve()
                    or self.history != verified):
                raise ValueError('Call resume_history with the explicit checkpoint before training')
        return super().train(resume_from_checkpoint=resume_from_checkpoint, trial=trial,
                             ignore_keys_for_eval=ignore_keys_for_eval, **kwargs)
