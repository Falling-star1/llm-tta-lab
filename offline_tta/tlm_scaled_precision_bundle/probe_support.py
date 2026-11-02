"""Dependencies and reversible diagnostic state, shared by the two probes."""
from contextlib import contextmanager
import copy
import importlib.util
import math
from pathlib import Path
import random

HERE = Path(__file__).resolve().parent
V3 = HERE.parent / 'tlm_offline_v3_bundle'
DIAG = HERE.parent / 'tlm_precision_diagnostic_bundle'
V3_ID = '267c3593e1d2c62f7d741168dcfd82ff76dc60f5458a06e481beca570997915d'
DIAG_SUMS = '7d606a2b95c62e4e2468ea59b1addfc1a7daf1a5cc470260b775dee6a6a1b2fd'
_reference = None
_missing = object()


def reference():
    global _reference
    if _reference is None:
        spec = importlib.util.spec_from_file_location('tlm_frozen_precision_reference', DIAG / 'diagnose.py')
        _reference = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_reference)
    return _reference


def snapshot(model):
    return {name: p.detach().to('cpu', copy=True) for name, p in model.named_parameters() if p.requires_grad}


def restore_rng(state):
    import numpy as np
    import torch
    torch.set_rng_state(state['torch_rng'])
    if state['cuda_rng'] is not None:
        torch.cuda.set_rng_state_all(state['cuda_rng'])
    random.setstate(state['python_rng'])
    np.random.set_state(state['numpy_rng'])


def reset_path(trainer, state, nonzero_b=False):
    import torch
    model = trainer.model
    model.zero_grad(set_to_none=True)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name in state['weights']:
                p.copy_(state['weights'][name].to(p.device))
                if nonzero_b and 'lora_B' in name:
                    p.fill_(.001)
    trainer.history.clear()
    trainer.base_scores.clear()
    restore_rng(state)


@contextmanager
def preserve_forward(model):
    attributes = {name: model.__dict__.get(name, _missing) for name in ('forward', '_original_forward')}
    try:
        yield
    finally:
        for name, value in attributes.items():
            if value is _missing:
                model.__dict__.pop(name, None)
            else:
                model.__dict__[name] = value


@contextmanager
def preserve_trainer(trainer):
    import numpy as np
    import torch
    model = trainer.model
    if trainer.history or trainer.base_scores or any(p.grad is not None for p in model.parameters()):
        raise ValueError('Use a fresh diagnostic trainer')
    weights = snapshot(model)
    if not weights or any('lora_' not in key for key in weights):
        raise ValueError('Expected only trainable LoRA parameters')
    if any(bool(value.count_nonzero()) for key, value in weights.items() if 'lora_B' in key):
        raise ValueError('Expected fresh zero-B initialization')
    state = {'weights': weights, 'torch_rng': torch.get_rng_state(),
             'cuda_rng': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
             'python_rng': random.getstate(), 'numpy_rng': np.random.get_state(),
             'scaler': trainer.accelerator.scaler,
             'scaler_state': copy.deepcopy(trainer.accelerator.scaler.state_dict())
                 if trainer.accelerator.scaler is not None else None,
             'modes': [(module, module.training) for module in model.modules()]}
    with preserve_forward(model):
        try:
            yield state
        finally:
            reset_path(trainer, state)
            trainer.accelerator.scaler = state['scaler']
            for module, training in state['modes']:
                module.training = training


def summarize_statistics(rows):
    finite = [value for value in rows.values() if value.get('finite')]
    ref = sum(value.get('reference_l2', 0.) ** 2 for value in finite)
    got = sum(value.get('candidate_l2', 0.) ** 2 for value in finite)
    diff = sum(value.get('difference_l2', 0.) ** 2 for value in finite)
    dot = sum(value['cosine'] * value['reference_l2'] * value['candidate_l2']
              for value in finite if value.get('cosine') is not None)
    return {'parameters': len(rows), 'nonfinite_or_missing': len(rows) - len(finite),
            'pointwise_failed_parameters': sum(not value.get('pointwise_close', False) for value in rows.values()),
            'global_relative_l2': math.sqrt(diff / ref) if ref else (0. if not diff else None),
            'global_cosine': max(-1., min(1., dot / math.sqrt(ref * got))) if ref and got else None,
            'max_absolute_difference': max((value.get('max_absolute_difference', 0.) for value in finite), default=None)}


@contextmanager
def projection_checkpoint(enabled):
    import tlm_loss
    original = tlm_loss.checkpoint
    if not enabled:
        def direct(function, *args, **kwargs):
            return function(*args)
        tlm_loss.checkpoint = direct
    try:
        yield
    finally:
        tlm_loss.checkpoint = original
