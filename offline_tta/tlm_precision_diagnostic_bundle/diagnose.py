"""Diagnose frozen v3 FP16 paths; never update weights or issue a preflight receipt."""
import argparse
from contextlib import contextmanager
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys

V3_CODE_ID = '267c3593e1d2c62f7d741168dcfd82ff76dc60f5458a06e481beca570997915d'
PATHS = ('full', 'direct_full_head', 'direct_single', 'direct_multi', 'trainer_multi')


def verify_delivery(directory):
    directory = Path(directory).resolve()
    rows = (directory / 'SHA256SUMS').read_text(encoding='utf-8').splitlines()
    if not rows:
        raise ValueError('Missing delivery hashes')
    for row in rows:
        expected, name = row.split(maxsplit=1)
        path = (directory / name.lstrip('*')).resolve()
        if not path.is_relative_to(directory) or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError('Delivery differs: ' + name)


def tensor_difference(candidate, reference, atol=1e-5, rtol=1e-2):
    """Full-tensor statistics, with bounded CPU work buffers and JSON-safe zeros."""
    import torch
    if candidate is None or reference is None:
        return {'finite': False, 'pointwise_close': False,
                'missing_candidate': candidate is None, 'missing_reference': reference is None}
    if candidate.shape != reference.shape:
        return {'finite': False, 'pointwise_close': False,
                'candidate_shape': list(candidate.shape), 'reference_shape': list(reference.shape)}
    got, expected = candidate.detach().cpu().reshape(-1), reference.detach().cpu().reshape(-1)
    result = {'elements': got.numel(), 'atol': atol, 'rtol': rtol,
              'candidate_dtype': str(candidate.dtype), 'reference_dtype': str(reference.dtype)}
    squared_got = squared_ref = squared_difference = dot = 0.
    outside = near_zero = sign_disagreements = active_sign_elements = 0
    maximum = 0.
    for start in range(0, got.numel(), 262144):
        a = got[start:start + 262144].double()
        b = expected[start:start + 262144].double()
        if not bool(torch.isfinite(a).all() and torch.isfinite(b).all()):
            return {**result, 'finite': False, 'pointwise_close': False}
        difference = a - b
        maximum = max(maximum, float(difference.abs().max()))
        squared_got += float(a.square().sum())
        squared_ref += float(b.square().sum())
        squared_difference += float(difference.square().sum())
        dot += float((a * b).sum())
        outside += int((difference.abs() > atol + rtol * b.abs()).sum())
        near_zero += int((b.abs() <= atol).sum())
        active = (a.abs() > atol) & (b.abs() > atol)
        active_sign_elements += int(active.sum())
        sign_disagreements += int(((a.sign() != b.sign()) & active).sum())
    norm_got, norm_ref, norm_difference = map(math.sqrt, (squared_got, squared_ref, squared_difference))
    relative = norm_difference / norm_ref if norm_ref else (0. if not norm_difference else None)
    cosine = max(-1., min(1., dot / (norm_got * norm_ref))) if norm_got and norm_ref else None
    return {**result, 'finite': True, 'pointwise_close': outside == 0,
            'outside_tolerance': outside, 'outside_fraction': outside / max(1, got.numel()),
            'max_absolute_difference': maximum, 'reference_l2': norm_ref, 'candidate_l2': norm_got,
            'difference_l2': norm_difference, 'relative_l2': relative, 'cosine': cosine,
            'reference_near_zero_fraction': near_zero / max(1, got.numel()),
            'active_sign_elements': active_sign_elements, 'active_sign_disagreements': sign_disagreements}


@contextmanager
def capture_forward(model, device):
    """Capture raw head outputs before wrappers; omit checkpoint backward replays."""
    base = model.get_base_model()
    groups, autocast_trace = [], []
    enabled = [True]

    def decoder_before(module, args):
        import torch
        autocast_trace.append(torch.is_autocast_enabled(device))
        if enabled[0]:
            groups.append({'head_outputs': [], 'projection_sizes': []})

    def decoder_after(module, args, output):
        if enabled[0]:
            group = groups[-1]
            states = output.last_hidden_state
            group['hidden'] = states.detach().to('cpu', copy=True)
            group['hidden_gradient'] = None
            if states.requires_grad:
                def retain_gradient(gradient):
                    group['hidden_gradient'] = gradient.detach().to('cpu', copy=True)
                hooks.append(states.register_hook(retain_gradient))

    def head_after(module, args, output):
        if enabled[0]:
            groups[-1]['head_outputs'].append(output.detach().to('cpu', copy=True))
            groups[-1]['projection_sizes'].append(output.shape[1])

    hooks = [base.model.register_forward_pre_hook(decoder_before),
             base.model.register_forward_hook(decoder_after),
             base.get_output_embeddings().register_forward_hook(head_after)]
    try:
        yield groups, autocast_trace, enabled
    finally:
        for hook in hooks:
            hook.remove()


def _path_forward(trainer, batch, name):
    import torch
    from tlm_loss import input_cross_entropy, model_input_cross_entropy, weighted_input_loss, sample_weights
    model = trainer.model
    ids, mask = batch['input_ids'], batch['attention_mask']

    def ce():
        if name == 'full':
            logits = model(input_ids=ids, attention_mask=mask, use_cache=False).logits
            return input_cross_entropy(logits, ids, mask)
        if name == 'direct_full_head':
            base = model.get_base_model()
            states = base.model(input_ids=ids, attention_mask=mask, use_cache=False,
                                return_dict=True).last_hidden_state
            return input_cross_entropy(base.get_output_embeddings()(states), ids, mask)
        size = ids.shape[1] if name == 'direct_single' else trainer.chunk_size
        return model_input_cross_entropy(model, ids, mask, size)

    if name == 'trainer_multi':
        model.train()
        loss = trainer.compute_loss(model, batch)
        history = trainer.history[-1]
        values = {key: history[key] for key in ('base_ce', 'current_ce', 'weight', 'selected')}
    else:
        model.eval()
        with torch.no_grad(), model.disable_adapter(), trainer.accelerator.autocast():
            base_ce = ce()
        model.train()
        with trainer.accelerator.autocast():
            current_ce = ce()
        loss = weighted_input_loss(current_ce, base_ce, trainer.threshold, trainer.lamb)
        selected, weights = sample_weights(base_ce, trainer.threshold, trainer.lamb)
        values = {'base_ce': float(base_ce[0]), 'current_ce': float(current_ce[0].detach()),
                  'weight': float(weights[0]), 'selected': bool(selected[0])}
    return loss, values


def _compare_run(candidate, reference):
    import torch
    gradients = {name: tensor_difference(candidate['gradients'][name], value)
                 for name, value in reference['gradients'].items()}
    representations = {}
    for index, role in enumerate(('base', 'current')):
        got, expected = candidate['captures'][index], reference['captures'][index]
        representations[role] = {
            'hidden': tensor_difference(got['hidden'], expected['hidden']),
            'head_logits': tensor_difference(got['logits'], expected['logits']),
        }
        if role == 'current':
            representations[role]['hidden_gradient'] = tensor_difference(
                got['hidden_gradient'], expected['hidden_gradient'])
    scalar_comparisons = {key: tensor_difference(torch.tensor(candidate['values'][key]),
                                                torch.tensor(reference['values'][key]), rtol=1e-4)
                          for key in ('base_ce', 'current_ce', 'weight', 'weighted_loss')}
    gradients_close = all(row['pointwise_close'] and row['finite']
                          and (row['reference_l2'] <= 1e-8 or row['relative_l2'] <= .01)
                          for row in gradients.values())
    return {'gradients': gradients, 'representations': representations,
            'scalars': scalar_comparisons,
            'strict_comparison_match': bool(gradients_close
                and scalar_comparisons['weighted_loss']['pointwise_close']
                and candidate['autocast'] and all(candidate['autocast']))}


def collect_precision_report(trainer, feature, on_progress=None):
    """Compare all paths even on numerical mismatch; always restore the fresh model."""
    import numpy as np
    import torch
    if trainer.history or trainer.base_scores or any(p.grad is not None for p in trainer.model.parameters()):
        raise ValueError('Diagnostic requires fresh trainer history, cache and gradients')
    model = trainer.model
    parameters = {name: p for name, p in model.named_parameters() if p.requires_grad}
    if not parameters or any('lora_' not in name for name in parameters):
        raise ValueError('Expected only LoRA trainable parameters')
    if any(bool(p.count_nonzero()) for name, p in parameters.items() if 'lora_B' in name):
        raise ValueError('Expected fresh zero-B adapter')
    initial = {name: p.detach().to('cpu', copy=True) for name, p in parameters.items()}
    modes = [(module, module.training) for module in model.modules()]
    rng = (torch.get_rng_state(), torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
           random.getstate(), np.random.get_state())
    batch = trainer._prepare_inputs(trainer.data_collator([feature]))
    positions = batch['input_ids'].shape[1] - 1
    if positions <= trainer.chunk_size:
        raise ValueError('Diagnostic input must span multiple projection chunks')
    report = {'diagnostic_schema_version': 1, 'comparison_status': 'INCOMPLETE',
              'not_a_preflight_receipt': True, 'formal_training_authorized_by_this_report': False,
              'source_index': feature['source_index'], 'tokens': positions + 1,
              'chunk_size': trainer.chunk_size, 'gradient_atol': 1e-5, 'gradient_rtol': 1e-2,
              'gradient_relative_l2_limit': .01, 'states': {}}
    try:
        for state_name in ('initial_zero_B', 'nonzero_B'):
            with torch.no_grad():
                for name, p in parameters.items():
                    p.copy_(initial[name].to(p.device))
                    if state_name == 'nonzero_B' and 'lora_B' in name:
                        p.fill_(1e-3)
            state = {'temporary_B': 0. if state_name == 'initial_zero_B' else .001, 'paths': {}}
            report['states'][state_name] = state
            reference = previous = None
            for name in PATHS:
                trainer.history.clear()
                trainer.base_scores.clear()
                model.zero_grad(set_to_none=True)
                torch.set_rng_state(rng[0])
                if rng[1] is not None:
                    torch.cuda.set_rng_state_all(rng[1])
                random.setstate(rng[2])
                np.random.set_state(rng[3])
                with capture_forward(model, batch['input_ids'].device.type) as (groups, trace, enabled):
                    loss, values = _path_forward(trainer, batch, name)
                    enabled[0] = False
                    values['weighted_loss'] = float(loss.detach())
                    loss.backward()
                del loss
                if len(groups) != 2 or any(not group['head_outputs'] for group in groups):
                    raise ValueError('Expected separate base/current decoder and head captures')
                for group in groups:
                    group['logits'] = torch.cat(group.pop('head_outputs'), dim=1)[:, :positions]
                    if group['logits'].shape[1] != positions:
                        raise ValueError('Missing projected prediction positions')
                current = {'values': values, 'captures': groups, 'autocast': trace,
                           'gradients': {key: None if p.grad is None else p.grad.detach().to('cpu', copy=True)
                                         for key, p in parameters.items()}}
                if reference is None:
                    reference = current
                comparison = _compare_run(current, reference)
                comparison.update(values)
                comparison.update({'all_decoder_calls_under_autocast': bool(trace and all(trace)),
                                   'decoder_autocast_trace': trace,
                                   'base_projection_sizes': groups[0]['projection_sizes'],
                                   'current_projection_sizes': groups[1]['projection_sizes'],
                                   'comparisons_with_previous_path': None if previous is None
                                       else {'path': PATHS[PATHS.index(name) - 1], **_compare_run(current, previous)}})
                state['paths'][name] = comparison
                previous = current
                if on_progress:
                    on_progress(report)
                model.zero_grad(set_to_none=True)
                gc.collect()
            state['reference_factor_norms'] = {}
            for factor in ('lora_A', 'lora_B'):
                tensors = [value for key, value in reference['gradients'].items() if factor in key]
                values = [float(value.double().norm()) if value is not None else float('nan')
                          for value in tensors]
                state['reference_factor_norms'][factor] = (
                    sum(values) if values and all(math.isfinite(value) for value in values) else None)
            del reference, previous, current
        matches = all(path['strict_comparison_match'] for state in report['states'].values()
                      for path in state['paths'].values())
        factors_nonzero = all(report['states']['nonzero_B']['reference_factor_norms'].values())
        report['comparison_status'] = 'MATCH' if matches and factors_nonzero else 'MISMATCH'
        if on_progress:
            on_progress(report)
        return report
    finally:
        model.zero_grad(set_to_none=True)
        with torch.no_grad():
            for name, p in parameters.items():
                p.copy_(initial[name].to(p.device))
        trainer.history.clear()
        trainer.base_scores.clear()
        for module, training in modes:
            module.training = training
        torch.set_rng_state(rng[0])
        if rng[1] is not None:
            torch.cuda.set_rng_state_all(rng[1])
        random.setstate(rng[2])
        np.random.set_state(rng[3])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, default=Path(__file__).resolve().parent.parent / 'tlm_offline_v3_bundle')
    parser.add_argument('--model', choices=['llama3b', 'qwen3b'], required=True)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error('Use a new output directory; existing results are never overwritten')
    verify_delivery(Path(__file__).resolve().parent)
    verify_delivery(args.bundle)
    sys.path.insert(0, str(args.bundle.resolve()))
    from runtime import code_identity, atomic_json, gpu_lock, LoadedModel, read_inputs, specifications
    if code_identity() != V3_CODE_ID:
        raise ValueError('Diagnostic requires the unchanged frozen v3 bundle')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    from run import precision_feature, probe_training_arguments, PROBE_LAMB
    from train_core import TLMTrainer, InputCollator
    import torch
    torch.set_num_threads(4)
    with gpu_lock():
        args.output.mkdir(parents=True, exist_ok=False)
        metadata = {'v3_code_id': code_identity(), 'model': args.model, 'dataset': 'dolly_5k',
                    'lora_init': 'gaussian', 'seed': 42, 'optimizer_steps': 0,
                    'diagnostic_delivery_sha256': hashlib.sha256(
                        (Path(__file__).resolve().parent / 'SHA256SUMS').read_bytes()).hexdigest(),
                    'visible_gpu': os.environ.get('CUDA_VISIBLE_DEVICES'),
                    'torch': torch.__version__, 'cuda': torch.version.cuda,
                    'gpu_name': torch.cuda.get_device_name(0)}
        def save(report):
            atomic_json(args.output / 'report.json', {**metadata, **report})
            print(json.dumps({'comparison_status': report['comparison_status'],
                              'completed_paths': {key: len(value['paths']) for key, value in report['states'].items()}},
                             ensure_ascii=False), flush=True)
        try:
            _, spec = specifications(args.model, 'dolly_5k')
            rows, prepared = read_inputs(args.inputs, spec['sha256'])
            metadata['input_sha256'] = prepared['files']['inputs.jsonl']
            loaded = LoadedModel(args.model, 'dolly_5k', seed=42, initialization='gaussian')
            metadata['contract'] = loaded.contract
            metadata['precision_settings'] = {
                'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
                'matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
                'matmul_allow_fp16_reduced_precision_reduction':
                    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
                'float32_matmul_precision': torch.get_float32_matmul_precision(),
                'cublas_workspace_config': os.environ.get('CUBLAS_WORKSPACE_CONFIG')}
            feature = precision_feature(loaded.tokens(rows), 128)
            metadata['input_token_sha256'] = hashlib.sha256(json.dumps(feature['input_ids']).encode()).hexdigest()
            trainer = TLMTrainer(model=loaded.model,
                                 args=probe_training_arguments(args.output / 'unused_trainer', 42),
                                 train_dataset=[feature], data_collator=InputCollator(loaded.pad_id),
                                 threshold=0., lamb=PROBE_LAMB, chunk_size=128)
            result = collect_precision_report(trainer, feature, on_progress=save)
            save(result)
            print('REPORT=' + str(args.output / 'report.json'), flush=True)
        except BaseException as error:
            atomic_json(args.output / 'failure.json', {**metadata, 'type': type(error).__name__,
                                                       'message': str(error), 'not_a_preflight_receipt': True})
            raise
    return 0 if result['comparison_status'] == 'MATCH' else 2


if __name__ == '__main__':
    raise SystemExit(main())
