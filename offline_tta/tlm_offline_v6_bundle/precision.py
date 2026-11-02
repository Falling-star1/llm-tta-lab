"""Compare full/chunk paths with real GradScaler and unchanged tolerances."""
import copy
from pathlib import Path
import random

import numpy as np
import torch

from tlm_loss import input_cross_entropy, weighted_input_loss

MAX_BACKOFFS = 24


def _gradient_statistics(gradients):
    result = {}
    for name, gradient in gradients.items():
        if gradient is None:
            result[name] = {'missing': True, 'finite': False, 'l2': None}
            continue
        finite = torch.isfinite(gradient)
        bad = int((~finite).sum())
        result[name] = {'missing': False, 'finite': bad == 0,
                        'elements': gradient.numel(), 'nonfinite_elements': bad,
                        'l2': float(gradient.double().norm()) if bad == 0 else None}
    return result


def compare_precision_paths(trainer, feature, b_value=1e-3):
    if trainer.history or trainer.base_scores:
        raise ValueError('Precision check requires a fresh diagnostic trainer')
    model = trainer.model
    parameters = {name: p for name, p in model.named_parameters() if p.requires_grad}
    if any(p.grad is not None for p in model.parameters()):
        raise ValueError('Precision check requires empty gradients')
    batch = trainer._prepare_inputs(trainer.data_collator([feature]))
    ids, mask = batch['input_ids'], batch['attention_mask']
    device = ids.device.type
    original_scaler = trainer.accelerator.scaler
    if device == 'cuda' and (original_scaler is None or not original_scaler.is_enabled()):
        raise ValueError('CUDA precision check requires the actual FP16 GradScaler')
    scaler_template = (copy.deepcopy(original_scaler.state_dict())
                       if original_scaler is not None and original_scaler.is_enabled()
                       else torch.amp.GradScaler('cpu').state_dict())
    initial = {name: p.detach().cpu().clone() for name, p in parameters.items()}
    modes = [(module, module.training) for module in model.modules()]
    rng = (random.getstate(), np.random.get_state(), torch.get_rng_state(),
           torch.cuda.get_rng_state_all() if device == 'cuda' else None)
    trace = []
    hook = model.get_base_model().model.register_forward_pre_hook(
        lambda module, inputs: trace.append(torch.is_autocast_enabled(device)))
    head_calls=[]
    head_gradients={}
    active_path=[None]
    def capture_head(module,inputs,output):
        head_calls.append({'positions':inputs[0].shape[1],'output_dtype':str(output.dtype)})
        if output.requires_grad:
            path=active_path[0]
            output.register_hook(lambda gradient:head_gradients.__setitem__(path,gradient.detach().cpu().clone()))
    head_hook=model.get_base_model().get_output_embeddings().register_forward_hook(capture_head)
    report_path = Path(trainer.args.output_dir) / 'precision_path_check.json'

    def save(value):
        from train_core import _atomic_json
        _atomic_json(report_path, value)

    def restore_rng():
        random.setstate(rng[0]); np.random.set_state(rng[1]); torch.set_rng_state(rng[2])
        if rng[3] is not None:
            torch.cuda.set_rng_state_all(rng[3])

    def run_path(full, scale):
        active_path[0]='reference' if full else 'candidate'
        model.zero_grad(set_to_none=True)
        trainer.history.clear(); trainer.base_scores.clear()
        restore_rng()
        scaler = torch.amp.GradScaler(device)
        state = copy.deepcopy(scaler_template)
        state['scale'] = scale
        scaler.load_state_dict(state)
        trainer.accelerator.scaler = scaler
        optimizer = torch.optim.AdamW(list(parameters.values()), lr=1e-6, weight_decay=0.)
        if full:
            model.eval()
            with torch.no_grad(), model.disable_adapter(), trainer.accelerator.autocast():
                base = input_cross_entropy(model(input_ids=ids, attention_mask=mask, use_cache=False).logits, ids, mask)
            model.train()
            with trainer.accelerator.autocast():
                current = input_cross_entropy(model(input_ids=ids, attention_mask=mask, use_cache=False).logits, ids, mask)
            loss = weighted_input_loss(current, base, trainer.threshold, trainer.lamb)
        else:
            model.train()
            loss = trainer.compute_loss(model, batch)
        value = float(loss.detach())
        trainer.accelerator.backward(loss)
        missing = [name for name, p in parameters.items() if p.grad is None]
        if missing:
            gradients = {name: None if p.grad is None else p.grad.detach().cpu().clone()
                         for name, p in parameters.items()}
            save({'status': 'FAIL', 'backward_mode': 'gradscaler', 'error_kind': 'missing_gradient',
                  'path': 'reference' if full else 'chunk', 'scale_used': scale,
                  'gradients_are_scaled': True, 'missing_parameters': missing,
                  'parameters': _gradient_statistics(gradients)})
            raise ValueError('Missing precision-check gradients: ' + ', '.join(missing))
        scaler.unscale_(optimizer)
        gradients = {name: None if p.grad is None else p.grad.detach().cpu().clone()
                     for name, p in parameters.items()}
        finite = all(g is not None and bool(torch.isfinite(g).all()) for g in gradients.values())
        scaler.update()
        del loss, optimizer
        return value, gradients, finite

    try:
        with torch.no_grad():
            for name, p in parameters.items():
                if 'lora_B' in name:
                    p.fill_(b_value)
        attempts = []
        scale = float(scaler_template['scale'])
        for _ in range(MAX_BACKOFFS):
            reference_value, expected, reference_finite = run_path(True, scale)
            actual_value, got, actual_finite = run_path(False, scale)
            attempts.append({'scale': scale, 'reference_finite': reference_finite, 'chunk_finite': actual_finite})
            if reference_finite and actual_finite:
                break
            scale *= float(scaler_template['backoff_factor'])
        else:
            save({'status': 'FAIL', 'backward_mode': 'gradscaler', 'attempts': attempts,
                  'error': 'No jointly finite gradients within %d attempts' % MAX_BACKOFFS,
                  'parameters': {'reference': _gradient_statistics(expected),
                                 'chunk': _gradient_statistics(got)}})
            raise FloatingPointError('Precision comparison has no jointly finite scaled backward')
        rows, failures = {}, []
        factor_norms = {'lora_A': 0., 'lora_B': 0.}
        max_abs = max_relative = 0.
        for name in parameters:
            a, b = got[name], expected[name]
            delta = a.double() - b.double()
            absolute = float(delta.abs().max())
            norm = float(b.double().norm())
            relative = float(delta.norm()) / norm if norm else (0. if not bool(delta.count_nonzero()) else None)
            outside = int((~torch.isclose(a, b, atol=1e-5, rtol=1e-2)).sum())
            relative_ok = norm <= 1e-8 or (relative is not None and relative <= .01)
            rows[name] = {'max_absolute_difference': absolute, 'reference_l2': norm,
                          'relative_l2': relative, 'outside_tolerance': outside,
                          'elements': a.numel(), 'relative_l2_within_limit': relative_ok}
            if outside or not relative_ok:
                failures.append(name)
            max_abs = max(max_abs, absolute)
            if norm > 1e-8:
                max_relative = max(max_relative, relative)
            for factor in factor_norms:
                if factor in name:
                    factor_norms[factor] += norm
        loss_matches = bool(torch.isclose(torch.tensor(actual_value), torch.tensor(reference_value), atol=1e-5, rtol=1e-4))
        autocast_matches = bool(trace and all(trace))
        factors_nonzero = factor_norms['lora_B']>0 and (b_value==0. or factor_norms['lora_A']>0)
        whole_head=bool(head_calls and all(row['positions']==ids.shape[1] for row in head_calls))
        dtype_matches=device!='cuda' or all(row['output_dtype']=='torch.float16' for row in head_calls)
        a,b=head_gradients['candidate'],head_gradients['reference']
        dlogits={'exact_match':bool(torch.equal(a,b)),
                 'max_absolute_difference':float((a-b).abs().max()),
                 'outside_tolerance':int((~torch.isclose(a,b,atol=1e-5,rtol=1e-2)).sum())}
        passed = not failures and loss_matches and autocast_matches and factors_nonzero and whole_head and dtype_matches and dlogits['outside_tolerance']==0
        report = {'status': 'PASS' if passed else 'FAIL', 'device': device, 'tokens': ids.shape[1],
                  'backward_mode': 'gradscaler', 'scale_used': scale, 'attempts': attempts,
                  'scaler_source': 'actual_trainer' if original_scaler is not None else 'cpu_mechanism_test',
                  'projection_chunks': 1,'ce_chunks':(ids.shape[1]-2)//trainer.chunk_size+1,
                  'projection_policy': 'whole-head-gemm; cpu-logits; streamed-fp32-ce',
                  'head_calls':head_calls,'full_head_shape_preserved':whole_head,
                  'raw_head_dtype_matches':dtype_matches,'logits_gradient':dlogits,
                  'probe_adapter_state': 'temporary B=%g; original A/B restored' % b_value,
                  'probe_b_value':b_value,
                  'nonzero_gradient_factors': [key for key,value in factor_norms.items() if value > 0],
                  'reference_loss': reference_value, 'chunk_loss': actual_value,
                  'compared_parameters': len(parameters), 'failed_parameters': failures,
                  'max_gradient_absolute_difference': max_abs,
                  'max_gradient_relative_l2_difference': max_relative,
                  'all_decoder_calls_under_autocast': autocast_matches,
                  'gradient_atol': 1e-5, 'gradient_rtol': 1e-2, 'gradient_relative_l2_limit': .01,
                  'parameters': rows, 'report_path': str(report_path),
                  'note': 'No optimizer steps; actual Trainer disk continuation is checked next'}
        save(report)
        if not passed:
            raise AssertionError('Scaled full/chunk comparison failed; parameters=%d; report=%s' % (len(failures), report_path))
        return report
    finally:
        hook.remove()
        head_hook.remove()
        model.zero_grad(set_to_none=True)
        with torch.no_grad():
            for name, p in parameters.items():
                p.copy_(initial[name].to(p.device))
        trainer.accelerator.scaler = original_scaler
        trainer.history.clear(); trainer.base_scores.clear()
        for module, training in modes:
            module.training = training
        restore_rng()
