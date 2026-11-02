"""Actual GradScaler backward/unscale comparison; no optimizer steps."""
import copy
import gc

from probe_support import preserve_trainer, projection_checkpoint, reference, reset_path, snapshot, summarize_statistics


def scaled_backward(accelerator, optimizer, loss):
    import torch
    scaler = accelerator.scaler
    if scaler is None or not scaler.is_enabled():
        raise ValueError('An enabled real GradScaler is required')
    scale = float(scaler.get_scale())
    accelerator.backward(loss)
    # Same GradScaler operation used by Accelerate.unscale_gradients on CUDA FP16.
    # Explicit call also lets the CPU test exercise real scaling and unscaling.
    scaler.unscale_(optimizer)
    parameters = [p for group in optimizer.param_groups for p in group['params']]
    finite = all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in parameters)
    scaler.update()
    return {'scale_used': scale, 'next_scale': float(scaler.get_scale()), 'finite': finite}


def _run_path(trainer, feature, state, nonzero_b, name, scale, template):
    import torch
    reset_path(trainer, state, nonzero_b)
    fresh = torch.amp.GradScaler(feature['input_ids'].device.type)
    configuration = copy.deepcopy(template)
    configuration['scale'] = scale
    fresh.load_state_dict(configuration)
    trainer.accelerator.scaler = fresh
    optimizer = torch.optim.AdamW([p for p in trainer.model.parameters() if p.requires_grad], lr=1e-6)
    ref = reference()
    path = 'full' if name == 'full' else 'trainer_multi'
    with projection_checkpoint(name != 'trainer_multi_no_checkpoint'):
        with ref.capture_forward(trainer.model, feature['input_ids'].device.type) as (groups, trace, enabled):
            loss, values = ref._path_forward(trainer, feature, path)
            enabled[0] = False
            values['weighted_loss'] = float(loss.detach())
            backward = scaled_backward(trainer.accelerator, optimizer, loss)
        del loss
    positions = feature['input_ids'].shape[1] - 1
    if len(groups) != 2:
        raise ValueError('Expected base and current forward captures')
    for group in groups:
        group['logits'] = torch.cat(group.pop('head_outputs'), dim=1)[:, :positions]
        if group['hidden_gradient'] is not None:
            # Captured activation gradients are scaled FP16; compare them after FP32 unscale.
            group['hidden_gradient'] = group['hidden_gradient'].float() / scale
    gradients = {name: None if p.grad is None else p.grad.detach().to('cpu', copy=True)
                 for name, p in trainer.model.named_parameters() if p.requires_grad}
    del optimizer
    return {'values': values, 'captures': groups, 'autocast': trace, 'gradients': gradients,
            'backward': backward}


def compare_scaled_gradients(trainer, feature, on_progress=None):
    import torch
    if trainer.accelerator.scaler is None or not trainer.accelerator.scaler.is_enabled():
        raise ValueError('CUDA FP16 real GradScaler is required; CPU tests supply a real CPU GradScaler')
    batch = trainer._prepare_inputs(trainer.data_collator([feature]))
    if batch['input_ids'].shape[1] - 1 <= trainer.chunk_size:
        raise ValueError('Input must span multiple chunks')
    ref = reference()
    report = {'not_a_preflight_receipt': True, 'comparison_status': 'INCOMPLETE',
              'formal_training_authorized_by_this_report': False, 'optimizer_steps': 0,
              'states': {}, 'gradient_atol': 1e-5, 'gradient_rtol': 1e-2,
              'gradient_relative_l2_limit': .01,
              'scale_policy': 'Start at actual scaler value; back off jointly only for nonfinite gradients; stop at first jointly finite scale'}
    with preserve_trainer(trainer) as original:
        template = original['scaler_state']
        for state_name in ('initial_zero_B', 'nonzero_B'):
            state = {'attempts': [], 'paths': {}, 'finite_scale': None}
            report['states'][state_name] = state
            scale = float(template['scale'])
            for _ in range(24):
                full = _run_path(trainer, batch, original, state_name == 'nonzero_B', 'full', scale, template)
                multi = _run_path(trainer, batch, original, state_name == 'nonzero_B', 'trainer_multi', scale, template)
                attempt = {'scale': scale, 'full': full['backward'], 'trainer_multi': multi['backward']}
                state['attempts'].append(attempt)
                if full['backward']['finite'] and multi['backward']['finite']:
                    state['finite_scale'] = scale
                    break
                if on_progress:
                    on_progress(report)
                del full, multi
                scale *= float(template['backoff_factor'])
                gc.collect()
            else:
                state['status'] = 'NONFINITE_AFTER_24_ATTEMPTS'
                continue
            no_checkpoint = _run_path(trainer, batch, original, state_name == 'nonzero_B',
                                      'trainer_multi_no_checkpoint', scale, template)
            for name, candidate in [('full', full), ('trainer_multi', multi),
                                    ('trainer_multi_no_checkpoint', no_checkpoint)]:
                comparison = ref._compare_run(candidate, full)
                comparison.update(candidate['values'])
                comparison.update(candidate['backward'])
                comparison['all_decoder_calls_under_autocast'] = bool(candidate['autocast'] and all(candidate['autocast']))
                comparison['gradient_summary'] = summarize_statistics(comparison['gradients'])
                state['paths'][name] = comparison
            state['status'] = 'MATCH' if state['paths']['trainer_multi']['strict_comparison_match'] else 'MISMATCH'
            state['reference_factor_norms'] = {}
            for factor in ('lora_A', 'lora_B'):
                values = [value.double().norm() for name, value in full['gradients'].items()
                          if factor in name and value is not None]
                state['reference_factor_norms'][factor] = float(torch.stack(values).norm()) if values else 0.
            required = ('lora_A', 'lora_B') if state_name == 'nonzero_B' else ('lora_B',)
            if not all(state['reference_factor_norms'][factor] > 0. for factor in required):
                state['status'] = 'MISMATCH'
            state['checkpoint_control_status'] = (
                'MATCH' if state['paths']['trainer_multi_no_checkpoint']['strict_comparison_match'] else 'MISMATCH')
            del full, multi, no_checkpoint
            if on_progress:
                on_progress(report)
        report['comparison_status'] = 'MATCH' if all(state.get('status') == 'MATCH'
                                                     for state in report['states'].values()) else 'MISMATCH'
        if on_progress:
            on_progress(report)
    return report
