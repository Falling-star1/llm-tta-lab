"""Two actual Trainer trajectories from a common zero-B initialization."""
from contextlib import contextmanager
import gc

from probe_support import preserve_forward, preserve_trainer, reference, reset_path, snapshot, summarize_statistics


@contextmanager
def full_forward_loss(enabled):
    import train_core
    from tlm_loss import input_cross_entropy
    original = train_core.model_input_cross_entropy
    if enabled:
        def full(model, input_ids, attention_mask=None, chunk_size=128):
            return input_cross_entropy(model(input_ids=input_ids, attention_mask=attention_mask,
                                             use_cache=False).logits, input_ids, attention_mask)
        train_core.model_input_cross_entropy = full
    try:
        yield
    finally:
        train_core.model_input_cross_entropy = original


def compare_trajectories(carrier, create_trainer, on_progress=None):
    import torch
    from transformers import TrainerCallback
    ref = reference()
    report = {'not_a_preflight_receipt': True, 'formal_training_authorized_by_this_report': False,
              'steps': [], 'full_source_order': [], 'chunk_source_order': [],
              'purpose': 'Bounded engineering updates, not formal one-epoch training',
              'paths': {}, 'status': 'INCOMPLETE'}
    expected = []
    with preserve_trainer(carrier) as original:
        for name in ('full', 'chunk'):
            reset_path(carrier, original)

            class Observe(TrainerCallback):
                trainer = None

                def on_step_end(self, args, state, control, **kwargs):
                    trainer = self.trainer
                    weights = snapshot(trainer.model)
                    record = {'step': state.global_step,
                              'parameters_finite': all(bool(torch.isfinite(value).all()) for value in weights.values()),
                              'optimizer_step_skipped': bool(trainer.accelerator.optimizer_step_was_skipped),
                              'scale': float(trainer.accelerator.scaler.get_scale())
                                  if trainer.accelerator.scaler is not None else None,
                              'learning_rate_after_step': float(trainer.optimizer.param_groups[0]['lr'])}
                    report.setdefault('progress', {})[name] = state.global_step
                    if name == 'full':
                        expected.append({'weights': weights, 'step': record})
                    else:
                        target = expected[state.global_step - 1]
                        values = {key: ref.tensor_difference(value, target['weights'][key])
                                  for key, value in weights.items()}
                        updates = {key: ref.tensor_difference(value - original['weights'][key],
                                                              target['weights'][key] - original['weights'][key], atol=1e-8)
                                   for key, value in weights.items()}
                        report['steps'].append({'step': state.global_step, 'full': target['step'], 'chunk': record,
                            'parameters_bitwise_equal': all(torch.equal(value, target['weights'][key])
                                                            for key, value in weights.items()),
                            'parameter_values': summarize_statistics(values),
                            'parameter_changes': summarize_statistics(updates)})
                    if on_progress:
                        on_progress(report)
                    return control

            observer = Observe()
            trainer = create_trainer(name, callbacks=[observer])
            if trainer.model is not carrier.model:
                raise ValueError('Trajectory factory must reuse the common frozen base and adapter')
            observer.trainer = trainer
            if carrier.args.device.type == 'cuda' and trainer.accelerator.scaler is None:
                raise ValueError('Actual FP16 trajectory must use GradScaler')
            with preserve_forward(trainer.model), full_forward_loss(name == 'full'):
                trainer.train()
            optimizer = trainer.optimizer
            while hasattr(optimizer, 'optimizer'):
                optimizer = optimizer.optimizer
            counts = sorted({int(state['step']) for state in optimizer.state.values() if 'step' in state})
            del optimizer
            report['paths'][name] = {
                'scaler_enabled': trainer.accelerator.scaler is not None,
                'global_steps': trainer.state.global_step,
                'history': list(trainer.history),
                'optimizer_state_step_counts': counts,
                'nonzero_update': any(not torch.equal(original['weights'][key], value)
                                      for key, value in snapshot(trainer.model).items())}
            report[name + '_source_order'] = [row['source_index'] for row in trainer.history]
            del observer, trainer
            gc.collect()
        report['steps_per_path'] = len(expected)
        report['nonzero_parameter_updates'] = all(value['nonzero_update'] for value in report['paths'].values())
        report['same_input_order'] = report['full_source_order'] == report['chunk_source_order']
        report['same_step_count'] = (len(report['steps']) == len(expected)
                                    == report['paths']['full']['global_steps']
                                    == report['paths']['chunk']['global_steps'] == carrier.args.max_steps)
        report['all_parameter_states_finite'] = bool(report['steps']) and all(
            row['full']['parameters_finite'] and row['chunk']['parameters_finite'] for row in report['steps'])
        report['all_parameter_states_bitwise_equal'] = all(row['parameters_bitwise_equal'] for row in report['steps'])
        report['status'] = 'RECORDED'
        if on_progress:
            on_progress(report)
    return report
