"""Bounded real-GradScaler and 16-step TLM update diagnostics."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

from probe_support import HERE, V3, DIAG, V3_ID, DIAG_SUMS, reference, snapshot


def verify_dependencies():
    ref = reference()
    for directory in (HERE, V3, DIAG):
        ref.verify_delivery(directory)
    if hashlib.sha256((DIAG / 'SHA256SUMS').read_bytes()).hexdigest() != DIAG_SUMS:
        raise ValueError('Use the unchanged first diagnostic bundle')
    sys.path.insert(0, str(V3))
    from runtime import code_identity
    if code_identity() != V3_ID:
        raise ValueError('Use the unchanged frozen v3 bundle')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=['llama3b', 'qwen3b'], required=True)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error('Use a new output directory')
    verify_dependencies()
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    from runtime import atomic_json, gpu_lock, specifications, read_inputs, LoadedModel, frozen_digest
    from run import probe_training_arguments, PROBE_LAMB, PROBE_STEPS
    from train_core import TLMTrainer, InputCollator
    from transformers.trainer_utils import IntervalStrategy
    from scaled_probe import compare_scaled_gradients
    from trajectory import compare_trajectories
    import torch
    with gpu_lock():
        args.output.mkdir(parents=True, exist_ok=False)
        metadata = {'model': args.model, 'dataset': 'dolly_5k', 'seed': 42, 'lora_init': 'gaussian',
                    'v3_code_id': V3_ID, 'first_diagnostic_delivery': DIAG_SUMS,
                    'scaled_diagnostic_delivery': hashlib.sha256((HERE / 'SHA256SUMS').read_bytes()).hexdigest(),
                    'not_a_preflight_receipt': True, 'ready_for_full_training': False,
                    'torch': torch.__version__, 'cuda': torch.version.cuda,
                    'gpu': torch.cuda.get_device_name(0), 'gpu_index': os.environ.get('CUDA_VISIBLE_DEVICES')}
        try:
            _, spec = specifications(args.model, 'dolly_5k')
            rows, prepared = read_inputs(args.inputs, spec['sha256'])
            metadata['input_sha256'] = prepared['files']['inputs.jsonl']
            loaded = LoadedModel(args.model, 'dolly_5k', seed=42, initialization='gaussian')
            encoded = loaded.tokens(rows)
            indices = sorted(range(len(encoded)), key=lambda i: (-len(encoded[i][0]), i))[:4]
            features = [{'source_index': i, 'input_ids': encoded[i][0][:258]} for i in indices]
            if len(features) != 4 or any(len(item['input_ids']) <= 129 for item in features):
                raise ValueError('Four real inputs spanning multiple chunks are required')
            metadata['features'] = [{'source_index': item['source_index'], 'tokens': len(item['input_ids']),
                                      'tokens_sha256': hashlib.sha256(json.dumps(item['input_ids']).encode()).hexdigest()}
                                     for item in features]
            metadata['contract'] = loaded.contract

            def create(name, callbacks=None):
                configuration = probe_training_arguments(args.output / name, 42)
                configuration.save_strategy = IntervalStrategy.NO
                configuration.logging_strategy = IntervalStrategy.NO
                return TLMTrainer(model=loaded.model, args=configuration, train_dataset=features,
                                  data_collator=InputCollator(loaded.pad_id), threshold=0., lamb=PROBE_LAMB,
                                  chunk_size=128, callbacks=callbacks or [])

            carrier = create('unused_carrier')
            original_adapter = snapshot(loaded.model)
            metadata['initial_scaler_state'] = carrier.accelerator.scaler.state_dict()
            metadata['engineering_update_recipe'] = {'steps_per_path': PROBE_STEPS, 'learning_rate': 1e-6,
                'batch_size': 1, 'warmup_ratio': .1, 'lr_scheduler': 'cosine', 'max_grad_norm': 1.,
                'threshold': 0., 'lambda': PROBE_LAMB,
                'input_policy': 'Four real long prefixes repeated for a bounded engineering update comparison'}
            def save_gradient(report):
                atomic_json(args.output / 'scaled_report.json', {**metadata, **report})
                print(json.dumps({'phase': 'scaled_gradient', 'status': report['comparison_status'],
                                  'states': {key: {'attempts': len(value['attempts']), 'finite_scale': value['finite_scale']}
                                             for key, value in report['states'].items()}}), flush=True)
            scaled = compare_scaled_gradients(carrier, features[0], on_progress=save_gradient)
            save_gradient(scaled)
            def save_trajectory(report):
                atomic_json(args.output / 'trajectory_report.json', {**metadata, **report})
                print(json.dumps({'phase': 'update_trajectory', 'status': report['status'],
                                  'progress': report.get('progress', {})}), flush=True)
            if all(value['finite_scale'] is not None for value in scaled['states'].values()):
                trajectory = compare_trajectories(carrier, create, on_progress=save_trajectory)
                save_trajectory(trajectory)
            else:
                trajectory = {'status': 'NOT_RUN_NONFINITE_PROBE'}
                save_trajectory(trajectory)
            unchanged = frozen_digest(loaded.model) == loaded.contract['frozen_sha256']
            if not unchanged:
                raise ValueError('Frozen base changed')
            restored_adapter = snapshot(loaded.model)
            adapter_restored = all(torch.equal(value, restored_adapter[key]) for key, value in original_adapter.items())
            if not adapter_restored:
                raise ValueError('Original adapter was not restored exactly')
            result = {**metadata, 'diagnostic_status': 'RECORDED', 'scaled_pair_status': scaled['comparison_status'],
                      'trajectory_status': trajectory['status'], 'base_unchanged': unchanged,
                      'original_adapter_restored': adapter_restored,
                      'nonzero_engineering_updates': trajectory.get('nonzero_parameter_updates', False),
                      'same_input_order': trajectory.get('same_input_order', False),
                      'same_step_count': trajectory.get('same_step_count', False),
                      'all_parameter_states_finite': trajectory.get('all_parameter_states_finite', False),
                      'full_training_started': False}
            atomic_json(args.output / 'run_summary.json', result)
            print('SUMMARY=' + str(args.output / 'run_summary.json'), flush=True)
        except BaseException as error:
            atomic_json(args.output / 'failure.json', {**metadata, 'type': type(error).__name__, 'message': str(error)})
            raise
    return 0 if (scaled['comparison_status'] == 'MATCH' and result['nonzero_engineering_updates']
                 and result['same_input_order'] and result['same_step_count']
                 and result['all_parameter_states_finite']) else 2


if __name__ == '__main__':
    raise SystemExit(main())
