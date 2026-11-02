"""Isolated whole-head GEMM/streamed CE diagnostic; never formal runs."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--code',type=Path,required=True)
    parser.add_argument('--model',choices=['llama3b','qwen3b'],required=True)
    parser.add_argument('--dataset',choices=['geosignal_5k','dolly_5k','gsm8k_5k'],required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():parser.error('New output required')
    sys.path.insert(0,str(args.code))
    from runtime import code_identity,LoadedModel,specifications,read_inputs,atomic_json,gpu_lock,frozen_digest
    from launch import verify_delivery
    verify_delivery(args.code)
    assert code_identity()=='f1b5c7026a9429c4008b923a42b459ae47070585d51ec28b70498806a85a459b'
    from run import probe_training_arguments,PROBE_LAMB
    from train_core import TLMTrainer,InputCollator
    import train_core
    from precision import compare_precision_paths
    from full_projection import full_projection_cross_entropy
    import torch
    from datasets import Dataset
    from transformers.trainer_utils import IntervalStrategy
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
    os.environ.setdefault('HF_HUB_OFFLINE','1');os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
    probe=Path(__file__).resolve().parent
    report={'model':args.model,'dataset':args.dataset,'diagnostic_only':True,'ready_for_formal':False,
            'probe_files':{p.relative_to(probe).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in sorted(probe.rglob('*.py')) if '__pycache__' not in p.parts},
            'policy':'original whole FP16 head GEMMs including unused final position; CPU logits; streamed FP32 CE',
            'precision_checks':[]}
    with gpu_lock():
        args.output.mkdir(parents=True)
        original_loss=train_core.model_input_cross_entropy
        try:
            train_core.model_input_cross_entropy=full_projection_cross_entropy
            _,spec=specifications(args.model,args.dataset)
            rows,prepared=read_inputs(Path('/data/chuwenjing/tta_offline/tlm_inputs_v1')/args.dataset,spec['sha256'])
            report.update(input_sha256=prepared['files']['inputs.jsonl'],torch=torch.__version__,
                cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0),physical_gpu=os.environ['CUDA_VISIBLE_DEVICES'],
                fp16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction)
            loaded=LoadedModel(args.model,args.dataset,42,initialization='gaussian')
            encoded=loaded.tokens(rows)
            longest=max(range(len(encoded)),key=lambda i:len(encoded[i][0]))
            initial={n:p.detach().cpu().clone() for n,p in loaded.model.named_parameters() if p.requires_grad}
            head_calls=set()
            hook=loaded.model.get_base_model().get_output_embeddings().register_forward_hook(
                lambda m,a,o:head_calls.add((a[0].shape[1],str(a[0].dtype),str(o.dtype))))
            def create(name,features,steps=16):
                config=probe_training_arguments(args.output/name,42)
                config.max_steps=steps;config.save_strategy=IntervalStrategy.NO
                config.logging_steps=1
                return TLMTrainer(model=loaded.model,args=config,train_dataset=Dataset.from_list(features),
                    data_collator=InputCollator(loaded.pad_id),threshold=0.,lamb=PROBE_LAMB,chunk_size=128)
            report['declared_lengths']=sorted({min(n,len(encoded[longest][0])) for n in [130,258,386]})
            for length in report['declared_lengths']:
                feature={'source_index':longest,'input_ids':encoded[longest][0][:length]}
                carrier=create('precision_'+str(length),[feature])
                result=compare_precision_paths(carrier,feature)
                result.update(projection_policy=report['policy'],projection_chunks=1,
                              ce_chunks=(length-2)//128+1)
                report['precision_checks'].append({'tokens':length,**result})
                atomic_json(args.output/'report.json',report)
                print(json.dumps({'phase':'precision','tokens':length,'status':result['status'],
                    'max_abs':result['max_gradient_absolute_difference'],'relative_l2':result['max_gradient_relative_l2_difference']}),flush=True)
                del carrier;gc.collect();torch.cuda.empty_cache()
            # Allocate real optimizer state and exercise checkpoint recomputation
            # under the actual mixed-precision Trainer on the longest input.
            loaded.model.zero_grad(set_to_none=True)
            fullfeature={'source_index':longest,'input_ids':encoded[longest][0]}
            trainer=create('longest_actual_trainer',[fullfeature],steps=16)
            torch.cuda.reset_peak_memory_stats();start=time.monotonic()
            trainer.train()
            steps={int(v['step']) for v in trainer.optimizer.state.values() if 'step' in v}
            parameters_finite=all(bool(torch.isfinite(p).all()) for p in loaded.model.parameters() if p.requires_grad)
            optimizer_finite=all(bool(torch.isfinite(v).all()) for state in trainer.optimizer.state.values()
                                 for v in state.values() if torch.is_tensor(v))
            adapter_changed=any(not torch.equal(initial[n],p.detach().cpu())
                                for n,p in loaded.model.named_parameters() if p.requires_grad)
            report['longest_actual_trainer']={'tokens':len(fullfeature['input_ids']),'global_steps':trainer.state.global_step,
                'optimizer_update_counts':sorted(steps),'seconds':time.monotonic()-start,
                'peak_allocated_mib':torch.cuda.max_memory_allocated()/1024**2,
                'peak_reserved_mib':torch.cuda.max_memory_reserved()/1024**2,
                'all_history_finite':all(all(__import__('math').isfinite(row[k]) for k in ['base_ce','current_ce','weight','loss','lr']) for row in trainer.history),
                'parameters_finite':parameters_finite,'optimizer_finite':optimizer_finite,'adapter_changed':adapter_changed}
            if not (len(steps)==1 and min(steps)>=2 and trainer.state.global_step==16
                    and parameters_finite and optimizer_finite and adapter_changed
                    and report['longest_actual_trainer']['all_history_finite']):
                raise ValueError('Longest actual Trainer lacks successful finite updates')
            del trainer;loaded.model.zero_grad(set_to_none=True)
            with torch.no_grad():
                for name,p in loaded.model.named_parameters():
                    if p.requires_grad:p.copy_(initial[name].to(p.device))
            report['base_unchanged']=frozen_digest(loaded.model)==loaded.contract['frozen_sha256']
            report['adapter_restored']=all(torch.equal(initial[n],p.detach().cpu()) for n,p in loaded.model.named_parameters() if p.requires_grad)
            report['head_calls']=sorted(head_calls);hook.remove()
            if not report['base_unchanged']:raise ValueError('Frozen base changed')
            if not report['adapter_restored']:raise ValueError('Adapter not restored')
            report['status']='PASS'
        except BaseException as error:
            report['status']='FAIL';report['error']={'type':type(error).__name__,'message':str(error)}
            raise
        finally:
            train_core.model_input_cross_entropy=original_loss
            atomic_json(args.output/'report.json',report)
            print(json.dumps({k:v for k,v in report.items() if k!='precision_checks'},allow_nan=False),flush=True)


if __name__=='__main__':main()
