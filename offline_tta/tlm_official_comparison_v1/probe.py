"""Eight real Trainer visits: pinned official loss versus the frozen v6 path.

Both sides deliberately share Gaussian adapters and FP16. This is a controlled
implementation comparison, not a run of the official default/BF16 recipe.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time

REPO=Path('/home/chuwenjing/TTA_LLM/TLM')
RUNTIME=REPO/'local_repro/tlm_offline_b6da067b55f5'
STEPS=8
OFFICIAL_HASHES={
    'src/llamafactory/train/ttl/trainer.py':'c2b833d9ae51355349b86dbe3a6355d99c5c2cd02d7cc371430532cb05e436c6',
    'src/llamafactory/data/processor/self_supervised.py':'e08fd66f627961a6958f7684758db32ff4d0db6b698a15beeaed424f6dbfac6a',
    'src/llamafactory/model/adapter.py':'6aa865cdac38803699824a444f064b8beb960e3d137de584b715c438fb357592',
}

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',choices=['llama3b','qwen3b'],required=True)
    parser.add_argument('--gpu',type=int,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    model_key=args.model
    root=Path('/data/chuwenjing/results/tlm_official_comparison').resolve()
    out=args.output.resolve()
    if root not in out.parents or len(out.relative_to(root).parts)!=2:
        raise ValueError('Output must be a new comparison run/model directory')
    if out.exists():
        raise ValueError('Refusing to reuse an existing output directory')
    os.environ.update(CUDA_VISIBLE_DEVICES=str(args.gpu),CUBLAS_WORKSPACE_CONFIG=':4096:8',
                      HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
    sys.path[:0]=[str(RUNTIME),str(REPO/'src')]
    from runtime import LoadedModel,code_identity,gpu_lock,atomic_json,frozen_digest

    if code_identity()!='b6da067b55f5c65241ae6fbd8c706b88523f64b27dd9568ad5661255b89d7999':
        raise ValueError('Original runtime identity changed')
    if any(sha(REPO/rel)!=expected for rel,expected in OFFICIAL_HASHES.items()):
        raise ValueError('Official sources differ from independently fetched fixed commit')
    formal=Path('/data/chuwenjing/results/tlm_offline_v6_ready/run_20261026_000443_2ca89d/formal')/(args.model+'__gsm8k_5k')
    protected=[formal/name for name in ['metrics.json','adaptation.json','training_complete.json','final_adapter/adapter_model.safetensors']]
    protected += [Path('/data/chuwenjing/tta_offline/tlm_inputs_v1/gsm8k_5k/inputs.jsonl')]
    before={str(p):sha(p) for p in protected}
    original_history=json.loads((formal/'adaptation.json').read_text())[:STEPS]
    rows=[json.loads(s) for s in protected[-1].read_text().splitlines()]
    assert len(rows)==5000
    assert all(set(r)=={'source_index','instruction','input'} for r in rows)
    started=time.monotonic()
    with gpu_lock():
        # Some upstream imports probe CUDA capability and create a context.
        # Take the shared idle-device lock before importing those modules.
        import torch
        from datasets import Dataset
        from transformers import Seq2SeqTrainingArguments,TrainerCallback
        from llamafactory.train.ttl.trainer import CustomSeq2SeqTrainer
        from llamafactory.hparams import FinetuningArguments
        from train_core import TLMTrainer,InputCollator
        from run import training_arguments,restore_adapter,snapshot_adapter
        out.mkdir(parents=True,exist_ok=False)
        atomic_json(out/'status.json',{'status':'RUNNING','model':args.model,'steps':STEPS,'scope':'controlled FP16 shared-Gaussian comparison'})
        loaded=LoadedModel(args.model,'gsm8k_5k',42,initialization='gaussian')
        loaded.tokenizer.padding_side='right'
        encoded=loaded.tokens(rows)
        dataset=Dataset.from_list([{'source_index':row['source_index'],'input_ids':token[0]} for row,token in zip(rows,encoded)])
        initial=snapshot_adapter(loaded.model)
        reference=[]
        records=[]
        current_mode=[None]
        capture={}

        def snapshot_optimizer(optimizer):
            return [{k:v.detach().cpu().clone() for k,v in state.items() if torch.is_tensor(v)} for state in optimizer.state.values()]

        def differences(left,right):
            a=torch.cat([t.reshape(-1).double() for t in left])
            b=torch.cat([t.reshape(-1).double() for t in right])
            diff=a-b
            norm=float(a.norm())
            return {'exact':bool(torch.equal(a,b)),'max_abs':float(diff.abs().max()),
                    'relative_l2':float(diff.norm())/norm if norm else (0. if not bool(diff.count_nonzero()) else None),
                    'reference_l2':norm,'finite':bool(torch.isfinite(a).all() and torch.isfinite(b).all()),
                    'outside_prior_gradient_tolerance':int((~torch.isclose(a,b,atol=1e-5,rtol=1e-2)).sum())}

        class Capture(TrainerCallback):
            def on_pre_optimizer_step(self,args,state,control,model=None,optimizer=None,**kwargs):
                parameters=[p for p in model.parameters() if p.requires_grad]
                if any(p.grad is None for p in parameters):
                    raise ValueError('Missing LoRA gradient')
                capture['gradients']=[p.grad.detach().cpu().clone() for p in parameters]
            def on_step_end(self,args,state,control,model=None,optimizer=None,**kwargs):
                trainer=current_mode[0]
                params=snapshot_adapter(model)
                step={'parameters':params,'gradients':capture.pop('gradients'),
                      'optimizer':snapshot_optimizer(optimizer),
                      'scale':trainer.accelerator.scaler.get_scale(),
                      'history':dict(trainer.history[-1]),'step':state.global_step}
                if trainer.mode=='official':
                    reference.append(step)
                    print(json.dumps({'phase':'official_step','model':model_key,'step':state.global_step,'history':step['history']}),flush=True)
                else:
                    ref=reference[state.global_step-1]
                    keys=list(initial)
                    deltas_a=[ref['parameters'][k]-initial[k] for k in keys]
                    deltas_b=[params[k]-initial[k] for k in keys]
                    opt_a=[v for item in ref['optimizer'] for v in item.values()]
                    opt_b=[v for item in step['optimizer'] for v in item.values()]
                    row={'step':state.global_step,'official':ref['history'],'current':step['history'],
                         'gradients':differences(ref['gradients'],step['gradients']),
                         'adapter_update':differences(deltas_a,deltas_b),
                         'optimizer_state':differences(opt_a,opt_b),
                         'scaler_exact':ref['scale']==step['scale']}
                    records.append(row)
                    atomic_json(out/'steps.json',records)
                    print(json.dumps({'phase':'comparison_step','model':model_key,**row}),flush=True)
                if state.global_step>=STEPS:
                    control.should_training_stop=True
                    control.should_save=False
                return control

        class Official(CustomSeq2SeqTrainer):
            mode='official'
            def __init__(self,**kwargs):
                self.history=[]
                super().__init__(**kwargs)
            def cal_ce(self,logits,labels):
                value=super().cal_ce(logits,labels)
                self.last_base=float(value[0])
                self.last_weight=float((.1*torch.exp(value.detach()-3.))[0])
                return value
            def cal_kl(self,logits,labels):
                value=super().cal_kl(logits,labels)
                self.last_current=float(value[0].detach())
                return value
            def compute_loss(self,model,inputs,return_outputs=False,**kwargs):
                inputs=dict(inputs)
                index=int(inputs.pop('source_index')[0])
                loss=super().compute_loss(model,inputs,return_outputs=return_outputs,**kwargs)
                self.history.append({'source_index':index,'base_ce':self.last_base,
                    'current_ce':self.last_current,'selected':self.last_base>3.,
                    'weight':self.last_weight,
                    'loss':float((loss[0] if return_outputs else loss).detach()),
                    'lr':float(self.optimizer.param_groups[0]['lr'])})
                return loss

        for mode in ['official','current']:
            restore_adapter(loaded.model,initial)
            loaded.model.zero_grad(set_to_none=True)
            loaded.model.train()
            configuration=training_arguments(out/mode,1e-6,42,5000)
            # Preserve the full 5000-visit sampler/schedule; stop only by callback.
            configuration.save_strategy='no'
            configuration.logging_strategy='no'
            configuration.disable_tqdm=True
            Path(configuration.output_dir).mkdir(parents=True,exist_ok=False)
            if mode=='official':
                fields={k:getattr(configuration,k) for k in [
                    'output_dir','num_train_epochs','per_device_train_batch_size','gradient_accumulation_steps',
                    'learning_rate','lr_scheduler_type','warmup_ratio','optim','weight_decay','max_grad_norm',
                    'fp16','bf16','gradient_checkpointing','gradient_checkpointing_kwargs','seed','data_seed',
                    'remove_unused_columns','dataloader_num_workers','disable_tqdm']}
                fields.update(save_strategy='no',logging_strategy='no',report_to=[])
                official_args=Seq2SeqTrainingArguments(**fields)
                trainer=Official(model=loaded.model,args=official_args,train_dataset=dataset,
                    data_collator=InputCollator(loaded.pad_id),tokenizer=loaded.tokenizer,processor=None,
                    finetuning_args=FinetuningArguments(lora_target='q_proj,v_proj',setting='offline_ttl',threshold=3.,lamb=.1),callbacks=[Capture()])
            else:
                trainer=TLMTrainer(model=loaded.model,args=configuration,train_dataset=dataset,
                    data_collator=InputCollator(loaded.pad_id),threshold=3.,lamb=.1,chunk_size=128,callbacks=[Capture()])
                trainer.mode='current'
            current_mode[0]=trainer
            trainer.train()
            assert len(trainer.history)==STEPS
            loaded.model=trainer.accelerator.unwrap_model(trainer.model,keep_fp32_wrapper=False)
            current_mode[0]=None
            del trainer
            gc.collect();torch.cuda.empty_cache()

        original_replay=[]
        for row,saved in zip(records,original_history):
            now=row['current']
            original_replay.append({'step':row['step'],'source_index_exact':now['source_index']==saved['source_index'],
                'selection_exact':now['selected']==saved['selected'],'lr_exact':now['lr']==saved['lr'],
                'base_ce_abs_difference':abs(now['base_ce']-saved['base_ce']),
                'current_ce_abs_difference':abs(now['current_ce']-saved['current_ce']),
                'loss_abs_difference':abs(now['loss']-saved['loss'])})
        frozen_unchanged=frozen_digest(loaded.model)==loaded.contract['frozen_sha256']
        after={str(p):sha(p) for p in protected}
        same_indices=all(r['official']['source_index']==r['current']['source_index'] for r in records)
        same_decisions=all(r['official']['selected']==r['current']['selected'] for r in records)
        summary={'status':'COMPLETE','model':args.model,'steps_per_path':STEPS,'dataset_size':5000,
            'shared_initialization':'Gaussian A / zero B, seed 42; controlled, not upstream default',
            'precision':'FP16 base and autocast, FP32 adapters; BF16 not tested',
            'input_lengths':[len(encoded[r['current']['source_index']][0]) for r in records],
            'same_sample_order':same_indices,'same_selection':same_decisions,
            'loss_max_abs_difference':max(abs(r['official']['loss']-r['current']['loss']) for r in records),
            'gradients_max_abs_difference':max(r['gradients']['max_abs'] for r in records),
            'gradients_max_relative_l2':max(r['gradients']['relative_l2'] or 0 for r in records),
            'adapter_update_max_relative_l2':max(r['adapter_update']['relative_l2'] or 0 for r in records),
            'all_gradients_exact':all(r['gradients']['exact'] for r in records),
            'all_adapter_updates_exact':all(r['adapter_update']['exact'] for r in records),
            'all_optimizer_states_exact':all(r['optimizer_state']['exact'] for r in records),
            'all_scalers_exact':all(r['scaler_exact'] for r in records),
            'original_history_replay':original_replay,'frozen_base_unchanged':frozen_unchanged,
            'protected_files_unchanged':before==after,'protected_sha256':after,
            'seconds':time.monotonic()-started,'peak_allocated_mib':torch.cuda.max_memory_allocated()/1024**2,
            'scope_limit':'8 GSM8K visits per model; not full workflow/BF16/default initialization equivalence'}
        atomic_json(out/'summary.json',summary)
        atomic_json(out/'status.json',{'status':'COMPLETE','model':args.model,'steps':STEPS})
        print(json.dumps(summary),flush=True)
        if not (frozen_unchanged and before==after and same_indices and same_decisions):
            raise ValueError('Comparison integrity check failed; evidence preserved')

if __name__=='__main__':
    main()
