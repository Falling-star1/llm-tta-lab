"""Single-GPU offline TLM preflight, pilot, formal training and exact resume."""
import argparse
import copy
import json
import os
from pathlib import Path
import time

from runtime import (HERE, atomic_json, code_identity, frozen_digest, output_lock, gpu_lock,
                     read_inputs, sha256, specifications)

TENT_RUN=Path('/data/chuwenjing/results/tent_formal_v3_6gpu/run_20261020_001210_900089')


def require_preflight(path, expected):
    record=json.loads(Path(path).read_text(encoding='utf-8'))
    if any(record.get(k)!=v for k,v in expected.items()):
        raise ValueError('Preflight belongs to different code/model/data')
    if not (record.get('status')=='PASS' and record.get('frozen_base_unchanged') is True
            and record.get('checkpoint_continuation')=='EXACT_MATCH'
            and record.get('source_sample_match') is True):
        raise ValueError('A complete real-model preflight is required')
    return record


def read_predictions(path):
    path=Path(path)
    if not path.exists():
        return []
    raw=path.read_bytes()
    end=len(raw) if raw.endswith(b'\n') else raw.rfind(b'\n')+1
    rows=[json.loads(x) for x in raw[:end].splitlines()]
    if any(row.get('source_index')!=index for index,row in enumerate(rows)):
        raise ValueError('Prediction journal contains duplicate/missing indexes')
    if end!=len(raw):
        with path.open('r+b') as stream:
            stream.truncate(end)
    return rows


def emit(**record):
    print(json.dumps(record,ensure_ascii=False,allow_nan=False),flush=True)


def identity(args, prepared):
    return {'code_id':code_identity(),'model':args.model,'dataset':args.dataset,
            'input_sha256':prepared['files']['inputs.jsonl']}


def snapshot_adapter(model):
    return {name:p.detach().cpu().clone() for name,p in model.named_parameters() if p.requires_grad}


def restore_adapter(model, snapshot):
    import torch
    with torch.no_grad():
        for name,p in model.named_parameters():
            if p.requires_grad:
                p.copy_(snapshot[name].to(p.device))


def probe_trainer_resume(loaded, encoded, initial, args):
    """Actual FP16 Trainer checkpoint/scheduler/scaler/sampler continuation."""
    import gc
    import torch
    from datasets import Dataset
    from transformers import TrainerCallback
    from train_core import TLMTrainer,InputCollator,verify_checkpoint
    class StopAtTwo(TrainerCallback):
        def on_step_end(self,args,state,control,**kwargs):
            if state.global_step==2:
                control.should_training_stop=True
                control.should_save=True
            return control
    # Bound diagnostic sequence length, not formal input length.
    features=[{'source_index':i,'input_ids':ids[:min(96,len(ids))]}
              for i,(ids,_) in enumerate(encoded[:4])]
    dataset=Dataset.from_list(features)
    def create(name,stop=False):
        output=args.output/'resume_probe'/name
        configuration=training_arguments(output,1e-6,args.seed,4)
        configuration.max_steps=4
        configuration.save_steps=2
        configuration.logging_steps=1
        restore_adapter(loaded.model,initial)
        loaded.model.zero_grad(set_to_none=True)
        loaded.model.train()
        return TLMTrainer(model=loaded.model,args=configuration,train_dataset=dataset,
            data_collator=InputCollator(loaded.pad_id),threshold=0.,chunk_size=args.chunk_size,
            callbacks=[StopAtTwo()] if stop else [])
    full=create('full')
    full.train()
    expected=snapshot_adapter(loaded.model)
    full_history=copy.deepcopy(full.history)
    full_checkpoint=args.output/'resume_probe/full/trainer/checkpoint-4'
    verify_checkpoint(full_checkpoint)
    del full
    gc.collect(); torch.cuda.empty_cache()
    partial=create('partial',True)
    partial.train()
    partial_checkpoint=args.output/'resume_probe/partial/trainer/checkpoint-2'
    verify_checkpoint(partial_checkpoint)
    del partial
    gc.collect(); torch.cuda.empty_cache()
    resumed=create('partial')
    resumed.resume_history(partial_checkpoint)
    resumed.train(resume_from_checkpoint=str(partial_checkpoint))
    actual=snapshot_adapter(loaded.model)
    if full_history!=resumed.history or not all(torch.equal(expected[k],actual[k]) for k in expected):
        raise ValueError('Actual FP16 Trainer disk continuation differs')
    resume_checkpoint=args.output/'resume_probe/partial/trainer/checkpoint-4'
    verify_checkpoint(resume_checkpoint)
    def equal(left,right):
        if torch.is_tensor(left):
            return torch.equal(left,right)
        if isinstance(left,dict):
            return left.keys()==right.keys() and all(equal(left[k],right[k]) for k in left)
        if isinstance(left,(list,tuple)):
            return len(left)==len(right) and all(equal(a,b) for a,b in zip(left,right))
        return left==right
    state_files=['optimizer.pt','scheduler.pt','rng_state.pth','scaler.pt']
    # numpy RNG is compared explicitly rather than using array truth values.
    def state(path):
        value=torch.load(path,map_location='cpu',weights_only=False)
        if path.name=='rng_state.pth' and 'numpy' in value:
            value['numpy']=repr(value['numpy'])
        return value
    for name in state_files:
        if not (full_checkpoint/name).exists() or not (resume_checkpoint/name).exists():
            raise ValueError('FP16 resume probe lacks '+name)
        if not equal(state(full_checkpoint/name),state(resume_checkpoint/name)):
            raise ValueError('Restored FP16 training state differs: '+name)
    if not any(not torch.equal(initial[k],actual[k]) for k in initial):
        raise ValueError('FP16 resume probe made no parameter updates')
    del resumed
    gc.collect(); torch.cuda.empty_cache()
    restore_adapter(loaded.model,initial)
    return {'status':'EXACT_MATCH','steps':4,'interrupted_at':2,
            'state_files':state_files,'full_commit_sha256':sha256(full_checkpoint/'commit.json'),
            'resume_commit_sha256':sha256(resume_checkpoint/'commit.json')}


def preflight(args, rows, prepared):
    import random
    import numpy as np
    import torch
    from runtime import LoadedModel
    from tlm_loss import model_input_cross_entropy, input_cross_entropy, weighted_input_loss
    loaded=LoadedModel(args.model,args.dataset,args.seed)
    encoded=loaded.tokens(rows)
    initial=snapshot_adapter(loaded.model)
    before=loaded.contract['frozen_sha256']
    longest=max(range(len(encoded)),key=lambda i:len(encoded[i][0]))
    indices=sorted({0,longest,max(range(len(encoded)),key=lambda i:encoded[i][1]['original_prompt_tokens'])})
    checks=[]
    torch.cuda.reset_peak_memory_stats()
    params=[p for p in loaded.model.parameters() if p.requires_grad]
    for index in indices:
        ids,info=encoded[index]
        tensor=torch.tensor([ids],device=loaded.device)
        restore_adapter(loaded.model,initial)
        loaded.model.eval()
        with torch.no_grad(),loaded.model.disable_adapter():
            base_ce=model_input_cross_entropy(loaded.model,tensor,chunk_size=args.chunk_size)
        if index==0:
            # Whole-vocabulary reference is deliberately bounded for this check.
            short=tensor[:,:min(128,tensor.shape[1])]
            with torch.no_grad():
                ref=input_cross_entropy(loaded.model(input_ids=short,use_cache=False).logits,short)
                got=model_input_cross_entropy(loaded.model,short,chunk_size=args.chunk_size)
            torch.testing.assert_close(got,ref,atol=1e-5,rtol=1e-3)
        loaded.model.train()
        opt=torch.optim.AdamW(params,lr=1e-6,weight_decay=0.)
        def step():
            opt.zero_grad(set_to_none=True)
            ce=model_input_cross_entropy(loaded.model,tensor,chunk_size=args.chunk_size)
            # Engineering probe forces selection; formal protocol remains >3.
            loss=weighted_input_loss(ce,base_ce,threshold=0.)
            loss.backward()
            if not all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in params):
                raise FloatingPointError('Missing or nonfinite adapter gradients')
            torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
            opt.step()
            return float(ce.detach()),float(loss.detach())
        started=time.monotonic()
        ce,loss=step()
        checkpoint=snapshot_adapter(loaded.model)
        if not any(not torch.equal(initial[k],checkpoint[k]) for k in initial):
            raise ValueError('Engineering step did not update adapters')
        optimizer_state=copy.deepcopy(opt.state_dict())
        rng=(random.getstate(),np.random.get_state(),torch.get_rng_state(),torch.cuda.get_rng_state_all())
        step()
        expected=snapshot_adapter(loaded.model)
        restore_adapter(loaded.model,checkpoint)
        opt.load_state_dict(optimizer_state)
        random.setstate(rng[0]); np.random.set_state(rng[1]); torch.set_rng_state(rng[2]); torch.cuda.set_rng_state_all(rng[3])
        step()
        actual=snapshot_adapter(loaded.model)
        if not all(torch.equal(expected[k],actual[k]) for k in expected):
            raise ValueError('Restored continuation differs')
        checks.append({'source_index':index,**info,'base_ce':float(base_ce),
                       'current_ce':ce,'engineering_loss':loss,'seconds':time.monotonic()-started})
        emit(phase='preflight_training',model=args.model,dataset=args.dataset,**checks[-1])
        del opt,tensor
    restore_adapter(loaded.model,initial)
    trainer_probe=probe_trainer_resume(loaded,encoded,initial,args)
    generated=loaded.generate(encoded[0][0])
    old=TENT_RUN/(args.model+'__'+args.dataset)/'source.jsonl'
    with old.open(encoding='utf-8') as stream:
        old_first=json.loads(next(stream))
    match=(generated['predict']==old_first['predict'] and
           encoded[0][1]['original_prompt_sha256']==old_first['original_prompt_sha256'])
    if not match:
        raise ValueError('Source sample differs: prepare a new matched Source before formal run')
    # Also test 512-token generation on the longest real input.
    if longest!=0:
        loaded.generate(encoded[longest][0])
    if frozen_digest(loaded.model)!=before:
        raise ValueError('Frozen base changed during preflight')
    record={**identity(args,prepared),'status':'PASS','seed':args.seed,
            'frozen_base_unchanged':True,'checkpoint_continuation':'EXACT_MATCH',
            'source_sample_match':match,'contract':loaded.contract,
            'fp16_trainer_disk_resume':trainer_probe,
            'checks':checks,'longest_index':longest,'chunk_size':args.chunk_size,
            'peak_allocated_mib':torch.cuda.max_memory_allocated()/1024**2,
            'peak_reserved_mib':torch.cuda.max_memory_reserved()/1024**2}
    atomic_json(args.output/'receipt.json',record)
    emit(phase='preflight',**record)


def training_arguments(output, lr, seed, count):
    from transformers import TrainingArguments
    return TrainingArguments(output_dir=str(output/'trainer'),num_train_epochs=1.,
        per_device_train_batch_size=1,gradient_accumulation_steps=1,learning_rate=lr,
        lr_scheduler_type='cosine',warmup_ratio=.1,optim='adamw_torch',
        adam_beta1=.9,adam_beta2=.999,adam_epsilon=1e-8,weight_decay=0.,max_grad_norm=1.,
        fp16=True,bf16=False,gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant':False},seed=seed,
        save_steps=25,save_total_limit=2,logging_steps=25,report_to=[],
        remove_unused_columns=False,dataloader_num_workers=0,disable_tqdm=True,
        save_safetensors=True)


def evaluate(loaded, encoded, prepared_dir, prepared, output):
    from runtime import Scorer
    path=Path(prepared_dir)/'references.jsonl'
    if sha256(path)!=prepared['files']['references.jsonl']:
        raise ValueError('Scoring references differ from prepared manifest')
    refs=[json.loads(x) for x in path.read_text(encoding='utf-8').splitlines()]
    if len(refs)!=prepared['record_count'] or any(x['source_index']!=i for i,x in enumerate(refs)):
        raise ValueError('Reference indexes differ')
    scorer=Scorer(loaded.dataset['metric'])
    journal=output/'tlm.jsonl'
    done=read_predictions(journal)
    if len(done)>len(encoded):
        raise ValueError('Too many predictions')
    start=time.monotonic()
    for i in range(len(done),len(encoded)):
        ids,info=encoded[i]
        result=loaded.generate(ids)
        row={**info,**result,'score':scorer.score(result['predict'],refs[i]['output'])}
        with journal.open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
            stream.flush()
            os.fsync(stream.fileno())
        done.append(row)
        if (i+1)%25==0:
            emit(phase='evaluation',completed=i+1,total=len(encoded),seconds=time.monotonic()-start)
    # Recompute all scores; journal values alone are not the final authority.
    scores=[scorer.score(x['predict'],refs[i]['output']) for i,x in enumerate(done)]
    if any(abs(v-row['score'])>1e-12 for v,row in zip(scores,done)):
        raise ValueError('Stored and recomputed scores differ')
    return {'n':len(done),'metric':loaded.dataset['metric'],'value_0_to_1':sum(scores)/len(scores),
            'generated_tokens':sum(x['generated_tokens'] for x in done),
            'hit_generation_limit_rate':sum(x['hit_generation_limit'] for x in done)/len(done),
            'unique_answers':len({x['predict'] for x in done}),
            'predictions_sha256':sha256(journal)}


def train(args,rows,prepared):
    import torch
    from datasets import Dataset
    from runtime import LoadedModel
    from train_core import TLMTrainer,InputCollator,verify_checkpoint
    expected=identity(args,prepared)
    receipt=require_preflight(args.receipt,expected)
    if receipt['chunk_size']!=args.chunk_size:
        raise ValueError('Projection configuration changed after preflight')
    manifest=args.output/'run_manifest.json'
    config={**expected,'seed':args.seed,'count':5000 if args.command=='formal' else args.samples,
            'chunk_size':args.chunk_size,'protocol':'A','initialized_from':'base',
            'source_run':str(TENT_RUN),'preflight_sha256':sha256(args.receipt)}
    if manifest.exists():
        old=json.loads(manifest.read_text())
        if old['config']!=config or args.resume is None:
            raise ValueError('Existing run requires identical configuration and explicit resume')
    else:
        if args.resume is not None:
            raise ValueError('Cannot resume without the original run manifest')
        atomic_json(manifest,{'status':'RUNNING','config':config,'stage':'initializing'})
    loaded=LoadedModel(args.model,args.dataset,args.seed)
    if loaded.contract['frozen_sha256']!=receipt['contract']['frozen_sha256']:
        raise ValueError('Base differs from the real preflight')
    count=config['count']
    encoded=loaded.tokens(rows[:count])
    atomic_json(args.output/'input_tokens.json',{'records':[info for _,info in encoded]})
    source=TENT_RUN/(args.model+'__'+args.dataset)/'source.jsonl'
    source_rows=[json.loads(x) for x in source.read_text(encoding='utf-8').splitlines()]
    if len(source_rows)!=5000:
        raise ValueError('Reusable Source is incomplete')
    for i,(_,info) in enumerate(encoded):
        old=source_rows[i]
        for key in ['source_index','original_prompt_sha256','used_prompt_tokens']:
            if old.get(key)!=info[key]:
                raise ValueError('Reusable Source input differs: '+str(i)+' '+key)
    atomic_json(args.output/'source_reuse.json',{'path':str(source),'sha256':sha256(source),
                 'matched_input_records':count,'original_records':5000,'sample_generation_match':True})
    features=[{'input_ids':ids,'source_index':i} for i,(ids,_) in enumerate(encoded)]
    lr=1e-6 if args.dataset=='gsm8k_5k' else 5e-5
    trainer=TLMTrainer(model=loaded.model,args=training_arguments(args.output,lr,args.seed,count),
        train_dataset=Dataset.from_list(features),data_collator=InputCollator(loaded.pad_id),
        chunk_size=args.chunk_size)
    atomic_json(args.output/'resolved_config.json',trainer.args.to_dict())
    atomic_json(args.output/'model_contract.json',loaded.contract)
    training_done=args.output/'training_complete.json'
    start=time.monotonic()
    if training_done.exists():
        # Restore the final adapter when only evaluation remains after a crash.
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file
        set_peft_model_state_dict(loaded.model,load_file(str(args.output/'final_adapter/adapter_model.safetensors')))
    else:
        if args.resume is not None:
            checkpoint=Path(args.resume).resolve()
            if not checkpoint.is_relative_to((args.output/'trainer').resolve()):
                raise ValueError('Checkpoint is outside this run')
            verify_checkpoint(checkpoint)
            trainer.resume_history(checkpoint)
        trainer.train(resume_from_checkpoint=str(args.resume) if args.resume else None)
        if len(trainer.history)!=count or {x['source_index'] for x in trainer.history}!=set(range(count)):
            raise ValueError('Training did not visit every input exactly once')
        if frozen_digest(loaded.model)!=loaded.contract['frozen_sha256']:
            raise ValueError('Base changed during adaptation')
        loaded.model.save_pretrained(str(args.output/'final_adapter'),safe_serialization=True,save_embedding_layers=False)
        atomic_json(args.output/'adaptation.json',trainer.history)
        atomic_json(training_done,{'n':count,'seconds':time.monotonic()-start,
             'selected':sum(x['selected'] for x in trainer.history),'global_step':trainer.state.global_step,
             'frozen_base_unchanged':True})
    atomic_json(manifest,{'status':'RUNNING','config':config,'stage':'evaluation'})
    metrics=evaluate(loaded,encoded,args.inputs,prepared,args.output)
    atomic_json(args.output/'metrics.json',metrics)
    atomic_json(manifest,{'status':'PASS','config':config,'stage':'complete',
                         'metrics':metrics,'peak_allocated_mib':torch.cuda.max_memory_allocated()/1024**2})
    emit(phase='complete',model=args.model,dataset=args.dataset,**metrics)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['preflight','pilot','formal'])
    parser.add_argument('--model',choices=['llama3b','qwen3b'],required=True)
    parser.add_argument('--dataset',choices=['geosignal_5k','dolly_5k','gsm8k_5k'],required=True)
    parser.add_argument('--inputs',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--receipt',type=Path)
    parser.add_argument('--resume',type=Path)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--samples',type=int,choices=[16,100],default=16)
    parser.add_argument('--chunk-size',type=int,default=128)
    args=parser.parse_args(argv)
    if args.command!='preflight' and args.receipt is None:
        parser.error('Training requires --receipt from a real GPU preflight')
    if args.output.exists() and args.resume is None:
        parser.error('Output exists; use a new directory or explicit --resume')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    os.environ.setdefault('HF_HUB_OFFLINE','1')
    os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
    os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
    profile,spec=specifications(args.model,args.dataset)
    rows,prepared=read_inputs(args.inputs,spec['sha256'])
    with gpu_lock(), output_lock(args.output):
        try:
            if args.command=='preflight':
                preflight(args,rows,prepared)
            else:
                train(args,rows,prepared)
        except BaseException as error:
            atomic_json(args.output/'failure.json',{'type':type(error).__name__,'message':str(error)})
            manifest=args.output/'run_manifest.json'
            if manifest.exists():
                record=json.loads(manifest.read_text())
                atomic_json(manifest,{**record,'status':'FAIL','error':str(error)})
            raise


if __name__=='__main__':
    main()
