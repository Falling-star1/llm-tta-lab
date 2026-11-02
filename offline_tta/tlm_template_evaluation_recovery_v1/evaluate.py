"""Evaluate preserved control weights after independently explaining AMP/LR drift."""
import argparse
import json
from pathlib import Path
import sys
import time
from schedule import verify_lr_trace

CODE='b6da067b55f5c65241ae6fbd8c706b88523f64b27dd9568ad5661255b89d7999'
BASES={'llama3b':'c80d597af8e0f5bb8e99a6545680edb580ed2fe59c9cba4af0f39b568c15be77',
       'qwen3b':'644c521c2e650bb46c5bfcedea082bbc3a02b9a5c0f67297a44633cb035384bc'}


def read(p):return json.loads(Path(p).read_text())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['runtime','failed','baseline','original','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--model',choices=BASES,required=True);args=p.parse_args()
    if any(args.output.resolve().is_relative_to(v.resolve()) for v in [args.failed,args.baseline,args.original]):
        raise ValueError('Recovery output must be outside every original and diagnostic run')
    args.output.mkdir(parents=True,exist_ok=False);sys.path.insert(0,str(args.runtime))
    import torch
    from accelerate import Accelerator
    from peft import set_peft_model_state_dict,get_peft_model_state_dict
    from safetensors.torch import load_file
    from transformers import GenerationConfig
    from runtime import LoadedModel,atomic_json,read_inputs,code_identity,frozen_digest,output_lock,gpu_lock,Scorer,sha256
    from train_core import verify_checkpoint
    from quality import repeated
    import re
    title_only=lambda text:bool(re.fullmatch(r'\s*(?:#{1,6}[ \t]*(?:response|answer)[ \t]*:[ \t\r\n]*)+',text,re.I))
    write=lambda name,value:atomic_json(args.output/name,value)
    started=time.monotonic()
    try:
        with output_lock(args.output),gpu_lock():
            if code_identity()!=CODE:raise ValueError('Frozen runtime changed')
            failure=read(args.failed/'failure.json');manifest=read(args.failed/'manifest.json');baseline_summary=read(args.baseline/'summary.json')
            if (failure.get('type')!='ValueError' or not failure.get('error','').startswith('Original trajectory controls differ:')
                    or "'learning_rates_exact': False" not in failure['error']):raise ValueError('Not the recognized LR comparison rejection')
            expected={'status':'COMPLETE','mode':'all','model':args.model,'steps':1000,'original':str(args.original)}
            if any(baseline_summary.get(k)!=v for k,v in expected.items()) or baseline_summary['replay']['step100_adapter_exact'] is not True:
                raise ValueError('No exact matching baseline')
            baseline_replay=baseline_summary['replay']
            if (not all(baseline_replay.get(k) is True for k in ['sample_order_exact','learning_rates_exact'])
                    or baseline_replay.get('max_loss_difference',float('inf'))>1e-6
                    or baseline_replay.get('max_base_ce_difference',float('inf'))>1e-6):raise ValueError('Baseline trajectory did not replay')
            if manifest['original']!=str(args.original) or manifest['code_id']!=CODE or manifest['steps']!=1000 or manifest['mode']=='all':
                raise ValueError('Wrong failed diagnostic identity')
            old_cfg=read(args.baseline/'resolved_config.json');cfg=read(args.failed/'resolved_config.json')
            keys=['learning_rate','num_train_epochs','per_device_train_batch_size','gradient_accumulation_steps','lr_scheduler_type',
                  'warmup_ratio','optim','weight_decay','seed','fp16','bf16','adam_beta1','adam_beta2','adam_epsilon','max_grad_norm']
            if any(cfg[k]!=old_cfg[k] for k in keys) or cfg['learning_rate']!=5e-5 or cfg['warmup_ratio']!=.1:raise ValueError('Actual scheduler recipe changed')
            checkpoint=args.failed/'trainer/checkpoint-1000';history=verify_checkpoint(checkpoint)
            original=read(args.original/'adaptation.json')[:1000]
            if len(history)!=1000 or [r['source_index'] for r in history]!=[r['source_index'] for r in original]:raise ValueError('Sample order changed')
            if any(a['base_ce']!=b['base_ce'] or a['weight']!=b['weight'] or a['selected']!=b['selected'] for a,b in zip(history,original)):
                raise ValueError('Frozen-base selection/weights changed')
            opt=torch.load(checkpoint/'optimizer.pt',map_location='cpu',weights_only=True)
            if not all(bool(torch.isfinite(v).all()) for s in opt['state'].values() for v in s.values() if torch.is_tensor(v)):
                raise ValueError('Nonfinite preserved optimizer')
            raw_counts={float(s['step']) for s in opt['state'].values()}
            if any(not v.is_integer() for v in raw_counts):raise ValueError('Noninteger optimizer step counter')
            counts={int(v) for v in raw_counts}
            if len(counts)!=1 or min(counts)<2:raise ValueError('Invalid optimizer update counts')
            updates=min(counts);scheduler=torch.load(checkpoint/'scheduler.pt',map_location='cpu',weights_only=True)
            lr_proof=verify_lr_trace(history,updates,scheduler['last_epoch'],5e-5,500,5000)
            drift={'per_visit_learning_rates_exact':all(a['lr']==b['lr'] for a,b in zip(history,original)),
                   'max_absolute_lr_difference':max(abs(a['lr']-b['lr']) for a,b in zip(history,original)),
                   'mismatched_visits':sum(a['lr']!=b['lr'] for a,b in zip(history,original)),
                   'baseline_optimizer':baseline_summary['optimizer'],'control_actual_schedule':lr_proof,
                   'interpretation':'Loss masks also change gradient/clipping/AMP behavior; semantic effects are not isolated from these effects.'}
            write('schedule_proof.json',drift)
            rows,prepared=read_inputs(Path('/data/chuwenjing/tta_offline/tlm_inputs_v1/dolly_5k'),
                                     '5f4b68a265000ebb725bd05e5395989a18c4e52e588b8422eee9ca20f341e354')
            loaded=LoadedModel(args.model,'dolly_5k',42,initialization='gaussian')
            if loaded.contract['frozen_sha256']!=BASES[args.model]:raise ValueError('Noncanonical base')
            encoded=loaded.tokens(rows);saved=read(args.original/'input_tokens.json')['records']
            if any(info['token_sha256']!=saved[i]['token_sha256'] for i,(_,info) in enumerate(encoded)):raise ValueError('Input tokens differ')
            loaded.model=Accelerator(mixed_precision='fp16').prepare_model(loaded.model,evaluation_mode=True)
            loaded.model.gradient_checkpointing_disable();loaded.model.eval()
            def load(path):
                adapter=load_file(str(path))
                if not all(bool(torch.isfinite(v).all()) for v in adapter.values()):raise ValueError('Nonfinite adapter')
                set_peft_model_state_dict(loaded.model,adapter);actual=get_peft_model_state_dict(loaded.model)
                if actual.keys()!=adapter.keys() or not all(torch.equal(actual[k].detach().cpu(),adapter[k]) for k in adapter):raise ValueError('Adapter content differs')
            load(args.baseline/'trainer/checkpoint-1000/adapter_model.safetensors')
            baseline_answers=read(args.baseline/'answers.json');replay=[]
            for row in baseline_answers[:2]:
                result=loaded.generate(encoded[row['source_index']][0])
                if result['generated_ids']!=row['generated_ids'] or result['predict']!=row['predict']:raise ValueError('Baseline generation replay failed')
                replay.append({'source_index':row['source_index'],'generated_ids_exact':True})
            load(checkpoint/'adapter_model.safetensors')
            selected=read(args.baseline/'probe_indices.json')
            if len(selected)!=100 or len(set(selected))!=100 or any(type(i) is not int or not 0<=i<5000 for i in selected):raise ValueError('Probe indexes differ')
            cfg_generation=GenerationConfig(max_new_tokens=1,do_sample=False,num_beams=1,repetition_penalty=1.,eos_token_id=loaded.stop_ids,pad_token_id=loaded.pad_id)
            probes=[]
            for i in selected:
                ids=torch.tensor([encoded[i][0]],device=loaded.device)
                with torch.no_grad():result=loaded.model.generate(input_ids=ids,attention_mask=torch.ones_like(ids),generation_config=cfg_generation,use_cache=True,return_dict_in_generate=True,output_scores=True)
                token=int(result.sequences[0,-1]);prob=result.scores[0][0].float().softmax(-1)
                probes.append({'source_index':i,'top1_id':token,'starts_hash':loaded.tokenizer.decode([token]).lstrip().startswith('###'),
                               'stop_is_argmax':token in loaded.stop_ids,'p_stop':float(prob[loaded.stop_ids].sum())})
            write('first_tokens.json',probes)
            ref_path=Path('/data/chuwenjing/tta_offline/tlm_inputs_v1/dolly_5k/references.jsonl')
            if sha256(ref_path)!=prepared['files']['references.jsonl']:raise ValueError('Scoring references changed')
            refs=[json.loads(line) for line in ref_path.read_text().splitlines()];reuse=read(args.original/'source_reuse.json');source_path=Path(reuse['path'])
            if sha256(source_path)!=reuse['sha256']:raise ValueError('Source changed')
            source=[json.loads(line) for line in source_path.read_text().splitlines()]
            if any(len(v)!=5000 or any(r['source_index']!=i for i,r in enumerate(v)) for v in [refs,source]):raise ValueError('Scoring indexes changed')
            scorer=Scorer('rougeLsum');answers=[]
            for old in baseline_answers:
                i=old['source_index'];result=loaded.generate(encoded[i][0]);answers.append({'source_index':i,**result,
                    'score':scorer.score(result['predict'],refs[i]['output']),'source_score':source[i]['score'],
                    'title_only':title_only(result['predict']),'repetition':repeated(result['predict'])})
                if scorer.score(source[i]['predict'],refs[i]['output'])!=source[i]['score']:raise ValueError('Source score differs')
                write('answers.json',answers)
            if frozen_digest(loaded.model)!=BASES[args.model]:raise ValueError('Base changed')
            write('summary.json',{'status':'COMPLETE','model':args.model,'mode':manifest['mode'],'steps':1000,'training_performed':False,
                'original_rejected_diagnostic':str(args.failed),'baseline_replay':replay,'schedule_proof':drift,
                'next_token_hash_count':sum(r['starts_hash'] for r in probes),'next_token_stop_argmax_count':sum(r['stop_is_argmax'] for r in probes),
                'probe_n':100,'answer_n':len(answers),'answer_empty':sum(not r['predict'].strip() for r in answers),
                'answer_title_only':sum(r['title_only'] for r in answers),'answer_repetition':sum(r['repetition'] for r in answers),
                'small_set_mean_score':sum(r['score'] for r in answers)/len(answers),'not_a_full_score':True,
                'original_failure_record_preserved':True,'seconds':time.monotonic()-started})
    except BaseException as error:
        write('failure.json',{'type':type(error).__name__,'error':str(error),'seconds':time.monotonic()-started});raise


if __name__=='__main__':main()
