"""Isolated 200-step replay and 2x2 formatting-target mechanism controls."""
import argparse
import collections
import contextlib
import json
import math
from pathlib import Path
import random
import sys
import time

from targets import MODES, target_groups, loss_mask, title_only

BASES={'llama3b':'c80d597af8e0f5bb8e99a6545680edb580ed2fe59c9cba4af0f39b568c15be77',
       'qwen3b':'644c521c2e650bb46c5bfcedea082bbc3a02b9a5c0f67297a44633cb035384bc'}
CODE='b6da067b55f5c65241ae6fbd8c706b88523f64b27dd9568ad5661255b89d7999'


def read(p):return json.loads(Path(p).read_text())


def require_baseline(record,model,steps,original):
    expected={'status':'COMPLETE','mode':'all','model':model,'steps':steps,'original':str(original)}
    if any(record.get(k)!=v for k,v in expected.items()):raise ValueError('No matching successful baseline')
    replay=record.get('replay',{})
    if (not all(replay.get(k) is True for k in ['sample_order_exact','learning_rates_exact','step100_adapter_exact'])
            or replay.get('max_loss_difference',math.inf)>1e-6
            or replay.get('max_base_ce_difference',math.inf)>1e-6):raise ValueError('Baseline trajectory replay failed')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime',type=Path,required=True)
    parser.add_argument('--original',type=Path,required=True)
    parser.add_argument('--model',choices=BASES,required=True)
    parser.add_argument('--mode',choices=MODES,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--steps',type=int,default=200)
    parser.add_argument('--baseline',type=Path)
    args=parser.parse_args()
    if not 100<=args.steps<=1000:raise ValueError('Diagnostic step limit must be 100..1000')
    if args.mode!='all':
        if args.baseline is None:raise ValueError('Masked controls require a completed exact baseline')
        require_baseline(read(args.baseline/'summary.json'),args.model,args.steps,str(args.original))
    if args.output.resolve().is_relative_to(args.original.resolve()):raise ValueError('Never write into an original run')
    args.output.mkdir(parents=True,exist_ok=False)
    sys.path.insert(0,str(args.runtime))
    import torch
    from datasets import Dataset
    from safetensors.torch import load_file
    from transformers import TrainerCallback,GenerationConfig
    from runtime import LoadedModel,atomic_json,read_inputs,code_identity,frozen_digest,output_lock,gpu_lock,Scorer,sha256
    from train_core import TLMTrainer,InputCollator
    from run import training_arguments,optimizer_statistics
    from tlm_loss import model_input_cross_entropy,weighted_input_loss,sample_weights
    from masked_ce import masked_input_cross_entropy
    from quality import repeated
    if code_identity()!=CODE:raise ValueError('Frozen training runtime changed')
    write=lambda name,value:atomic_json(args.output/name,value)

    class StopAtLimit(TrainerCallback):
        def on_step_end(self,configuration,state,control,**kwargs):
            if state.global_step>=args.steps:control.should_training_stop=True
            return control

    class MaskedTrainer(TLMTrainer):
        def compute_loss(self,model,inputs,return_outputs=False,num_items_in_batch=None):
            ids,attn=inputs['input_ids'],inputs['attention_mask'];index=int(inputs['source_index'][0])
            key=tuple(ids[0].detach().cpu().tolist())
            cached=self.base_scores.get(index)
            if cached is None:
                model.eval()
                with torch.no_grad(),model.disable_adapter(),self.accelerator.autocast():
                    base=model_input_cross_entropy(model,ids,attn,self.chunk_size)
                self.base_scores[index]=(key,float(base[0]));model.train()
            else:
                if cached[0]!=key:raise ValueError('Input token identity changed')
                base=torch.tensor([cached[1]],device=ids.device,dtype=torch.float32)
            active=torch.tensor([masks[index]],device=ids.device,dtype=torch.bool)
            with self.accelerator.autocast():current=masked_input_cross_entropy(model,ids,attn,active,self.chunk_size)
            loss=weighted_input_loss(current,base,self.threshold,self.lamb)
            selected,weights=sample_weights(base,self.threshold,self.lamb)
            self.history.append({'visit':len(self.history)+1,'source_index':index,'base_ce':float(base[0]),
                'selected':bool(selected[0]),'weight':float(weights[0]),'current_ce':float(current[0].detach()),
                'loss':float(loss.detach()),'lr':float(self.optimizer.param_groups[0]['lr'])})
            return (loss,{'loss':loss}) if return_outputs else loss

    started=time.monotonic()
    try:
        with output_lock(args.output),gpu_lock():
            original=read(args.original/'run_manifest.json')
            expected={'code_id':CODE,'model':args.model,'dataset':'dolly_5k','lora_init':'gaussian','count':5000,'seed':42,'protocol':'A'}
            if any(original['config'].get(k)!=v for k,v in expected.items()):raise ValueError('Wrong original run')
            rows,prepared=read_inputs(Path('/data/chuwenjing/tta_offline/tlm_inputs_v1/dolly_5k'),
                                     '5f4b68a265000ebb725bd05e5395989a18c4e52e588b8422eee9ca20f341e354')
            loaded=LoadedModel(args.model,'dolly_5k',42,initialization='gaussian')
            if loaded.contract['frozen_sha256']!=BASES[args.model]:raise ValueError('Noncanonical frozen base')
            encoded=loaded.tokens(rows);saved=read(args.original/'input_tokens.json')['records']
            if any(info['token_sha256']!=saved[i]['token_sha256'] for i,(_,info) in enumerate(encoded)):
                raise ValueError('Diagnostic prompts differ from original')
            masks=[];group_records=[]
            for i,(ids,info) in enumerate(encoded):
                text=loaded.tokenizer.decode(ids,skip_special_tokens=False,clean_up_tokenization_spaces=False)
                tokenized=loaded.tokenizer(text,add_special_tokens=False,return_offsets_mapping=True)
                if tokenized['input_ids']!=ids:raise ValueError('Token offsets fail exact decode/re-encode: '+str(i))
                offsets=tokenized['offset_mapping']
                end_index=max(j for j,t in enumerate(ids) if t==loaded.tokenizer.eos_token_id)
                groups=target_groups(text,offsets,offsets[end_index][0],info['input_truncated'])
                mask=loss_mask(groups,args.mode)
                if not any(mask[1:]):raise ValueError('No causal body targets: '+str(i))
                masks.append(mask)
                group_records.append({'source_index':i,'tokens':len(ids),'counts':dict(collections.Counter(groups[1:])),
                    'retained_targets':sum(mask[1:]),'normalization_denominator':len(ids)-1,
                    'groups':groups,'token_sha256':info['token_sha256']})
            write('target_positions.json',group_records)
            features=[{'input_ids':ids,'source_index':i} for i,(ids,_) in enumerate(encoded)]
            configuration=training_arguments(args.output,5e-5,42,5000)
            cls=TLMTrainer if args.mode=='all' else MaskedTrainer
            trainer=cls(model=loaded.model,args=configuration,train_dataset=Dataset.from_list(features),
                        data_collator=InputCollator(loaded.pad_id),chunk_size=128,callbacks=[StopAtLimit()])
            write('resolved_config.json',trainer.args.to_dict())
            write('manifest.json',{'status':'TRAINING','diagnostic_only':True,'mode':args.mode,'steps':args.steps,
                'planned_schedule_steps':5000,'warmup_steps':500,'normalization':'original_full_input_target_count',
                'selection':'unchanged full-input frozen-base CE','original':str(args.original),'code_id':CODE,
                'canonical_base':loaded.contract['frozen_sha256'],'generation':'unchanged Source-matched greedy512'})
            trainer.train()
            stats=optimizer_statistics(trainer,args.steps)
            if len(trainer.history)!=args.steps or trainer.state.max_steps!=5000:raise ValueError('Short-run schedule or step count changed')
            expected=read(args.original/'adaptation.json')[:args.steps]
            order_ok=[x['source_index'] for x in expected]==[x['source_index'] for x in trainer.history]
            base_delta=max(abs(a['base_ce']-b['base_ce']) for a,b in zip(trainer.history,expected))
            weight_delta=max(abs(a['weight']-b['weight']) for a,b in zip(trainer.history,expected))
            lr_exact=all(a['lr']==b['lr'] for a,b in zip(trainer.history,expected))
            replay={'sample_order_exact':order_ok,'max_base_ce_difference':base_delta,'max_weight_difference':weight_delta,'learning_rates_exact':lr_exact}
            if not order_ok or not lr_exact or base_delta>1e-6:raise ValueError('Original trajectory controls differ: '+str(replay))
            if args.mode=='all':
                replay['max_loss_difference']=max(abs(a['loss']-b['loss']) for a,b in zip(trainer.history,expected))
                old=load_file(str(args.original/'trainer/milestones/step-100/adapter_model.safetensors'))
                new=load_file(str(args.output/'trainer/milestones/step-100/adapter_model.safetensors'))
                replay['step100_adapter_exact']=old.keys()==new.keys() and all(torch.equal(old[k],new[k]) for k in old)
                replay['max_step100_adapter_difference']=max(float((old[k]-new[k]).abs().max()) for k in old)
                if not replay['step100_adapter_exact'] or replay['max_loss_difference']>1e-6:
                    write('replay.json',replay)
                    raise ValueError('Original baseline cannot be replayed: '+str(replay))
            write('replay.json',replay);write('adaptation.json',trainer.history)
            loaded.model.save_pretrained(str(args.output/'final_adapter'),safe_serialization=True,save_embedding_layers=False)
            if frozen_digest(loaded.model)!=BASES[args.model]:raise ValueError('Base changed')
            trained_indices={x['source_index'] for x in trainer.history}
            rng=random.Random(314159);pool=[i for i in range(20,5000) if i not in trained_indices]
            heldout=sorted(rng.sample(pool,80));selected=list(range(20))+heldout;write('probe_indices.json',selected)
            loaded.model.gradient_checkpointing_disable();loaded.model.eval()
            next_tokens=[];base_next_tokens=[]
            cfg=GenerationConfig(max_new_tokens=1,do_sample=False,num_beams=1,repetition_penalty=1.,
                                 eos_token_id=loaded.stop_ids,pad_token_id=loaded.pad_id)
            for values,disabled in [(base_next_tokens,True),(next_tokens,False)]:
                with loaded.model.disable_adapter() if disabled else contextlib.nullcontext():
                    for i in selected:
                        ids=encoded[i][0];tensor=torch.tensor([ids],device=loaded.device)
                        with torch.no_grad():
                            result=loaded.model.generate(input_ids=tensor,attention_mask=torch.ones_like(tensor),generation_config=cfg,
                                                        use_cache=True,return_dict_in_generate=True,output_scores=True)
                        token=int(result.sequences[0,-1]);prob=result.scores[0][0].float().softmax(-1)
                        values.append({'source_index':i,'top1_id':token,'top1_text':loaded.tokenizer.decode([token]),
                                       'stop_is_argmax':token in loaded.stop_ids,'p_stop':float(prob[loaded.stop_ids].sum()),
                                       'starts_hash':loaded.tokenizer.decode([token]).lstrip().startswith('###')})
            write('first_tokens.json',next_tokens);write('base_first_tokens.json',base_next_tokens)
            segments=[]
            for disabled in [True,False]:
                with loaded.model.disable_adapter() if disabled else contextlib.nullcontext():
                    for i in range(20):
                        ids=encoded[i][0];tensor=torch.tensor([ids],device=loaded.device)
                        base=loaded.model.get_base_model()
                        with torch.no_grad(),trainer.accelerator.autocast():
                            hidden=base.model(input_ids=tensor,attention_mask=torch.ones_like(tensor),use_cache=False,return_dict=True).last_hidden_state
                            logits=base.get_output_embeddings()(hidden)
                        losses=[]
                        for start in range(0,len(ids)-1,128):
                            stop=min(start+128,len(ids)-1)
                            tile=torch.nn.functional.cross_entropy(logits[0,start:stop].float(),tensor[0,start+1:stop+1],reduction='none')
                            losses.extend(tile.detach().cpu().tolist())
                            del tile
                        groups=group_records[i]['groups'][1:]
                        groups_report={g:{'count':sum(x==g for x in groups),
                            'sum_ce':sum(v for v,x in zip(losses,groups) if x==g)} for g in ['role','wrapper','body']}
                        for value in groups_report.values():value['mean_ce']=value['sum_ce']/value['count'] if value['count'] else None
                        segments.append({'source_index':i,'adapter_disabled':disabled,'full_ce':sum(losses)/len(losses),
                                         'groups':groups_report,'generation_position_is_not_an_input_loss_target':True})
                        del logits,hidden
            write('segmented_ce.json',segments)
            ref_path=Path('/data/chuwenjing/tta_offline/tlm_inputs_v1/dolly_5k/references.jsonl')
            if sha256(ref_path)!=prepared['files']['references.jsonl']:raise ValueError('Scoring reference hash changed')
            references=[json.loads(x) for x in ref_path.read_text().splitlines()]
            if len(references)!=5000 or any(r['source_index']!=i for i,r in enumerate(references)):raise ValueError('Reference index identity changed')
            reuse=read(args.original/'source_reuse.json');source_path=Path(reuse['path'])
            if sha256(source_path)!=reuse['sha256']:raise ValueError('Paired Source changed')
            source=[json.loads(x) for x in source_path.read_text().splitlines()]
            if len(source)!=5000 or any(r['source_index']!=i for i,r in enumerate(source)):raise ValueError('Source indexes changed')
            scorer=Scorer('rougeLsum');answers=[]
            # Small held-out set checks behavior only; it is not a replacement for full evaluation.
            for i in heldout[:8]:
                result=loaded.generate(encoded[i][0]);row={'source_index':i,**result,
                    'score':scorer.score(result['predict'],references[i]['output']),'title_only':title_only(result['predict']),
                    'repetition':repeated(result['predict']),'source_predict':source[i]['predict'],'source_score':source[i]['score']}
                if abs(scorer.score(source[i]['predict'],references[i]['output'])-source[i]['score'])>1e-12:raise ValueError('Paired Source score differs')
                answers.append(row);write('answers.json',answers)
            summary={'status':'COMPLETE','diagnostic_only':True,'mode':args.mode,'steps':args.steps,'model':args.model,'original':str(args.original),
                'replay':replay,'optimizer':stats,'probe_n':100,'next_token_hash_count':sum(r['starts_hash'] for r in next_tokens),
                'next_token_stop_argmax_count':sum(r['stop_is_argmax'] for r in next_tokens),
                'mean_p_stop':sum(r['p_stop'] for r in next_tokens)/100,'answer_n':len(answers),
                'base_next_token_hash_count':sum(r['starts_hash'] for r in base_next_tokens),
                'base_next_token_stop_argmax_count':sum(r['stop_is_argmax'] for r in base_next_tokens),
                'answer_empty':sum(not r['predict'].strip() for r in answers),'answer_title_only':sum(r['title_only'] for r in answers),
                'answer_repetition':sum(r['repetition'] for r in answers),
                'dominant_answer_count':max(collections.Counter(' '.join(r['predict'].split()) for r in answers).values()),
                'source_small_set_mean_score':sum(r['source_score'] for r in answers)/len(answers),
                'small_set_mean_score':sum(r['score'] for r in answers)/len(answers),'seconds':time.monotonic()-started,
                'not_a_full_score':True,'original_runs_unchanged':True}
            write('summary.json',summary);print(json.dumps(summary),flush=True)
    except BaseException as error:
        write('failure.json',{'type':type(error).__name__,'error':str(error),'seconds':time.monotonic()-started})
        raise


if __name__=='__main__':main()
