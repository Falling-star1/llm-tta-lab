"""Fixed-recipe GSM transition controls against the existing step-2500 baseline."""
import argparse
import collections
import contextlib
import json
from pathlib import Path
import sys
import time
from masking import MODES,HINT,transition_mask

CODE='b6da067b55f5c65241ae6fbd8c706b88523f64b27dd9568ad5661255b89d7999'
BASE='c80d597af8e0f5bb8e99a6545680edb580ed2fe59c9cba4af0f39b568c15be77'
MASK_SHA='c2ba55874311374f8c923bd3545a73b29a2043efbd46b2b449fba1c783934109'
SUPPORT_SHA='38ebe94c14f05127030366c399c90a22e845fa93d465aab34fa190bf3c925ca2'
BASELINE_SHA='3d0fc755e80442d9bce47647eee0fe1c3646784dac499ef88f98f93586b84162'


def read(p):return json.loads(Path(p).read_text())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['runtime','original','mask_code','support_code','output']:p.add_argument('--'+name.replace('_','-'),type=Path,required=True)
    p.add_argument('--mode',choices=MODES,required=True);args=p.parse_args()
    if args.output.resolve().is_relative_to(args.original.resolve()):raise ValueError('Output must be separate')
    args.output.mkdir(parents=True,exist_ok=False)
    sys.path.insert(0,str(args.runtime))
    import torch
    from accelerate import Accelerator
    from datasets import Dataset
    from peft import set_peft_model_state_dict,get_peft_model_state_dict
    from safetensors.torch import load_file
    from transformers import TrainerCallback,GenerationConfig
    from runtime import LoadedModel,atomic_json,read_inputs,code_identity,frozen_digest,output_lock,gpu_lock,Scorer,sha256
    from train_core import TLMTrainer,InputCollator
    from run import training_arguments,optimizer_statistics
    from tlm_loss import model_input_cross_entropy,weighted_input_loss,sample_weights
    if code_identity()!=CODE or sha256(args.mask_code/'masked_ce.py')!=MASK_SHA or sha256(args.support_code/'support.py')!=SUPPORT_SHA:
        raise ValueError('Frozen dependency identity differs')
    sys.path.insert(0,str(args.mask_code));from masked_ce import masked_input_cross_entropy
    sys.path.insert(0,str(args.support_code));from support import probe_indices,reasoning_stub_only,milestone_path
    write=lambda name,value:atomic_json(args.output/name,value)
    started=time.monotonic()

    class StopAtLimit(TrainerCallback):
        def on_step_end(self,configuration,state,control,**kwargs):
            if state.global_step>=2500:control.should_training_stop=True
            return control

    class ControlledTrainer(TLMTrainer):
        def compute_loss(self,model,inputs,return_outputs=False,num_items_in_batch=None):
            ids,attn=inputs['input_ids'],inputs['attention_mask'];index=int(inputs['source_index'][0])
            if not torch.equal(inputs['labels'],ids.masked_fill(attn==0,-100)):raise ValueError('Non-input training labels')
            model.eval()
            with torch.no_grad(),model.disable_adapter(),self.accelerator.autocast():base=model_input_cross_entropy(model,ids,attn,self.chunk_size)
            model.train();active=torch.tensor([masks[index]],device=ids.device,dtype=torch.bool)
            with self.accelerator.autocast():current=masked_input_cross_entropy(model,ids,attn,active,self.chunk_size)
            loss=weighted_input_loss(current,base,self.threshold,self.lamb);chosen,weight=sample_weights(base,self.threshold,self.lamb)
            self.history.append({'visit':len(self.history)+1,'source_index':index,'base_ce':float(base[0]),'selected':bool(chosen[0]),
                'weight':float(weight[0]),'current_ce':float(current[0].detach()),'loss':float(loss.detach()),'lr':float(self.optimizer.param_groups[0]['lr'])})
            return (loss,{'loss':loss}) if return_outputs else loss

    try:
        with output_lock(args.output),gpu_lock():
            manifest=read(args.original/'run_manifest.json');expected={'code_id':CODE,'model':'llama3b','dataset':'gsm8k_5k','count':5000,'seed':42,'lora_init':'gaussian'}
            if any(manifest['config'].get(k)!=v for k,v in expected.items()) or read(args.original/'training_complete.json')['n']!=5000:
                raise ValueError('Wrong completed original training')
            baseline_path=milestone_path(args.original,2500)
            if sha256(baseline_path)!=BASELINE_SHA:raise ValueError('Original baseline checkpoint changed')
            if sha256(args.original/'tlm.jsonl')!=read(args.original/'metrics.json')['predictions_sha256']:raise ValueError('Original evaluation changed')
            predictions=[json.loads(line) for line in (args.original/'tlm.jsonl').read_text().splitlines()]
            rows,prepared=read_inputs(Path('/data/chuwenjing/tta_offline/tlm_inputs_v1/gsm8k_5k'),
                                     'd2843770014f663dd3df3ab796816f16e1ff384622bfd8c83ea056315602c5ba')
            loaded=LoadedModel('llama3b','gsm8k_5k',42,initialization='gaussian')
            if loaded.contract['frozen_sha256']!=BASE:raise ValueError('Noncanonical base')
            initial={k:v.detach().cpu().clone() for k,v in get_peft_model_state_dict(loaded.model).items()}
            def load(adapter):
                if not all(bool(torch.isfinite(v).all()) for v in adapter.values()):raise ValueError('Nonfinite adapter')
                set_peft_model_state_dict(loaded.model,adapter)
                actual=get_peft_model_state_dict(loaded.model)
                if actual.keys()!=adapter.keys() or not all(torch.equal(actual[k].detach().cpu(),adapter[k]) for k in adapter):raise ValueError('Adapter load mismatch')
            encoded=loaded.tokens(rows);old=read(args.original/'input_tokens.json')['records'];masks=[];positions=[]
            for i,(ids,info) in enumerate(encoded):
                if info['token_sha256']!=old[i]['token_sha256'] or info['input_truncated']:raise ValueError('GSM input differs or is truncated')
                text=loaded.tokenizer.decode(ids,skip_special_tokens=False,clean_up_tokenization_spaces=False)
                tok=loaded.tokenizer(text,add_special_tokens=False,return_offsets_mapping=True)
                if tok['input_ids']!=ids:raise ValueError('Token offsets not exact')
                eot=max(j for j,v in enumerate(ids) if v==loaded.tokenizer.eos_token_id)
                mask=transition_mask(text,tok['offset_mapping'],eot,args.mode);masks.append(mask)
                positions.append({'source_index':i,'user_eot_position':eot,'removed_positions':[j for j,v in enumerate(mask) if not v],
                                  'original_denominator':len(ids)-1,'token_sha256':info['token_sha256']})
            write('target_positions.json',positions)
            replay_accelerator=Accelerator(mixed_precision='fp16')
            loaded.model=replay_accelerator.prepare_model(loaded.model,evaluation_mode=True)
            load(load_file(str(milestone_path(args.original,5000))));replay=[]
            for i in [0,99]:
                result=loaded.generate(encoded[i][0])
                if result['generated_ids']!=predictions[i]['generated_ids'] or result['predict']!=predictions[i]['predict']:raise ValueError('Original final generation replay failed')
                replay.append({'source_index':i,'generated_ids_exact':True})
            # Use a fresh model for training so the verified inference wrapper
            # cannot be nested into the new Trainer's preparation.
            loaded.model=None;replay_accelerator.free_memory();del replay_accelerator
            import gc
            gc.collect();torch.cuda.empty_cache()
            loaded=LoadedModel('llama3b','gsm8k_5k',42,initialization='gaussian')
            fresh=get_peft_model_state_dict(loaded.model)
            if fresh.keys()!=initial.keys() or not all(torch.equal(fresh[k].detach().cpu(),initial[k]) for k in initial):
                raise ValueError('Fresh Gaussian initial adapter differs')
            if loaded.contract['frozen_sha256']!=BASE:raise ValueError('Fresh training base differs')
            del fresh
            configuration=training_arguments(args.output,1e-6,42,5000);original_cfg=read(args.original/'resolved_config.json')
            keys=['learning_rate','num_train_epochs','per_device_train_batch_size','gradient_accumulation_steps','lr_scheduler_type',
                  'warmup_ratio','optim','weight_decay','seed','fp16','bf16','adam_beta1','adam_beta2','adam_epsilon','max_grad_norm']
            if any(configuration.to_dict()[k]!=original_cfg[k] for k in keys):raise ValueError('Training recipe differs')
            trainer=ControlledTrainer(model=loaded.model,args=configuration,
                train_dataset=Dataset.from_list([{'input_ids':ids,'source_index':i} for i,(ids,_) in enumerate(encoded)]),
                data_collator=InputCollator(loaded.pad_id),chunk_size=128,callbacks=[StopAtLimit()])
            write('manifest.json',{'status':'TRAINING','diagnostic_only':True,'mode':args.mode,'steps':2500,'schedule_steps':5000,
                  'lr':1e-6,'original':str(args.original),'baseline_kind':'original_saved_step2500','baseline_sha256':BASELINE_SHA,
                  'independent_baseline_retrained':False,'fresh_initial_adapter_exact':True,'final_inference_replay':replay})
            trainer.train()
            history=trainer.history;reference=read(args.original/'adaptation.json')[:2500]
            proof={'sample_order_exact':[r['source_index'] for r in history]==[r['source_index'] for r in reference],
                'learning_rates_exact':all(a['lr']==b['lr'] for a,b in zip(history,reference)),
                'max_base_ce_difference':max(abs(a['base_ce']-b['base_ce']) for a,b in zip(history,reference)),
                'max_weight_difference':max(abs(a['weight']-b['weight']) for a,b in zip(history,reference))}
            if len(history)!=2500 or trainer.state.max_steps!=5000 or not proof['sample_order_exact'] or not proof['learning_rates_exact'] or proof['max_base_ce_difference']>1e-6:
                raise ValueError('Original trajectory controls differ: '+str(proof))
            stats=optimizer_statistics(trainer,2500);write('controls.json',proof);write('adaptation.json',history)
            loaded.model.save_pretrained(str(args.output/'final_adapter'),safe_serialization=True,save_embedding_layers=False)
            variant={k:v.detach().cpu().clone() for k,v in get_peft_model_state_dict(loaded.model).items()}
            ref_path=Path('/data/chuwenjing/tta_offline/tlm_inputs_v1/gsm8k_5k/references.jsonl')
            if sha256(ref_path)!=prepared['files']['references.jsonl']:raise ValueError('Scoring references changed')
            refs=[json.loads(line) for line in ref_path.read_text().splitlines()];reuse=read(args.original/'source_reuse.json');source_path=Path(reuse['path'])
            if sha256(source_path)!=reuse['sha256']:raise ValueError('Source changed')
            source=[json.loads(line) for line in source_path.read_text().splitlines()]
            if any(len(v)!=5000 or any(r['source_index']!=i for i,r in enumerate(v)) for v in [refs,source]):raise ValueError('Scoring index identity changed')
            scorer=Scorer('exact_match');selected=probe_indices();forced=loaded.tokenizer.encode(HINT,add_special_tokens=False)
            loaded.model.gradient_checkpointing_disable();loaded.model.eval();reports=[]
            cfg=GenerationConfig(max_new_tokens=1,do_sample=False,num_beams=1,repetition_penalty=1.,eos_token_id=loaded.stop_ids,pad_token_id=loaded.pad_id)
            for label,adapter in [('original_step2500',load_file(str(baseline_path))),('transition_control',variant)]:
                load(adapter);probes=[];answers=[];write('status.json',{'status':'EVALUATING','label':label,'mode':args.mode})
                for i in selected:
                    entry={'source_index':i}
                    for name,prefix in [('first',encoded[i][0]),('after_forced_stub',encoded[i][0]+forced)]:
                        ids=torch.tensor([prefix],device=loaded.device)
                        with torch.no_grad():result=loaded.model.generate(input_ids=ids,attention_mask=torch.ones_like(ids),generation_config=cfg,use_cache=True,return_dict_in_generate=True,output_scores=True)
                        token=int(result.sequences[0,-1]);prob=result.scores[0][0].float().softmax(-1)
                        entry[name]={'stop_is_argmax':token in loaded.stop_ids,'p_stop':float(prob[loaded.stop_ids].sum()),'top1_id':token}
                    probes.append(entry)
                for i in selected[20:28]:
                    result=loaded.generate(encoded[i][0]);answers.append({'source_index':i,**result,'score':scorer.score(result['predict'],refs[i]['output']),
                        'stub_only':reasoning_stub_only(result['predict']),'source_score':source[i]['score']})
                    if scorer.score(source[i]['predict'],refs[i]['output'])!=source[i]['score']:raise ValueError('Source score mismatch')
                write(label+'_probes.json',probes);write(label+'_answers.json',answers)
                reports.append({'label':label,'n':100,'stop_after_forced_stub':sum(r['after_forced_stub']['stop_is_argmax'] for r in probes),
                    'mean_p_stop_after_forced_stub':sum(r['after_forced_stub']['p_stop'] for r in probes)/100,
                    'generated_n':8,'stub_only_n':sum(r['stub_only'] for r in answers),'small_set_em':sum(r['score'] for r in answers)/8,'not_a_full_score':True})
                write('comparisons.json',reports)
            if frozen_digest(loaded.model)!=BASE:raise ValueError('Frozen base changed')
            write('summary.json',{'status':'COMPLETE','mode':args.mode,'diagnostic_only':True,'steps':2500,'lr':1e-6,'controls':proof,
                'optimizer':stats,'final_inference_replay':replay,'comparisons':reports,'original_runs_modified':False,'seconds':time.monotonic()-started})
    except BaseException as error:
        write('failure.json',{'type':type(error).__name__,'error':str(error),'seconds':time.monotonic()-started});raise


if __name__=='__main__':main()
