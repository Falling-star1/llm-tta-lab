"""Inspect saved GSM adapters without any training or original-run mutation."""
import argparse
import contextlib
import json
from pathlib import Path
import sys
import time

from support import probe_indices,reasoning_stub_only,milestone_path

CODE='b6da067b55f5c65241ae6fbd8c706b88523f64b27dd9568ad5661255b89d7999'
BASE='c80d597af8e0f5bb8e99a6545680edb580ed2fe59c9cba4af0f39b568c15be77'


def read(p):return json.loads(Path(p).read_text())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runtime',type=Path,required=True);p.add_argument('--original',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.resolve().is_relative_to(args.original.resolve()):raise ValueError('Probe output must be separate')
    args.output.mkdir(parents=True,exist_ok=False);sys.path.insert(0,str(args.runtime))
    import torch
    from accelerate import Accelerator
    from peft import set_peft_model_state_dict,get_peft_model_state_dict
    from safetensors.torch import load_file
    from transformers import GenerationConfig
    from runtime import LoadedModel,atomic_json,gpu_lock,output_lock,code_identity,read_inputs,frozen_digest,sha256,Scorer
    if code_identity()!=CODE:raise ValueError('Frozen runtime changed')
    write=lambda name,value:atomic_json(args.output/name,value)
    start=time.monotonic()
    try:
        with output_lock(args.output),gpu_lock():
            original=read(args.original/'run_manifest.json');training=read(args.original/'training_complete.json')
            expected={'code_id':CODE,'model':'llama3b','dataset':'gsm8k_5k','lora_init':'gaussian','count':5000,'seed':42}
            if any(original['config'].get(k)!=v for k,v in expected.items()) or training['n']!=5000:raise ValueError('Wrong completed training')
            if sha256(args.original/'tlm.jsonl')!=read(args.original/'metrics.json')['predictions_sha256']:raise ValueError('Final predictions changed')
            saved=[json.loads(line) for line in (args.original/'tlm.jsonl').read_text().splitlines()]
            if len(saved)!=5000 or any(r['source_index']!=i for i,r in enumerate(saved)):raise ValueError('Final predictions incomplete')
            rows,prepared=read_inputs(Path('/data/chuwenjing/tta_offline/tlm_inputs_v1/gsm8k_5k'),
                                     'd2843770014f663dd3df3ab796816f16e1ff384622bfd8c83ea056315602c5ba')
            loaded=LoadedModel('llama3b','gsm8k_5k',42,initialization='gaussian')
            if loaded.contract['frozen_sha256']!=BASE:raise ValueError('Noncanonical base')
            encoded=loaded.tokens(rows);tokens=read(args.original/'input_tokens.json')['records']
            if any(info['token_sha256']!=tokens[i]['token_sha256'] for i,(_,info) in enumerate(encoded)):raise ValueError('Inputs differ')
            loaded.model=Accelerator(mixed_precision='fp16').prepare_model(loaded.model,evaluation_mode=True)
            loaded.model.gradient_checkpointing_disable();loaded.model.eval()
            def load(step):
                path=milestone_path(args.original,step);adapter=load_file(str(path))
                if not all(bool(torch.isfinite(v).all()) for v in adapter.values()):raise ValueError('Nonfinite adapter')
                set_peft_model_state_dict(loaded.model,adapter)
                actual=get_peft_model_state_dict(loaded.model)
                if actual.keys()!=adapter.keys() or not all(torch.equal(actual[k].detach().cpu(),adapter[k]) for k in adapter):
                    raise ValueError('Loaded adapter differs')
                return sha256(path)
            final_sha=load(5000);replay=[]
            for i in [0,99]:
                result=loaded.generate(encoded[i][0])
                if result['generated_ids']!=saved[i]['generated_ids'] or result['predict']!=saved[i]['predict']:
                    raise ValueError('Saved final predictions cannot be exactly replayed')
                replay.append({'source_index':i,'generated_ids_exact':True})
            write('replay.json',{'final_adapter_sha256':final_sha,'replay':replay,'training_performed':False})
            ref_path=Path('/data/chuwenjing/tta_offline/tlm_inputs_v1/gsm8k_5k/references.jsonl')
            if sha256(ref_path)!=prepared['files']['references.jsonl']:raise ValueError('References changed')
            refs=[json.loads(line) for line in ref_path.read_text().splitlines()]
            reuse=read(args.original/'source_reuse.json');source_path=Path(reuse['path'])
            if sha256(source_path)!=reuse['sha256']:raise ValueError('Source changed')
            source=[json.loads(line) for line in source_path.read_text().splitlines()]
            if any(len(v)!=5000 or any(r['source_index']!=i for i,r in enumerate(v)) for v in [refs,source]):raise ValueError('Scoring index mismatch')
            selected=probe_indices();scorer=Scorer('exact_match');summaries=[]
            forced=loaded.tokenizer.encode("Let's think step by step.",add_special_tokens=False)
            cfg=GenerationConfig(max_new_tokens=1,do_sample=False,num_beams=1,repetition_penalty=1.,
                                 eos_token_id=loaded.stop_ids,pad_token_id=loaded.pad_id)
            for stage in [0,1000,2500,5000]:
                digest=None if stage==0 else load(stage)
                values=[];answers=[]
                write('status.json',{'status':'RUNNING','stage':stage,'training_performed':False})
                with loaded.model.disable_adapter() if stage==0 else contextlib.nullcontext():
                    for i in selected:
                        ids=encoded[i][0];entry={'source_index':i}
                        for label,prefix in [('first',ids),('after_forced_reasoning_stub',ids+forced)]:
                            tensor=torch.tensor([prefix],device=loaded.device)
                            with torch.no_grad():
                                result=loaded.model.generate(input_ids=tensor,attention_mask=torch.ones_like(tensor),generation_config=cfg,
                                                            use_cache=True,return_dict_in_generate=True,output_scores=True)
                            token=int(result.sequences[0,-1]);probs=result.scores[0][0].float().softmax(-1)
                            entry[label]={'top1_id':token,'top1_text':loaded.tokenizer.decode([token]),
                                          'stop_is_argmax':token in loaded.stop_ids,'p_stop':float(probs[loaded.stop_ids].sum())}
                        values.append(entry)
                    for i in selected[20:24]:
                        result=loaded.generate(encoded[i][0]);score=scorer.score(result['predict'],refs[i]['output'])
                        if scorer.score(source[i]['predict'],refs[i]['output'])!=source[i]['score']:raise ValueError('Source score differs')
                        answers.append({'source_index':i,**result,'score':score,'reasoning_stub_only':reasoning_stub_only(result['predict']),
                                        'source_predict':source[i]['predict'],'source_score':source[i]['score']})
                write('stage_'+str(stage)+'_probes.json',values);write('stage_'+str(stage)+'_answers.json',answers)
                summary={'stage':stage,'adapter_sha256':digest,'probe_n':100,
                    'direct_stop_argmax_n':sum(r['first']['stop_is_argmax'] for r in values),
                    'after_forced_stub_stop_argmax_n':sum(r['after_forced_reasoning_stub']['stop_is_argmax'] for r in values),
                    'mean_p_stop_after_forced_stub':sum(r['after_forced_reasoning_stub']['p_stop'] for r in values)/100,
                    'answer_n':4,'answer_stub_only_n':sum(r['reasoning_stub_only'] for r in answers),
                    'small_set_em':sum(r['score'] for r in answers)/4,'small_set_source_em':sum(r['source_score'] for r in answers)/4,
                    'not_a_full_score':True}
                summaries.append(summary);write('milestones.json',summaries)
            if frozen_digest(loaded.model)!=BASE:raise ValueError('Base changed during read-only probe')
            write('summary.json',{'status':'COMPLETE','training_performed':False,'optimizer_created':False,'original_runs_modified':False,
                                  'replay':replay,'milestones':summaries,'seconds':time.monotonic()-start})
            write('status.json',{'status':'COMPLETE','training_performed':False})
    except BaseException as error:
        write('failure.json',{'type':type(error).__name__,'error':str(error),'seconds':time.monotonic()-start});raise


if __name__=='__main__':main()
