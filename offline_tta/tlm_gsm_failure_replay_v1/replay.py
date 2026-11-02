"""Inference-only paired replay on fixed actual GSM failure cases."""
import argparse
import json
from pathlib import Path
import sys
import time
from cases import failure_indices,is_stub

CODE='b6da067b55f5c65241ae6fbd8c706b88523f64b27dd9568ad5661255b89d7999'
BASE='c80d597af8e0f5bb8e99a6545680edb580ed2fe59c9cba4af0f39b568c15be77'
BASELINE='3d0fc755e80442d9bce47647eee0fe1c3646784dac499ef88f98f93586b84162'
FINAL='01c9dabc1294cc723ed579680ee65e85eba3db0166c19c2f0b5f04ba3b89ffa4'
MODES=['original_step2500','hint_masked','eot_masked','both_masked']


def read(path):return json.loads(Path(path).read_text())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['runtime','original','control_root','output']:p.add_argument('--'+name.replace('_','-'),type=Path,required=True)
    p.add_argument('--mode',choices=MODES,required=True);p.add_argument('--adapter-sha',required=True)
    args=p.parse_args()
    if any(args.output.resolve().is_relative_to(v.resolve()) for v in [args.original,args.control_root,args.runtime]):
        raise ValueError('Replay output must be separate')
    args.output.mkdir(parents=True,exist_ok=False);sys.path.insert(0,str(args.runtime))
    import torch
    from accelerate import Accelerator
    from peft import get_peft_model_state_dict,set_peft_model_state_dict
    from safetensors.torch import load_file
    from runtime import LoadedModel,code_identity,sha256,read_inputs,atomic_json,output_lock,gpu_lock,frozen_digest,Scorer
    from train_core import verify_checkpoint
    started=time.monotonic();write=lambda name,value:atomic_json(args.output/name,value)
    try:
        with output_lock(args.output),gpu_lock():
            if code_identity()!=CODE:raise ValueError('Frozen runtime changed')
            old=read(args.original/'run_manifest.json');cfg=old['config']
            if any(cfg.get(k)!=v for k,v in {'code_id':CODE,'model':'llama3b','dataset':'gsm8k_5k','seed':42,'count':5000,'lora_init':'gaussian'}.items()):
                raise ValueError('Wrong original experiment')
            if sha256(args.original/'tlm.jsonl')!=read(args.original/'metrics.json')['predictions_sha256']:
                raise ValueError('Original final predictions changed')
            predictions=[json.loads(line) for line in (args.original/'tlm.jsonl').read_text().splitlines()]
            selected=failure_indices(predictions)
            write('selection.json',{'indices':selected,'selection_seed':314159,'population_n':350,
                'selection_basis':'previously observed original step5000 reasoning-stub-only failures; no gold consulted',
                'not_an_unbiased_accuracy_sample':True})
            final=args.original/'final_adapter/adapter_model.safetensors'
            if sha256(final)!=FINAL:raise ValueError('Original final adapter changed')
            if args.mode=='original_step2500':
                adapter=args.original/'trainer/milestones/step-2500/adapter_model.safetensors'
                if args.adapter_sha!=BASELINE:raise ValueError('Wrong original step2500 identity')
            else:
                control=args.control_root/args.mode;summary=read(control/'summary.json');manifest=read(control/'manifest.json')
                if any(summary.get(k)!=v for k,v in {'status':'COMPLETE','mode':args.mode,'steps':2500,'diagnostic_only':True}.items()):
                    raise ValueError('Control not complete or incorrect identity')
                if manifest['original']!=str(args.original) or summary['lr']!=1e-6:
                    raise ValueError('Control belongs to different original recipe')
                proof=summary['controls']
                if not (proof['sample_order_exact'] and proof['learning_rates_exact'] and proof['max_base_ce_difference']==0 and proof['max_weight_difference']==0):
                    raise ValueError('Control trajectory not verified')
                checkpoint=control/'trainer/checkpoint-2500';history=verify_checkpoint(checkpoint)
                if len(history)!=2500:raise ValueError('Control checkpoint length changed')
                adapter=control/'final_adapter/adapter_model.safetensors'
                a=load_file(str(adapter));b=load_file(str(checkpoint/'adapter_model.safetensors'))
                if a.keys()!=b.keys() or not all(torch.equal(a[k],b[k]) for k in a):raise ValueError('Control final weights differ from committed checkpoint')
                del a,b
            if sha256(adapter)!=args.adapter_sha:raise ValueError('Registered target weights changed')
            inputs=Path('/data/chuwenjing/tta_offline/tlm_inputs_v1/gsm8k_5k')
            rows,prepared=read_inputs(inputs,'d2843770014f663dd3df3ab796816f16e1ff384622bfd8c83ea056315602c5ba')
            references=inputs/'references.jsonl';reuse=read(args.original/'source_reuse.json');source_path=Path(reuse['path'])
            if sha256(references)!=prepared['files']['references.jsonl'] or sha256(source_path)!=reuse['sha256']:
                raise ValueError('Scoring references or Source changed')
            refs=[json.loads(line) for line in references.read_text().splitlines()]
            source=[json.loads(line) for line in source_path.read_text().splitlines()]
            if any(len(v)!=5000 or any(type(r['source_index']) is not int or r['source_index']!=i for i,r in enumerate(v)) for v in [refs,source]):
                raise ValueError('Scoring index identity changed')
            loaded=LoadedModel('llama3b','gsm8k_5k',42,initialization='gaussian')
            if loaded.contract['frozen_sha256']!=BASE:raise ValueError('Noncanonical base')
            encoded=loaded.tokens(rows);saved=read(args.original/'input_tokens.json')['records']
            if any(info['token_sha256']!=saved[i]['token_sha256'] for i,(_,info) in enumerate(encoded)):
                raise ValueError('Input tokens changed')
            loaded.model=Accelerator(mixed_precision='fp16').prepare_model(loaded.model,evaluation_mode=True)
            def load(path):
                values=load_file(str(path))
                if not all(bool(torch.isfinite(v).all()) for v in values.values()):raise ValueError('Nonfinite adapter')
                set_peft_model_state_dict(loaded.model,values);actual=get_peft_model_state_dict(loaded.model)
                if actual.keys()!=values.keys() or not all(torch.equal(actual[k].detach().cpu(),values[k]) for k in values):raise ValueError('Weight load mismatch')
            load(final);replayed=[]
            write('status.json',{'status':'REPLAYING_ORIGINAL_FAILURES','mode':args.mode,'n':16})
            for i in selected:
                result=loaded.generate(encoded[i][0])
                if result['generated_ids']!=predictions[i]['generated_ids'] or result['predict']!=predictions[i]['predict']:
                    raise ValueError('Original step5000 failure did not exactly replay at '+str(i))
                replayed.append(i)
            write('original_failure_replay.json',{'generated_ids_and_text_exact':True,'indices':replayed})
            load(adapter);scorer=Scorer('exact_match');answers=[]
            for i in selected:
                result=loaded.generate(encoded[i][0])
                source_score=scorer.score(source[i]['predict'],refs[i]['output'])
                if source_score!=source[i]['score']:raise ValueError('Paired Source score differs')
                answers.append({'source_index':i,**result,'stub_only':is_stub(result['predict']),
                    'score':scorer.score(result['predict'],refs[i]['output']),'source_score':source_score})
                write('answers.json',answers);write('status.json',{'status':'EVALUATING','mode':args.mode,'completed':len(answers),'n':16})
            if frozen_digest(loaded.model)!=BASE:raise ValueError('Frozen base changed')
            summary={'status':'COMPLETE','mode':args.mode,'n':16,'stub_only':sum(r['stub_only'] for r in answers),
                'empty':sum(not r['predict'].strip() for r in answers),'correct':sum(r['score'] for r in answers),
                'source_correct':sum(r['source_score'] for r in answers),'hit_limit':sum(r['hit_generation_limit'] for r in answers),
                'original_step5000_stub_only':16,'exact_original_failure_replay':True,'adapter_sha256':args.adapter_sha,
                'training_performed':False,'optimizer_created':False,'not_an_unbiased_accuracy_sample':True,
                'original_runs_modified':False,'seconds':time.monotonic()-started}
            write('summary.json',summary);write('status.json',{'status':'COMPLETE','mode':args.mode})
    except BaseException as error:
        write('failure.json',{'type':type(error).__name__,'error':str(error),'seconds':time.monotonic()-started});raise


if __name__=='__main__':main()
