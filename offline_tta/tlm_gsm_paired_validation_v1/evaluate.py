"""Registered 200-question paired evaluation; no training or optimizer creation."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
import time
from protocol import SEED,select_indices,is_stub,artifact_paths,require_assets,protect_output,replay_references

CODE='b6da067b55f5c65241ae6fbd8c706b88523f64b27dd9568ad5661255b89d7999'
BASE='c80d597af8e0f5bb8e99a6545680edb580ed2fe59c9cba4af0f39b568c15be77'
FINAL='01c9dabc1294cc723ed579680ee65e85eba3db0166c19c2f0b5f04ba3b89ffa4'
ADAPTERS={'original_step2500':'3d0fc755e80442d9bce47647eee0fe1c3646784dac499ef88f98f93586b84162',
          'hint_masked':'fc87c4353f0f076c22f3eb1c2d8ab9d02c6b2ba128a861599cb539064c93c304',
          'eot_masked':'8810d26ca78766be2594253aa375fab2902d353acda7eebdd21f01fd9a0986ae',
          'both_masked':'71806639825909a49e53e2591898fca1e7270dc4c02cbe3d2dd4c187f5ed8e24'}


def read(path):return json.loads(Path(path).read_text())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registration',type=Path,required=True);parser.add_argument('--registration-sha',required=True)
    parser.add_argument('--mode',choices=ADAPTERS,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if hashlib.sha256(args.registration.read_bytes()).hexdigest()!=args.registration_sha:raise ValueError('Registration changed')
    reg=read(args.registration);original=Path(reg['original']);control_root=Path(reg['control_root']);replay_root=Path(reg['replay_root'])
    required=artifact_paths(reg);require_assets(reg.get('assets'),required)
    if reg['inputs']!='/data/chuwenjing/tta_offline/tlm_inputs_v1/gsm8k_5k' or reg['source_path']!=read(original/'source_reuse.json')['path']:
        raise ValueError('Registered inputs or Source identity differs')
    protect_output(args.output,[original,control_root,replay_root,reg['runtime'],Path(__file__).resolve().parent,
        '/data/chuwenjing/results/tent_formal_v3_6gpu','/data/chuwenjing/tta_offline/tlm_inputs_v1',
        *[Path(p).parent for p in required]])
    args.output.mkdir(parents=True,exist_ok=False);sys.path.insert(0,reg['runtime'])
    import torch
    from accelerate import Accelerator
    from peft import get_peft_model_state_dict,set_peft_model_state_dict
    from safetensors.torch import load_file
    from runtime import LoadedModel,atomic_json,sha256,code_identity,read_inputs,output_lock,gpu_lock,frozen_digest,Scorer
    from train_core import verify_checkpoint
    from protocol import require_control
    started=time.monotonic();write=lambda name,value:atomic_json(args.output/name,value)
    try:
        with output_lock(args.output),gpu_lock():
            if code_identity()!=CODE or reg['seed']!=SEED or reg['count']!=200 or reg['training_performed'] is not False:
                raise ValueError('Wrong frozen recipe or selection registration')
            if any(sha256(path)!=expected for path,expected in reg['assets'].items()):
                raise ValueError('Registered source artifact changed')
            previous=read(replay_root/'original_step2500/selection.json')['indices']
            probe=list(range(20))+sorted(random.Random(271828).sample(range(20,5000),80))
            excluded=sorted(set(probe+previous));selected=select_indices(5000,excluded)
            if reg['excluded']!=excluded or reg['indices']!=selected or reg['adapter_hashes']!=ADAPTERS:
                raise ValueError('Selected cases or registered weights changed')
            cfg=read(original/'run_manifest.json')['config']
            if any(cfg.get(k)!=v for k,v in {'code_id':CODE,'model':'llama3b','dataset':'gsm8k_5k','seed':42,'count':5000,'lora_init':'gaussian'}.items()):
                raise ValueError('Wrong completed original experiment')
            final=original/'final_adapter/adapter_model.safetensors'
            predictions=[json.loads(line) for line in (original/'tlm.jsonl').read_text().splitlines()]
            if sha256(final)!=FINAL or sha256(original/'tlm.jsonl')!=read(original/'metrics.json')['predictions_sha256']:
                raise ValueError('Original final weights or predictions changed')
            if len(predictions)!=5000 or any(r['source_index']!=i for i,r in enumerate(predictions)):
                raise ValueError('Original prediction indexes changed')
            if args.mode=='original_step2500':
                adapter=original/'trainer/milestones/step-2500/adapter_model.safetensors'
            else:
                folder=control_root/args.mode
                require_control(read(folder/'summary.json'),read(folder/'manifest.json'),args.mode,str(original))
                checkpoint=folder/'trainer/checkpoint-2500';history=verify_checkpoint(checkpoint)
                if len(history)!=2500:raise ValueError('Control checkpoint step differs')
                adapter=folder/'final_adapter/adapter_model.safetensors'
                a=load_file(str(adapter));b=load_file(str(checkpoint/'adapter_model.safetensors'))
                if a.keys()!=b.keys() or not all(torch.equal(a[k],b[k]) for k in a):raise ValueError('Final and committed control weights differ')
                del a,b
            if sha256(adapter)!=ADAPTERS[args.mode]:raise ValueError('Target weights differ')
            prior_summary=read(replay_root/args.mode/'summary.json')
            if prior_summary['status']!='COMPLETE' or prior_summary['adapter_sha256']!=ADAPTERS[args.mode] or not prior_summary['exact_original_failure_replay']:
                raise ValueError('No verified prior inference reference')
            inputs=Path('/data/chuwenjing/tta_offline/tlm_inputs_v1/gsm8k_5k')
            rows,prepared=read_inputs(inputs,'d2843770014f663dd3df3ab796816f16e1ff384622bfd8c83ea056315602c5ba')
            refs_path=inputs/'references.jsonl';reuse=read(original/'source_reuse.json');source_path=Path(reuse['path'])
            if sha256(refs_path)!=prepared['files']['references.jsonl'] or sha256(source_path)!=reuse['sha256']:
                raise ValueError('Source or references changed')
            refs=[json.loads(line) for line in refs_path.read_text().splitlines()]
            source=[json.loads(line) for line in source_path.read_text().splitlines()]
            if any(len(v)!=5000 or any(type(r['source_index']) is not int or r['source_index']!=i for i,r in enumerate(v)) for v in [refs,source]):
                raise ValueError('Reference indexes changed')
            loaded=LoadedModel('llama3b','gsm8k_5k',42,initialization='gaussian')
            if loaded.contract['frozen_sha256']!=BASE:raise ValueError('Noncanonical base')
            encoded=loaded.tokens(rows);saved=read(original/'input_tokens.json')['records']
            if any(info['token_sha256']!=saved[i]['token_sha256'] for i,(_,info) in enumerate(encoded)):
                raise ValueError('Input tokens changed')
            loaded.model=Accelerator(mixed_precision='fp16').prepare_model(loaded.model,evaluation_mode=True)
            def load(path):
                values=load_file(str(path))
                if not all(bool(torch.isfinite(v).all()) for v in values.values()):raise ValueError('Nonfinite adapter')
                set_peft_model_state_dict(loaded.model,values);actual=get_peft_model_state_dict(loaded.model)
                if actual.keys()!=values.keys() or not all(torch.equal(actual[k].detach().cpu(),values[k]) for k in values):
                    raise ValueError('Loaded adapter differs')
            def replay(path,expected):
                load(path);result=[]
                for row in expected:
                    i=row['source_index'];answer=loaded.generate(encoded[i][0])
                    if answer['generated_ids']!=row['generated_ids'] or answer['predict']!=row['predict']:
                        raise ValueError('Inference replay differs at '+str(i))
                    result.append({'source_index':i,'generated_ids_and_text_exact':True})
                return result
            write('status.json',{'status':'REPLAYING','mode':args.mode,'n':200,'training_performed':False})
            original_replay=replay(final,[predictions[i] for i in selected[:2]])
            references=replay_references(read(replay_root/args.mode/'answers.json'),previous)
            target_replay=replay(adapter,references)
            if len(original_replay)!=2 or len(target_replay)!=2:raise ValueError('Exactly two replay rows are required per checkpoint')
            write('replay.json',{'original_step5000':original_replay,'target_step2500':target_replay})
            write('selection.json',{'indices':selected,'excluded':excluded,'seed':SEED,'registration_sha256':args.registration_sha,
                                    'same_adaptation_population':True,'not_a_full_score':True})
            scorer=Scorer('exact_match');answers=[]
            for i in selected:
                answer=loaded.generate(encoded[i][0]);source_score=scorer.score(source[i]['predict'],refs[i]['output'])
                original_score=scorer.score(predictions[i]['predict'],refs[i]['output'])
                if source_score!=source[i]['score'] or original_score!=predictions[i]['score']:
                    raise ValueError('Stored comparison score differs')
                answers.append({'source_index':i,**answer,'stub_only':is_stub(answer['predict']),
                    'score':scorer.score(answer['predict'],refs[i]['output']),'source_score':source_score,'original_step5000_score':original_score})
                write('answers.json',answers);write('status.json',{'status':'EVALUATING','mode':args.mode,'completed':len(answers),'n':200,
                    'elapsed_seconds':time.monotonic()-started,'training_performed':False})
            if frozen_digest(loaded.model)!=BASE:raise ValueError('Frozen base changed')
            summary={'status':'COMPLETE','mode':args.mode,'steps':2500,'n':200,'correct':sum(r['score'] for r in answers),
                'source_correct':sum(r['source_score'] for r in answers),'original_step5000_correct':sum(r['original_step5000_score'] for r in answers),
                'stub_only':sum(r['stub_only'] for r in answers),'empty':sum(not r['predict'].strip() for r in answers),
                'hit_limit':sum(r['hit_generation_limit'] for r in answers),'registration_sha256':args.registration_sha,
                'adapter_sha256':ADAPTERS[args.mode],'inference_replay_exact':True,'training_performed':False,'optimizer_created':False,
                'same_adaptation_population':True,'not_a_full_score':True,'original_runs_modified':False,'seconds':time.monotonic()-started}
            write('summary.json',summary);write('status.json',{'status':'COMPLETE','mode':args.mode,'n':200,'completed':200})
    except BaseException as error:
        write('failure.json',{'type':type(error).__name__,'error':str(error),'seconds':time.monotonic()-started});raise


if __name__=='__main__':main()
