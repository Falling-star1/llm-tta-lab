"""Evaluation only: retain every prediction and record quality alerts without aborting."""
import argparse,hashlib,json,math,os,sys,time,uuid
from pathlib import Path

CODE='b6da067b55f5c65241ae6fbd8c706b88523f64b27dd9568ad5661255b89d7999'
BASES={'qwen3b':'644c521c2e650bb46c5bfcedea082bbc3a02b9a5c0f67297a44633cb035384bc',
       'llama3b':'c80d597af8e0f5bb8e99a6545680edb580ed2fe59c9cba4af0f39b568c15be77'}


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path,value):
    path=Path(path);temp=path.with_name(path.name+'.tmp-'+uuid.uuid4().hex)
    with temp.open('x',encoding='utf-8') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    os.replace(temp,path)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def require(value,message):
    if not value:raise ValueError(message)


def append(path,value):
    with Path(path).open('a',encoding='utf-8') as stream:
        stream.write(json.dumps(value,ensure_ascii=False,allow_nan=False)+'\n');stream.flush();os.fsync(stream.fileno())


def read_prefix(path):
    raw=Path(path).read_bytes() if Path(path).exists() else b''
    end=len(raw) if raw.endswith(b'\n') else raw.rfind(b'\n')+1
    rows=[json.loads(line) for line in raw[:end].splitlines()]
    require([r.get('source_index') for r in rows]==list(range(len(rows))),'Prediction indexes differ')
    return rows,raw[:end],raw[end:]


def record_quality(check,rows,source,output):
    output=Path(output);report=check(rows,source)
    write(output/'quality_latest.json',report)
    if report['status']!='WAIT':
        path=output/'quality_windows.jsonl'
        existing=[json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()] if path.exists() else []
        require(not any(v['window_start']==report['window_start'] and v!=report for v in existing),'Existing quality window changed')
        if not any(v['window_start']==report['window_start'] for v in existing):
            append(path,report)
            if report['status']=='ALERT':append(output/'quality_alerts.jsonl',report)
    if report['status']=='ALERT' and not (output/'quality_alert.json').exists():write(output/'quality_alert.json',report)
    return report


def validate_training_identity(manifest,training,contract,model,dataset):
    expected=dict(code_id=CODE,model=model,dataset=dataset,count=5000,seed=42,
                  lora_init='gaussian',initialized_from='base',chunk_size=128,protocol='A')
    require(all(manifest.get('config',{}).get(k)==v for k,v in expected.items()),'Completed formal training identity differs')
    require(training.get('n')==5000 and training.get('global_step')==5000
            and all(training.get(k) is True for k in ['frozen_base_unchanged','adapter_parameters_finite','optimizer_state_finite']),
            'A complete finite 5000-input adaptation is required')
    require(contract.get('frozen_sha256')==BASES[model],'Base does not match canonical model')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime',type=Path,required=True)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--model',choices=list(BASES),required=True)
    parser.add_argument('--dataset',choices=['geosignal_5k','dolly_5k','gsm8k_5k'],required=True)
    args=parser.parse_args();sys.path.insert(0,str(args.runtime))
    from runtime import (code_identity,gpu_lock,output_lock,LoadedModel,read_inputs,
                         specifications,Scorer,frozen_digest)
    from launch import verify_delivery
    verify_delivery(args.runtime);require(code_identity()==CODE,'Frozen training runtime changed')
    inputs=Path('/data/chuwenjing/tta_offline/tlm_inputs_v1')/args.dataset
    with gpu_lock(),output_lock(args.run):
        manifest=read(args.run/'run_manifest.json');training=read(args.run/'training_complete.json')
        contract=read(args.run/'model_contract.json')
        validate_training_identity(manifest,training,contract,args.model,args.dataset)
        failure=args.run/'failure.json'
        if failure.exists():
            old_failure=read(failure)
            require(old_failure.get('type')=='RuntimeError' and old_failure.get('message','').startswith('Output quality alert;'),
                    'Only an output-quality stop may be continued automatically')
        _,spec=specifications(args.model,args.dataset);rows,prepared=read_inputs(inputs,spec['sha256'])
        require(manifest['config']['input_sha256']==prepared['files']['inputs.jsonl'],'Training inputs changed')
        require(sha(inputs/'references.jsonl')==prepared['files']['references.jsonl'],'References changed')
        references=[json.loads(v) for v in (inputs/'references.jsonl').read_text(encoding='utf-8').splitlines()]
        require(len(references)==5000 and [r['source_index'] for r in references]==list(range(5000)),'Reference indexes differ')
        reuse=read(args.run/'source_reuse.json');source_path=Path(reuse['path'])
        require(sha(source_path)==reuse['sha256'],'Source changed')
        source=[json.loads(v) for v in source_path.read_text(encoding='utf-8').splitlines()]
        require(len(source)==5000 and [r['source_index'] for r in source]==list(range(5000)),'Source incomplete')
        done,prefix,tail=read_prefix(args.run/'tlm.jsonl');require(len(done)<5000,'Evaluation already complete')
        loaded=LoadedModel(args.model,args.dataset,42,initialization='gaussian')
        require(loaded.contract==contract,'Loaded model differs from completed training')
        import torch
        from safetensors.torch import load_file
        from peft import set_peft_model_state_dict,get_peft_model_state_dict
        adapter=args.run/'final_adapter/adapter_model.safetensors';adapter_sha=sha(adapter)
        saved=load_file(str(adapter));set_peft_model_state_dict(loaded.model,saved)
        actual=get_peft_model_state_dict(loaded.model)
        require(saved.keys()==actual.keys() and all(torch.equal(v.cpu(),actual[k].detach().cpu()) for k,v in saved.items()),
                'Loaded adapter differs from saved final adapter')
        require(frozen_digest(loaded.model)==contract['frozen_sha256'],'Base changed while loading adapter')
        encoded=loaded.tokens(rows);stored=read(args.run/'input_tokens.json')['records']
        require(len(stored)==5000 and [info for _,info in encoded]==stored,'Tokenization differs from completed training')
        for i,(_,info) in enumerate(encoded):
            require(all(source[i][k]==info[k] for k in ['source_index','original_prompt_sha256','used_prompt_tokens']),
                    'Source input differs')
        for i,row in enumerate(done):
            require(all(row.get(k)==v for k,v in encoded[i][1].items()),'Saved prediction input differs')
        # Restore the same outer Accelerate wrapper that the original Trainer
        # prepared. PEFT generate may delegate past it; measure actual decoder
        # context and require exact saved-token replay instead of assuming AMP.
        from accelerate import Accelerator
        accelerator=Accelerator(mixed_precision='fp16')
        loaded.model=accelerator.prepare_model(loaded.model,evaluation_mode=True)
        trace=[]
        hook=loaded.model.get_base_model().model.register_forward_pre_hook(
            lambda module,values:trace.append(torch.is_autocast_enabled('cuda')))
        replay=[]
        try:
            for index in sorted({0,len(done)-1} if done else set()):
                generated=loaded.generate(encoded[index][0])
                require(generated['generated_ids']==done[index]['generated_ids'] and generated['predict']==done[index]['predict'],
                        'Evaluation replay differs; no predictions appended')
                replay.append({'source_index':index,'generated_ids_exact':True})
        finally:hook.remove()
        scorer=Scorer(loaded.dataset['metric'])
        require(all(abs(scorer.score(r['predict'],references[i]['output'])-r['score'])<=1e-12 for i,r in enumerate(done)),
                'Existing prediction scores differ')
        audit=args.run/('evaluation_resume_'+time.strftime('%Y%m%d_%H%M%S',time.gmtime()))
        audit.mkdir(exist_ok=False)
        write(audit/'prior_manifest.json',manifest)
        if tail:(audit/'incomplete_tail_preserved.txt').write_bytes(tail)
        if failure.exists():os.rename(failure,audit/'prior_failure.json')
        write(audit/'verification.json',{'code_id':CODE,'evaluation_driver_sha256':sha(Path(__file__)),
            'policy':'Record quality alerts and evaluate every input; user requested original reproduction without tuning',
            'optimizer_steps':0,'training_performed':False,'final_adapter_sha256':adapter_sha,
            'saved_prefix_rows':len(done),'saved_prefix_sha256':hashlib.sha256(prefix).hexdigest(),
            'incomplete_tail_bytes_preserved':len(tail),'replay':replay,'decoder_autocast_values':sorted(set(trace))})
        if tail:
            with (args.run/'tlm.jsonl').open('r+b') as stream:stream.truncate(len(prefix));stream.flush();os.fsync(stream.fileno())
        write(args.run/'evaluation_policy.json',{'quality_alerts':'record_and_continue','all_predictions_retained':True,
            'training_unchanged':True,'audit':str(audit),'requested_at':'2026-10-26 user requested paper reproduction and full six-group evaluation'})
        write(args.run/'run_manifest.json',{'status':'RUNNING','stage':'evaluation','config':manifest['config'],
            'evaluation_policy':'record_quality_alerts_continue','resume_audit':str(audit)})
        from quality import quality_window
        for end in range(100,len(done)+1,100):record_quality(quality_window,done[:end],source,args.run)
        started=time.monotonic()
        try:
            for index in range(len(done),5000):
                result=loaded.generate(encoded[index][0])
                score=scorer.score(result['predict'],references[index]['output']);require(math.isfinite(score),'Nonfinite score')
                row={**encoded[index][1],**result,'score':score};append(args.run/'tlm.jsonl',row);done.append(row)
                if (index+1)%100==0:record_quality(quality_window,done,source,args.run)
                if (index+1)%25==0:print(json.dumps({'phase':'evaluation','completed':index+1,'total':5000,'seconds':time.monotonic()-started}),flush=True)
            scores=[scorer.score(r['predict'],references[i]['output']) for i,r in enumerate(done)]
            require(all(abs(s-r['score'])<=1e-12 for s,r in zip(scores,done)),'Stored scores differ')
            require(sha(adapter)==adapter_sha and frozen_digest(loaded.model)==contract['frozen_sha256'],'Weights changed during evaluation')
            with (args.run/'tlm.jsonl').open('rb') as stream:
                require(hashlib.sha256(stream.read(len(prefix))).hexdigest()==hashlib.sha256(prefix).hexdigest(),'Original prefix changed')
            metrics={'n':5000,'metric':loaded.dataset['metric'],'value_0_to_1':sum(scores)/5000,
                'generated_tokens':sum(r['generated_tokens'] for r in done),
                'hit_generation_limit_rate':sum(r['hit_generation_limit'] for r in done)/5000,
                'unique_answers':len({r['predict'] for r in done}),'predictions_sha256':sha(args.run/'tlm.jsonl')}
            write(args.run/'metrics.json',metrics)
            alerts=(args.run/'quality_alerts.jsonl').read_text(encoding='utf-8').splitlines() if (args.run/'quality_alerts.jsonl').exists() else []
            write(args.run/'run_manifest.json',{'status':'PASS','stage':'complete','config':manifest['config'],'metrics':metrics,
                'evaluation_policy':'record_quality_alerts_continue','quality_alert_windows':len(alerts),'quality_healthy':not bool(alerts)})
        except BaseException as error:
            write(args.run/'failure.json',{'type':type(error).__name__,'message':str(error)})
            write(args.run/'run_manifest.json',{'status':'FAIL','stage':'evaluation','config':manifest['config'],
                'evaluation_policy':'record_quality_alerts_continue','error':str(error)})
            raise


if __name__=='__main__':
    try:main()
    except BaseException as error:
        if '--run' in sys.argv:
            target=Path(sys.argv[sys.argv.index('--run')+1])
            if target.is_dir():
                write(target/'evaluation_continuation_failure.json',{'type':type(error).__name__,'message':str(error)})
        raise
