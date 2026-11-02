"""Wait for every preflight, then every 100-row pilot, then start six formal jobs."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

TASKS={(m,d) for m in ['llama3b','qwen3b'] for d in ['geosignal_5k','dolly_5k','gsm8k_5k']}
EXPECTED_BASES={
    'qwen3b':'644c521c2e650bb46c5bfcedea082bbc3a02b9a5c0f67297a44633cb035384bc',
    'llama3b':'c80d597af8e0f5bb8e99a6545680edb580ed2fe59c9cba4af0f39b568c15be77'}


def sha(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''): value.update(block)
    return value.hexdigest()


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path,value):
    path=Path(path);temp=path.with_suffix(path.suffix+'.tmp')
    with temp.open('w',encoding='utf-8') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.write('\n');stream.flush();os.fsync(stream.fileno())
    os.replace(temp,path)


def require(condition,message):
    if not condition: raise ValueError(message)


def validate_preflight(receipt,expected):
    require(all(receipt.get(k)==v for k,v in expected.items()),'Preflight identity differs')
    require(receipt.get('status')=='PASS' and receipt.get('frozen_base_unchanged') is True
            and receipt.get('checkpoint_continuation')=='EXACT_MATCH'
            and receipt.get('source_sample_match') is True,'Incomplete real preflight')
    precision=receipt.get('precision_path_check',{})
    require(precision.get('status')=='PASS' and precision.get('failed_parameters')==[]
            and precision.get('all_decoder_calls_under_autocast') is True,'Precision gate failed')
    continuation=receipt.get('fp16_trainer_disk_resume',{})
    require(continuation.get('status')=='EXACT_MATCH' and continuation.get('steps')==16
            and continuation.get('successful_optimizer_steps',0)>=2
            and continuation.get('successful_optimizer_steps',0)+continuation.get('skipped_optimizer_steps',-1)==16,
            'Actual optimizer continuation is incomplete')
    lengths=continuation.get('declared_precision_lengths',[])
    checks=continuation.get('precision_path_checks',[])
    require(len(lengths)>=2 and len(checks)==2*len(lengths)
            and {(r.get('tokens'),r.get('probe_b_value')) for r in checks}=={(n,b) for n in lengths for b in [0.,.001]}
            and all(r.get('status')=='PASS' for r in checks),'Declared precision coverage incomplete')
    longest=receipt.get('longest_actual_trainer',{})
    require(longest.get('status')=='PASS' and longest.get('steps')==16
            and longest.get('successful_optimizer_steps',0)>=2
            and all(longest.get(k) is True for k in ['parameters_finite','optimizer_finite',
                'history_finite','adapter_changed','adapter_restored']),'Longest actual Trainer check incomplete')
    contract=receipt.get('contract',{})
    require(expected.get('model') in EXPECTED_BASES
            and contract.get('frozen_sha256')==EXPECTED_BASES[expected['model']],
            'Preflight base differs from canonical registered model weights')
    require(contract.get('init_lora_weights')=='gaussian' and contract.get('frozen_sha256')
            and contract.get('projection_policy')=='whole-head-gemm; cpu-logits; streamed-fp32-ce'
            and contract.get('training_fp16_reduced_precision_reduction') is True
            and contract.get('generation_fp16_reduced_precision_reduction') is True,'Training precision contract differs')


def validate_pilot(directory,expected,receipt_path):
    directory=Path(directory);manifest=read(directory/'run_manifest.json');config=manifest['config']
    require(manifest.get('status')=='PASS' and manifest.get('stage')=='complete','Pilot did not complete')
    require(all(config.get(k)==v for k,v in expected.items()) and config.get('count')==100
            and config.get('initialized_from')=='base' and config.get('protocol')=='A'
            and config.get('preflight_sha256')==sha(receipt_path),'Pilot configuration differs')
    require(not (directory/'quality_alert.json').exists() and not (directory/'failure.json').exists(),'Pilot has an alert/failure')
    quality=read(directory/'quality_latest.json')
    require(quality.get('status')=='PASS' and quality.get('n')==100
            and quality.get('window_start')==0 and quality.get('alerts')==[],'100-row quality check missing or failed')
    training=read(directory/'training_complete.json')
    require(training.get('n')==100 and training.get('global_step')==100
            and training.get('frozen_base_unchanged') is True,'Pilot training incomplete')
    require(training.get('successful_optimizer_steps',0)>=2
            and training.get('successful_optimizer_steps',0)+training.get('skipped_optimizer_steps',-1)==100
            and training.get('adapter_parameters_finite') is True
            and training.get('optimizer_state_finite') is True,'Pilot optimizer updates incomplete/nonfinite')
    require(read(directory/'model_contract.json')==read(receipt_path)['contract'],'Pilot model contract differs')
    recipe=dict(learning_rate=1e-6 if expected['dataset']=='gsm8k_5k' else 5e-5,num_train_epochs=1.,
            per_device_train_batch_size=1,gradient_accumulation_steps=1,lr_scheduler_type='cosine',
            warmup_ratio=.1,fp16=True,bf16=False,weight_decay=0.,max_grad_norm=1.,seed=42,
            adam_beta1=.9,adam_beta2=.999,adam_epsilon=1e-8,optim='adamw_torch',
            gradient_checkpointing=True,gradient_checkpointing_kwargs={'use_reentrant':False})
    actual=read(directory/'resolved_config.json')
    require(all(actual.get(k)==v for k,v in recipe.items()),'Pilot training recipe differs')
    history=read(directory/'adaptation.json')
    require(len(history)==100 and {r.get('source_index') for r in history}==set(range(100)),
            'Pilot training indexes missing or duplicated')
    require(all(isinstance(r.get(k),(int,float)) and math.isfinite(r[k])
                for r in history for k in ['base_ce','current_ce','weight','loss','lr']), 'Nonfinite training values')
    raw=(directory/'tlm.jsonl').read_bytes()
    require(raw.endswith(b'\n'),'Incomplete prediction tail')
    predictions=[json.loads(row) for row in raw.splitlines()]
    require(len(predictions)==100 and [r.get('source_index') for r in predictions]==list(range(100))
            and all(isinstance(r.get('predict'),str) for r in predictions),'Pilot predictions incomplete')
    metrics=read(directory/'metrics.json')
    require(metrics.get('n')==100 and metrics.get('predictions_sha256')==sha(directory/'tlm.jsonl'),
            'Pilot metrics do not match predictions')
    return {'pilot':str(directory),'manifest_sha256':sha(directory/'run_manifest.json'),
            'prediction_sha256':sha(directory/'tlm.jsonl'),'quality':quality}


def run_barriers(stage):
    for name in ['preflight','pilot','formal']:
        if not stage(name): return False
    return True


def verify_code(directory,expected):
    directory=Path(directory)
    for row in (directory/'SHA256SUMS').read_text().splitlines():
        digest,name=row.split(maxsplit=1);path=(directory/name.lstrip('*')).resolve()
        require(path.is_relative_to(directory.resolve()) and sha(path)==digest,'Frozen code hash differs')
    files=sorted(p for p in directory.rglob('*') if p.is_file()
                 and p.suffix in ('.py','.json','.sh','.md') and '__pycache__' not in p.parts)
    values={p.relative_to(directory).as_posix():sha(p) for p in files}
    require(hashlib.sha256(json.dumps(values,sort_keys=True).encode()).hexdigest()==expected,'Wrong code identity')


def validate_all_receipts(jobs,code,code_id,receipt,expected,hashes):
    verify_code(code,code_id)
    for job in jobs:
        name=job['model']+'__'+job['dataset']
        require(sha(receipt(job))==hashes[name],'Receipt changed before the stage-wide launch barrier')
        validate_preflight(read(receipt(job)),expected(job))


def launch_batch(jobs,prelaunch,execute):
    prelaunch()
    with ThreadPoolExecutor(max_workers=6) as pool:return list(pool.map(execute,jobs))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preflights',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--expected-code',required=True)
    args=parser.parse_args()
    launch=read(args.preflights/'launch.json');code=Path(launch['code']);jobs=launch['jobs']
    require(launch.get('mode')=='preflight' and launch.get('lora_init')=='gaussian'
            and launch.get('code_id')==args.expected_code,'Wrong preflight launch')
    require(len(jobs)==6 and {(j['model'],j['dataset']) for j in jobs}==TASKS
            and len({j['gpu'] for j in jobs})==6,'Require all six distinct groups/GPUs')
    verify_code(code,args.expected_code)
    args.output.mkdir(parents=True,exist_ok=False)
    inputs=Path('/data/chuwenjing/tta_offline/tlm_inputs_v1')
    receipt_hashes={};pilot_audits=[]

    def expected(job):
        prepared=read(inputs/job['dataset']/'manifest.json')
        require(prepared['record_count']==5000,'Require 5000 inputs')
        require(all(sha(inputs/job['dataset']/name)==digest for name,digest in prepared['files'].items()),'Prepared data changed')
        return dict(code_id=args.expected_code,model=job['model'],dataset=job['dataset'],
                    seed=42,lora_init='gaussian',chunk_size=128,input_sha256=prepared['files']['inputs.jsonl'])

    def receipt(job):return args.preflights/'preflight'/(job['model']+'__'+job['dataset'])/'receipt.json'

    def execute(job,stage):
        name=job['model']+'__'+job['dataset'];out=args.output/stage/name
        verify_code(code,args.expected_code)
        require(sha(receipt(job))==receipt_hashes[name],'Receipt changed after verification')
        command=[sys.executable,'-B','-u',str(code/'run.py'),stage,'--model',job['model'],
                 '--dataset',job['dataset'],'--inputs',str(inputs/job['dataset']),'--output',str(out),
                 '--receipt',str(receipt(job)),'--lora-init','gaussian','--seed','42','--chunk-size','128']
        if stage=='pilot':command+=['--samples','100']
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(job['gpu']),CUBLAS_WORKSPACE_CONFIG=':4096:8',
                 OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
                 TOKENIZERS_PARALLELISM='false',PYTHONDONTWRITEBYTECODE='1')
        log=args.output/(name+'.'+stage+'.log')
        write(args.output/(name+'.status.json'),{**job,'stage':stage,'status':'RUNNING','log':str(log)})
        with log.open('xb') as stream:
            process=subprocess.run(command,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT)
        result={**job,'stage':stage,'status':'PASS' if process.returncode==0 else 'FAIL',
                'exit_code':process.returncode,'log':str(log)}
        write(args.output/(name+'.status.json'),result)
        return result

    def stage(name):
        write(args.output/'status.json',{'status':'RUNNING','stage':name,'time':time.time()})
        if name=='preflight':
            deadline=time.monotonic()+7200
            while True:
                states=[read(args.preflights/(j['model']+'__'+j['dataset']+'.status.json'))
                        if (args.preflights/(j['model']+'__'+j['dataset']+'.status.json')).exists()
                        else {'status':'WAIT'} for j in jobs]
                require(not any(r['status']=='FAIL' for r in states),'At least one real preflight failed; no pilot/formal started')
                if all(r['status']=='PASS' for r in states):break
                require(time.monotonic()<deadline,'Preflight wait timed out')
                time.sleep(5)
            for job in jobs:
                validate_preflight(read(receipt(job)),expected(job))
                receipt_hashes[job['model']+'__'+job['dataset']]=sha(receipt(job))
            write(args.output/'preflight_gate.json',{'status':'PASS','receipt_hashes':receipt_hashes})
            return True
        def prelaunch():
            # No worker is created until every identity/hash is checked serially.
            validate_all_receipts(jobs,code,args.expected_code,receipt,expected,receipt_hashes)
            if name=='formal':
                for job in jobs:
                    pilot_audits.append(validate_pilot(args.output/'pilot'/(job['model']+'__'+job['dataset']),expected(job),receipt(job)))
                write(args.output/'ready_for_formal.json',{'status':'PASS','code_id':args.expected_code,
                    'seed':42,'lora_init':'gaussian','count_per_group':5000,'receipt_hashes':receipt_hashes,
                    'pilots':pilot_audits,'time':time.time()})
        results=launch_batch(jobs,prelaunch,lambda job:execute(job,name))
        write(args.output/(name+'_summary.json'),results)
        require(all(r['status']=='PASS' for r in results),name+' failed; results preserved')
        return True

    try:
        run_barriers(stage)
        write(args.output/'status.json',{'status':'PASS','stage':'complete','time':time.time()})
    except BaseException as error:
        previous=read(args.output/'status.json') if (args.output/'status.json').exists() else {}
        write(args.output/'status.json',{**previous,'status':'BLOCKED','error':str(error),'time':time.time()})
        raise


if __name__=='__main__': main()
