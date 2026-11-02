"""Run a bounded diagnostic matrix on explicitly allowed idle GPUs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

MODES=('all','role_masked','wrapper_masked','both_masked')


def dispatchable(jobs):
    states={j['key']:j['state'] for j in jobs}
    return sorted([j for j in jobs if j['state']=='WAIT' and
                   (j['mode']=='all' or states.get(j['model']+'__all')=='COMPLETE')],
                  key=lambda j:j['mode']!='all')


def write(path,value):
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2)+'\n');os.replace(temp,path)


def validate_safety(jobs,gpus,output):
    if len(gpus)!=3 or any(type(v) is not int for v in gpus) or set(gpus)!={0,4,7}:
        raise ValueError('Expected exactly three distinct reserved integer GPU IDs')
    root=Path(output).resolve()
    for job in jobs:
        original=Path(job['original']).resolve()
        if root.is_relative_to(original) or original.is_relative_to(root):
            raise ValueError('Diagnostic coordinator must be outside every original output tree')


def verify_diagnostic(path,hashes):
    if not {'diagnose.py','targets.py','masked_ce.py','SHA256SUMS'}<=set(hashes):
        raise ValueError('Diagnostic source hash coverage incomplete')
    if not all(hashlib.sha256((path/k).read_bytes()).hexdigest()==v for k,v in hashes.items()):
        raise ValueError('Frozen diagnostic differs')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();config=json.loads(args.config.read_text());jobs=config['jobs']
    required={m+'__'+v for m in ['llama3b','qwen3b'] for v in MODES}
    if len(jobs)!=8 or {j['key'] for j in jobs}!=required:raise ValueError('Expected the registered 2x4 matrix')
    for j in jobs:
        if j['key']!=j['model']+'__'+j['mode']:raise ValueError('Unsafe task key')
    validate_safety(jobs,config['gpus'],args.output)
    diagnostic=Path(config['diagnostic'])
    verify_diagnostic(diagnostic,config['diagnostic_hashes'])
    args.output.mkdir(parents=True,exist_ok=False)
    import fcntl
    with (args.output/'.coordinator.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        children={}
        for j in jobs:j.update(state='WAIT',output=str(args.output/j['key']))
        while True:
            for key,child in list(children.items()):
                code=child.poll()
                if code is None:continue
                job=next(j for j in jobs if j['key']==key);out=Path(job['output'])
                if code==0 and (out/'summary.json').exists():
                    summary=json.loads((out/'summary.json').read_text())
                    if (summary.get('status')=='COMPLETE' and summary.get('steps')==1000 and
                            summary.get('model')==job['model'] and summary.get('mode')==job['mode']):
                        job.update(state='COMPLETE',summary=summary)
                    else:job.update(state='FAILED',error='Unexpected diagnostic summary')
                else:
                    failure=json.loads((out/'failure.json').read_text()) if (out/'failure.json').exists() else None
                    job.update(state='FAILED',exit_code=code,failure=failure)
                del children[key]
            for j in jobs:
                if j['state']=='WAIT' and j['mode']!='all':
                    baseline=next(x for x in jobs if x['key']==j['model']+'__all')
                    if baseline['state']=='FAILED':j.update(state='BLOCKED',error='Baseline replay failed')
            used={j['gpu'] for j in jobs if j['state']=='ACTIVE'}
            available=[]
            for gpu in config['gpus']:
                if gpu in used:continue
                state=subprocess.check_output(['nvidia-smi','--id='+str(gpu),'--query-gpu=memory.used,utilization.gpu',
                                               '--format=csv,noheader,nounits'],text=True)
                memory,util=[int(v.strip()) for v in state.strip().split(',')]
                if memory<128 and util<5:available.append(gpu)
            for job,gpu in zip(dispatchable(jobs),available):
                command=['/data/chuwenjing/conda-envs/tta-llm/bin/python','-B','-u',str(diagnostic/'diagnose.py'),
                         '--runtime',job['runtime'],'--original',job['original'],'--model',job['model'],
                         '--mode',job['mode'],'--steps','1000','--output',job['output']]
                if job['mode']!='all':command+=['--baseline',str(args.output/(job['model']+'__all'))]
                env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu),'CUBLAS_WORKSPACE_CONFIG':':4096:8',
                     'PYTHONHASHSEED':'0' if job['model']=='llama3b' else '1'}
                log=Path(job['output']+'.log')
                with log.open('xb') as f:
                    child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,env=env,start_new_session=True)
                children[job['key']]=child;job.update(state='ACTIVE',gpu=gpu,pid=child.pid,log=str(log))
            terminal=all(j['state'] in ['COMPLETE','FAILED','BLOCKED'] for j in jobs)
            state='COMPLETE' if terminal and all(j['state']=='COMPLETE' for j in jobs) else 'NEEDS_REVIEW' if terminal else 'RUNNING'
            write(args.output/'status.json',{'status':state,'jobs':jobs,'time':time.time(),
                  'diagnostic_only':True,'original_runs_modified':False,'steps':1000})
            if terminal:return
            time.sleep(20)


if __name__=='__main__':main()
