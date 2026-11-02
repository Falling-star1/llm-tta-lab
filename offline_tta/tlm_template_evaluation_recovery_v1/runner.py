"""Bounded evaluation-only recovery after the original diagnostic matrix exits."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def recognized_lr_rejection(error):
    return (error.get('type')=='ValueError' and error.get('error','').startswith('Original trajectory controls differ:')
            and "'learning_rates_exact': False" in error['error'])


def write(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');os.replace(tmp,path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-matrix',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();source=args.source_matrix
    initial=json.loads((source/'status.json').read_text())
    required={m+'__'+v for m in ['llama3b','qwen3b'] for v in ['all','role_masked','wrapper_masked','both_masked']}
    if len(initial['jobs'])!=8 or {j['key'] for j in initial['jobs']}!=required:raise ValueError('Incomplete source matrix')
    package=Path(__file__).resolve().parent
    for line in (package/'SHA256SUMS').read_text().splitlines():
        expected,name=line.split(maxsplit=1)
        if hashlib.sha256((package/name).read_bytes()).hexdigest()!=expected:raise ValueError('Recovery source changed')
    protected=[source]+[Path(j['original']) for j in initial['jobs']]
    if any(args.output.resolve().is_relative_to(v.resolve()) or v.resolve().is_relative_to(args.output.resolve()) for v in protected):
        raise ValueError('Recovery runner output overlaps protected run')
    args.output.mkdir(parents=True,exist_ok=False)
    while True:
        state=json.loads((source/'status.json').read_text())
        if len(state['jobs'])!=8 or {j['key'] for j in state['jobs']}!=required:raise ValueError('Source matrix identity changed')
        if all(j['state'] in ['COMPLETE','FAILED','BLOCKED'] for j in state['jobs']):break
        write(args.output/'status.json',{'status':'WAITING_FOR_MATRIX','training_performed':False,'time':time.time()})
        time.sleep(20)
    jobs=[];excluded=[]
    for j in state['jobs']:
        if j['state']!='FAILED':continue
        failed=Path(j['output']);failure=json.loads((failed/'failure.json').read_text()) if (failed/'failure.json').exists() else {}
        if not recognized_lr_rejection(failure):excluded.append({'task':j['key'],'failure':failure});continue
        if j['key']!=j['model']+'__'+j['mode'] or j['model'] not in ['llama3b','qwen3b'] or j['mode'] not in ['role_masked','wrapper_masked','both_masked']:
            raise ValueError('Unregistered recovery key')
        jobs.append({**j,'state':'WAIT','failed_output':str(failed),'output':str(args.output/j['key'])})
    children={}
    while True:
        for key,child in list(children.items()):
            code=child.poll()
            if code is None:continue
            job=next(j for j in jobs if j['key']==key);out=Path(job['output'])
            if code==0 and (out/'summary.json').exists():
                summary=json.loads((out/'summary.json').read_text());job.update(state='COMPLETE' if summary.get('status')=='COMPLETE' else 'FAILED',summary=summary)
            else:job.update(state='FAILED',failure=json.loads((out/'failure.json').read_text()) if (out/'failure.json').exists() else {'exit_code':code})
            del children[key]
        used={j['gpu'] for j in jobs if j['state']=='ACTIVE'};available=[]
        for gpu in [0,4,7]:
            if gpu in used:continue
            raw=subprocess.check_output(['nvidia-smi','--id='+str(gpu),'--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
            memory,util=[int(v.strip()) for v in raw.strip().split(',')]
            if memory<128 and util<5:available.append(gpu)
        for job,gpu in zip([j for j in jobs if j['state']=='WAIT'],available):
            cmd=['/data/chuwenjing/conda-envs/tta-llm/bin/python','-B','-u',str(package/'evaluate.py'),
                 '--runtime',job['runtime'],'--original',job['original'],'--failed',job['failed_output'],
                 '--baseline',str(source/(job['model']+'__all')),'--model',job['model'],'--output',job['output']]
            env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu),'CUBLAS_WORKSPACE_CONFIG':':4096:8',
                 'PYTHONHASHSEED':'0' if job['model']=='llama3b' else '1'}
            with Path(job['output']+'.log').open('xb') as f:
                child=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,env=env,start_new_session=True)
            children[job['key']]=child;job.update(state='ACTIVE',gpu=gpu,pid=child.pid)
        terminal=all(j['state'] in ['COMPLETE','FAILED'] for j in jobs)
        write(args.output/'status.json',{'status':('COMPLETE' if all(j['state']=='COMPLETE' for j in jobs) and not excluded else 'NEEDS_REVIEW') if terminal else 'RUNNING',
              'jobs':jobs,'excluded_failures':excluded,'training_performed':False,'source_matrix_modified':False,'time':time.time()})
        if terminal:return
        time.sleep(20)


if __name__=='__main__':main()
