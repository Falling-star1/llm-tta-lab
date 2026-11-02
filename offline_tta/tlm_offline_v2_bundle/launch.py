"""Six independent TLM jobs, each with preflight -> pilot -> formal gates."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from runtime import HERE,REPOSITORY,atomic_json,code_identity,read_inputs,sha256


def jobs_for_gpus(value,skip=()):
    items=value.split(',')
    if len(items)!=6 or not all(x.isdecimal() for x in items):
        raise ValueError('Select six distinct physical GPU indexes, e.g. 0,1,2,3,4,5')
    gpus=[int(x) for x in items]
    if len(set(gpus))!=6:
        raise ValueError('GPU indexes must be distinct')
    combinations=[(m,d) for m in ['llama3b','qwen3b']
                  for d in ['geosignal_5k','dolly_5k','gsm8k_5k']]
    jobs=[{'gpu':gpu,'model':model,'dataset':dataset}
          for gpu,(model,dataset) in zip(gpus,combinations)]
    names={model+'__'+dataset for model,dataset in combinations}
    skipped=set(skip)
    if not skipped<=names or skipped==names:
        raise ValueError('Skip must name registered tasks and leave at least one task')
    return [job for job in jobs if job['model']+'__'+job['dataset'] not in skipped]


def verify_delivery(root):
    root=Path(root)
    rows=(root/'SHA256SUMS').read_text().splitlines()
    if not rows:
        raise ValueError('Missing delivery manifest')
    for row in rows:
        digest,name=row.split(maxsplit=1)
        path=(root/name.lstrip('*')).resolve()
        if not path.is_relative_to(root.resolve()) or sha256(path)!=digest:
            raise ValueError('Delivery file differs: '+name)


def install_snapshot():
    verify_delivery(HERE)
    target=REPOSITORY/'local_repro'/('tlm_offline_'+code_identity()[:12])
    if not target.exists():
        shutil.copytree(HERE,target,ignore=shutil.ignore_patterns('__pycache__'))
    verify_delivery(target)
    if code_identity()!=_snapshot_code(target):
        raise ValueError('Installed snapshot differs')
    return target


def _snapshot_code(path):
    import hashlib
    files=sorted(p for p in path.rglob('*') if p.is_file() and
                 p.suffix in ('.py','.json','.sh','.md') and '__pycache__' not in p.parts)
    value={p.relative_to(path).as_posix():sha256(p) for p in files}
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()


def prepare_all(code,root):
    root.mkdir(parents=True,exist_ok=True)
    spec=json.loads((code/'datasets.json').read_text())
    raw=Path('/data/chuwenjing/datasets/AdaptEval/496c9ed97d0f')
    for item in spec['datasets']:
        target=root/item['alias']
        if target.exists():
            _,manifest=read_inputs(target,item['sha256'])
            if sha256(target/'references.jsonl')!=manifest['files']['references.jsonl']:
                raise ValueError('Prepared reference file differs')
        else:
            subprocess.run([sys.executable,'-B',str(code/'prepare_inputs.py'),
                '--data',str(raw/item['filename']),'--output',str(target),
                '--expected-sha256',item['sha256'],'--expected-records','5000'],check=True)


def execute_job(job,code,prepared,root,mode):
    name=job['model']+'__'+job['dataset']
    status=root/(name+'.status.json')
    env=os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(job['gpu']),CUBLAS_WORKSPACE_CONFIG=':4096:8',
               PYTHONDONTWRITEBYTECODE='1',TOKENIZERS_PARALLELISM='false',
               HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
    receipt=root/'preflight'/name/'receipt.json'
    for stage in (['preflight'] if mode=='preflight' else ['preflight','pilot','formal']):
        target=root/stage/name
        command=[sys.executable,'-B','-u',str(code/'run.py'),stage,
                 '--model',job['model'],'--dataset',job['dataset'],
                 '--inputs',str(prepared/job['dataset']),'--output',str(target)]
        if stage!='preflight':
            command+=['--receipt',str(receipt)]
        log=root/(name+'.'+stage+'.log')
        atomic_json(status,{**job,'status':'RUNNING','stage':stage,'log':str(log)})
        with log.open('xb') as stream:
            process=subprocess.run(command,env=env,stdout=stream,stderr=subprocess.STDOUT)
        if process.returncode:
            record={**job,'status':'FAIL','stage':stage,'log':str(log),'exit_code':process.returncode}
            atomic_json(status,record)
            return record
        print(json.dumps({'task':name,'stage':stage,'status':'PASS'},ensure_ascii=False),flush=True)
    record={**job,'status':'PASS','stage':mode}
    atomic_json(status,record)
    return record


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpus',required=True)
    parser.add_argument('--output',type=Path,required=True,help='new unique run root')
    parser.add_argument('--mode',choices=['preflight','all'],default='preflight')
    parser.add_argument('--skip',default='',help='comma-separated task names to leave untouched')
    parser.add_argument('--inputs',type=Path,default=Path('/data/chuwenjing/tta_offline/tlm_inputs_v1'))
    args=parser.parse_args()
    jobs=jobs_for_gpus(args.gpus,skip=args.skip.split(',') if args.skip else ())
    args.output.mkdir(parents=True,exist_ok=False)
    code=install_snapshot()
    prepare_all(code,args.inputs)
    atomic_json(args.output/'launch.json',{'code':str(code),'code_id':code_identity(),
                'jobs':jobs,'mode':args.mode,'skipped':args.skip,'started_at':time.time()})
    results=[]
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures=[pool.submit(execute_job,job,code,args.inputs,args.output,args.mode) for job in jobs]
        for future in as_completed(futures):
            results.append(future.result())
            atomic_json(args.output/'summary.json',results)
    if any(x['status']!='PASS' for x in results):
        raise SystemExit(1)


if __name__=='__main__':
    main()
