"""Small explicit experiment launcher: plan, run, status and retryable MLflow sync.

Uses only Python's standard library. The command runs in the foreground; use
systemd-run --user to detach it. This is coordination, not an OS security sandbox.
"""
import argparse
from contextlib import contextmanager, ExitStack, closing
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import re
import signal
import sqlite3
import subprocess
import sys
import time
import urllib.parse
import urllib.error
import urllib.request

class LabError(RuntimeError):
    pass

def require(condition, message):
    if not condition:
        raise LabError(message)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def write(path, value):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.'+str(os.getpid())+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    os.replace(temporary,path)

def now():
    return datetime.now(timezone.utc).isoformat()

def snapshot(root):
    root=Path(root).resolve()
    files={}
    for path in sorted(root.rglob('*')):
        rel=path.relative_to(root)
        if any(p in {'.git','__pycache__','.venv','.pytest_cache'} for p in rel.parts):
            continue
        if path.is_file() and (path.suffix in {'.py','.json','.yaml','.yml','.toml','.sh','.md'} or path.name=='SHA256SUMS'):
            require(not path.is_symlink() and path.resolve().is_relative_to(root),'Linked source file')
            files[rel.as_posix()]=sha(path)
            require(len(files)<=5000,'Code snapshot too large; use a dedicated release directory')
    return files

def within(path, roots):
    path=Path(path).resolve()
    return any(path.is_relative_to(Path(root).resolve()) for root in roots)

def validate(request, site):
    r=request
    require(r.get('schema_version')==1,'Unsupported request schema')
    require(bool(re.fullmatch(r'[a-z0-9][a-z0-9-]{2,79}',str(r.get('experiment_id','')))),'Invalid experiment_id')
    for key in ['owner','method','model','dataset','protocol_id']:
        require(isinstance(r.get(key),str) and 0<len(r[key])<=200,'Missing '+key)
    require(bool(re.fullmatch(r'https://github\.com/Falling-star1/llm-tta-lab/issues/[1-9][0-9]*',r.get('issue',''))),'Issue must identify a repository task')
    require(bool(re.fullmatch('[0-9a-f]{40}',r.get('commit',''))),'Full source commit required')
    require(isinstance(r.get('seed'),int),'Integer seed required')
    require(isinstance(r.get('gpu_count'),int) and 0<=r['gpu_count']<=8,'Invalid GPU count')
    for key in ['timeout_seconds','queue_timeout_seconds']:
        value=r.get(key)
        require(isinstance(value,(float,int)) and math.isfinite(value) and 0<value<=14*86400,'Invalid '+key)
    require(r.get('legacy_lock_policy') in {'wrapper','child'},'Choose wrapper or child legacy lock policy')
    code=Path(r.get('code_root',''))
    require(code.is_absolute() and code.is_dir() and not code.is_symlink(),'Missing code directory')
    require(within(code,site['code_roots']),'Code directory outside configured roots')
    entry=Path(r.get('entrypoint',''))
    require(not entry.is_absolute() and '..' not in entry.parts and entry.suffix=='.py','Unsafe entrypoint')
    expected=r.get('code_files')
    require(isinstance(expected,dict) and expected,'Code fingerprint required')
    require(expected==snapshot(code),'Code snapshot differs; regenerate and review the request')
    require(entry.as_posix() in expected,'Entrypoint not fingerprinted')
    argv=r.get('argv')
    require(isinstance(argv,list) and len(argv)>=2 and all(isinstance(v,str) for v in argv),'argv must be a string array')
    require(argv[0]=='{python}' and '{entrypoint}' in argv and '{output}' in argv,'Use {python}, {entrypoint} and {output}; shell entrypoints are unsupported')
    for value in argv:
        for token in re.findall(r'\{([^{}]+)\}',value):
            require(token in {'python','entrypoint','output','gpu','gpus'},'Unknown command token: '+token)
    require(Path(site['python']).is_file(),'Configured Python does not exist')
    inputs=r.get('inputs',{})
    require(isinstance(inputs,dict),'inputs must map names to path and SHA256')
    if r['gpu_count']:
        require('dataset' in inputs and 'model_contract' in inputs,'GPU experiments require dataset and model_contract fingerprints')
    for key,item in inputs.items():
        path=Path(item['path'])
        require(path.is_file() and not path.is_symlink() and within(path,site['input_roots']),'Input outside configured roots: '+key)
        require(sha(path)==item['sha256'],'Input fingerprint changed: '+key)
    candidates=r.get('gpu_candidates',site.get('gpu_ids',list(range(8))))
    require(isinstance(candidates,list) and all(isinstance(x,int) and x>=0 for x in candidates),'Invalid GPU candidates')
    require(len(candidates)==len(set(candidates)) and set(candidates)<=set(site.get('gpu_ids',candidates)),'GPU candidates outside site configuration')
    require(r['gpu_count']<=len(candidates),'Too few GPU candidates')
    # Identity omits the owner, ticket, human-readable ID and resource time budget.
    identity={k:r[k] for k in ['method','model','dataset','protocol_id','seed','commit','code_files','entrypoint','inputs','argv']}
    digest=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    return {'identity':digest,'request':r,'code_root':str(code.resolve()),'validated_at':now()}

class FileLease:
    def __init__(self,path):
        self.path=Path(path);self.stream=None
    def __enter__(self):
        self.path.parent.mkdir(parents=True,exist_ok=True)
        require(not self.path.is_symlink(),'Refusing linked lock')
        self.stream=self.path.open('a+b')
        if self.path.stat().st_size==0:
            self.stream.write(b'0');self.stream.flush()
        self.stream.seek(0)
        try:
            if os.name=='nt':
                import msvcrt
                msvcrt.locking(self.stream.fileno(),msvcrt.LK_NBLCK,1)
            else:
                import fcntl
                fcntl.flock(self.stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except OSError:
            self.stream.close();self.stream=None
            raise LabError('Already leased: '+str(self.path)) from None
        return self
    def __exit__(self,*args):
        if self.stream:
            self.stream.close();self.stream=None

def probe_gpus():
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used','--format=csv,noheader,nounits'],text=True,timeout=10)
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True,timeout=10)
    busy={line.split(',')[0].strip() for line in apps.splitlines() if ',' in line}
    rows=[]
    for line in raw.splitlines():
        index,uuid,memory=[x.strip() for x in line.split(',')]
        rows.append({'index':int(index),'memory_mib':int(memory),'busy':uuid in busy})
    return rows

@contextmanager
def reserve_gpus(request, site, probe=None):
    count=request['gpu_count']
    if not count:
        yield [],[]
        return
    probe=probe or probe_gpus
    candidates=request.get('gpu_candidates',site.get('gpu_ids',list(range(8))))
    devices=probe()
    available={r['index'] for r in devices if not r['busy'] and r['memory_mib']<site.get('max_idle_memory_mib',128)}
    for group in itertools.combinations([g for g in candidates if g in available],count):
        stack=ExitStack();leases=[]
        try:
            for gpu in group:
                leases.append(stack.enter_context(FileLease(Path(site['root'])/'locks'/('gpu-'+str(gpu)+'.lock'))))
            # Probe old lock domains under our central lease. Old code that holds
            # its own locks must acquire them itself; wrapper mode retains them.
            old=ExitStack()
            try:
                for gpu in group:
                    for pattern in site.get('legacy_lock_paths',[]):
                        lease=old.enter_context(FileLease(pattern.replace('{gpu}',str(gpu))))
                        if request['legacy_lock_policy']=='wrapper':
                            leases.append(lease)
                if request['legacy_lock_policy']=='wrapper':
                    stack.enter_context(old.pop_all())
            finally:old.close()
            # Recheck actual usage after locks, not just the earlier snapshot.
            live={r['index']:r for r in probe()}
            require(all(g in live and not live[g]['busy'] and live[g]['memory_mib']<site.get('max_idle_memory_mib',128) for g in group),'GPU became busy')
        except (LabError,OSError):
            stack.close()
            continue
        with stack:
            yield list(group),leases
        return
    raise LabError('No idle compatible GPU lease available')

def claim(request,site,digest):
    root=Path(site['root'])
    require(not root.is_symlink(),'Refusing linked registry root')
    root.mkdir(parents=True,exist_ok=True)
    with closing(sqlite3.connect(str(root/'registry.sqlite'),timeout=10)) as db, db:
        db.execute('CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, identity TEXT UNIQUE NOT NULL, created TEXT NOT NULL)')
        try:
            db.execute('INSERT INTO jobs VALUES(?,?,?)',(request['experiment_id'],digest,now()))
        except sqlite3.IntegrityError:
            raise LabError('Experiment ID or scientific identity already registered; inspect status instead of rerunning') from None
    directory=root/'runs'/request['experiment_id']
    directory.mkdir(parents=True,exist_ok=False)
    return directory

def metric_values(value,prefix=''):
    result={}
    if isinstance(value,dict):
        for key,item in value.items():
            name=(prefix+'.'+str(key)).strip('.')
            if isinstance(item,dict):
                result.update(metric_values(item,name))
            elif isinstance(item,(float,int)) and not isinstance(item,bool) and math.isfinite(float(item)):
                result[name]=float(item)
    return result

def post(uri,endpoint,payload):
    req=urllib.request.Request(uri.rstrip('/')+'/api/2.0/mlflow/'+endpoint,
        data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'},method='POST')
    with urllib.request.urlopen(req,timeout=3) as response:
        raw=response.read()
        return json.loads(raw) if raw else {}

def sync_run(directory,site):
    directory=Path(directory);state=read(directory/'state.json')
    require(state['status'] in {'SUCCEEDED','FAILED','CANCELLED','TIMED_OUT'},'Job is not finished')
    if state.get('mlflow_sync')=='SYNCED':return state
    if not site.get('tracking_uri'):
        state['mlflow_sync']='DISABLED';write(directory/'state.json',state);return state
    try:
        uri=site['tracking_uri']
        experiment='Team-experiments'
        query=urllib.parse.urlencode({'experiment_name':experiment})
        try:
            with urllib.request.urlopen(uri.rstrip('/')+'/api/2.0/mlflow/experiments/get-by-name?'+query,timeout=3) as response:
                eid=json.load(response)['experiment']['experiment_id']
        except urllib.error.HTTPError as exc:
            if exc.code!=404:raise
            eid=post(uri,'experiments/create',{'name':experiment})['experiment_id']
        r=read(directory/'request.json')
        rid=state.get('mlflow_run_id')
        if not rid:
            found=post(uri,'runs/search',{'experiment_ids':[eid],'filter':"tags.lab_job_id = '"+r['experiment_id']+"'",'max_results':2}).get('runs',[])
            require(len(found)<2,'Duplicate MLflow job identity')
            if found:rid=found[0]['info']['run_id']
            else:
                info=post(uri,'runs/create',{'experiment_id':eid,'start_time':state['started_ms'],
                    'tags':[{'key':'lab_job_id','value':r['experiment_id']},
                            {'key':'mlflow.runName','value':r['experiment_id']},
                            {'key':'owner','value':r['owner']},{'key':'issue','value':r['issue']},
                            {'key':'output_dir','value':state['output_dir']}]})
                rid=info['run']['info']['run_id']
            state['mlflow_run_id']=rid;write(directory/'state.json',state)
        metrics=state.get('metrics',{})
        post(uri,'runs/log-batch',{'run_id':rid,
            'params':[{'key':k,'value':str(r[k])} for k in ['method','model','dataset','protocol_id','seed','commit']],
            'metrics':[{'key':k,'value':v,'timestamp':state['finished_ms'],'step':0} for k,v in metrics.items()],
            'tags':[{'key':'execution_status','value':state['status']},
                    {'key':'quality_status','value':state.get('quality_status','NOT_REPORTED')},
                    {'key':'identity','value':state['identity']}]})
        post(uri,'runs/update',{'run_id':rid,'status':'FINISHED' if state['status']=='SUCCEEDED' else 'FAILED','end_time':state['finished_ms']})
        state.update(mlflow_sync='SYNCED',mlflow_run_id=rid)
        state.pop('mlflow_error',None)
    except Exception as exc:
        state.update(mlflow_sync='PENDING',mlflow_error=type(exc).__name__+': '+str(exc)[:300])
    write(directory/'state.json',state)
    return state

def run_request(request,site):
    plan=validate(request,site)
    directory=claim(request,site,plan['identity'])
    write(directory/'request.json',request)
    state={'experiment_id':request['experiment_id'],'identity':plan['identity'],'status':'QUEUED',
           'owner':request['owner'],'issue':request['issue'],'created_at':now(),
           'output_dir':str(directory/'output'),'mlflow_sync':'NOT_ATTEMPTED',
           'started_ms':int(time.time()*1000)}
    write(directory/'state.json',state)
    process=None;stop=False;old_handlers={};started=time.monotonic()
    def stop_handler(signum,frame):
        nonlocal stop
        stop=True
    for sig in ([signal.SIGTERM,signal.SIGINT] if os.name!='nt' else [signal.SIGINT]):
        old_handlers[sig]=signal.signal(sig,stop_handler)
    try:
        while True:
            if stop:
                raise LabError('Stopped while waiting for GPU')
            stack=reserve_gpus(request,site)
            try:
                gpus,leases=stack.__enter__()
                break
            except LabError:
                if time.monotonic()-started>=request['queue_timeout_seconds']:raise
                time.sleep(site.get('poll_seconds',5))
        try:
            # Revalidate after any queue delay, before launching the child.
            validate(request,site)
            mapping={'python':site['python'],'entrypoint':str(Path(request['code_root'])/request['entrypoint']),
                     'output':state['output_dir'],'gpus':','.join(map(str,gpus)),
                     'gpu':str(gpus[0]) if len(gpus)==1 else ''}
            argv=[v.format(**mapping) for v in request['argv']]
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=mapping['gpus'],PYTHONDONTWRITEBYTECODE='1',
                     TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
                     OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',LAB_EXPERIMENT_ID=request['experiment_id'])
            state.update(status='RUNNING',gpus=gpus,argv=argv,started_at=now())
            write(directory/'state.json',state)
            options={'start_new_session':True} if os.name!='nt' else {}
            if os.name!='nt':
                options['pass_fds']=tuple(lease.stream.fileno() for lease in leases)
            with (directory/'console.log').open('xb') as log:
                process=subprocess.Popen(argv,cwd=request['code_root'],env=env,stdin=subprocess.DEVNULL,
                                         stdout=log,stderr=subprocess.STDOUT,**options)
                state['child_pid']=process.pid;write(directory/'state.json',state)
                begin=time.monotonic();termination=None
                while process.poll() is None:
                    if termination is None and (stop or time.monotonic()-begin>request['timeout_seconds']):
                        termination='CANCELLED' if stop else 'TIMED_OUT'
                        state['status']='STOPPING';write(directory/'state.json',state)
                        if os.name=='nt':process.terminate()
                        else:os.killpg(process.pid,signal.SIGTERM)
                    time.sleep(0.1)
                state.update(returncode=process.returncode,
                             status=termination or ('SUCCEEDED' if process.returncode==0 else 'FAILED'),
                             elapsed_seconds=time.monotonic()-begin)
        finally:
            # Do not release leases while our child continues after a controller error.
            if process is not None and process.poll() is None:
                if os.name=='nt':process.terminate()
                else:os.killpg(process.pid,signal.SIGTERM)
                process.wait()
            stack.__exit__(None,None,None)
    except Exception as exc:
        state.update(status='CANCELLED' if stop else 'FAILED',error=type(exc).__name__+': '+str(exc)[:500])
    finally:
        for sig,handler in old_handlers.items():signal.signal(sig,handler)
    state.update(finished_at=now(),finished_ms=int(time.time()*1000),quality_status='NOT_REPORTED')
    metrics_file=Path(state['output_dir'])/'metrics.json'
    if metrics_file.is_file():
        try:
            metrics=read(metrics_file)
            state['metrics']=metric_values(metrics)
            state['quality_status']=str(metrics.get('quality_status','NOT_REPORTED'))
            state['metrics_sha256']=sha(metrics_file)
        except Exception as exc:
            state['metrics_warning']=str(exc)[:300]
    write(directory/'state.json',state)
    return sync_run(directory,site)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--site',type=Path,default=Path('/data/chuwenjing/lab/site.json'))
    sub=parser.add_subparsers(dest='action',required=True)
    for name in ['plan','run']:
        p=sub.add_parser(name);p.add_argument('request',type=Path)
    for name in ['status','sync']:
        p=sub.add_parser(name);p.add_argument('experiment_id',nargs='?')
    p=sub.add_parser('fingerprint');p.add_argument('code_root',type=Path)
    args=parser.parse_args()
    if args.action=='fingerprint':
        print(json.dumps(snapshot(args.code_root),ensure_ascii=False,indent=2));return
    site=read(args.site)
    if args.action in {'plan','run'}:
        r=read(args.request)
        value=validate(r,site) if args.action=='plan' else run_request(r,site)
    else:
        if args.experiment_id:
            require(bool(re.fullmatch(r'[a-z0-9][a-z0-9-]{2,79}',args.experiment_id)),'Invalid experiment_id')
            directory=Path(site['root'])/'runs'/args.experiment_id
            value=sync_run(directory,site) if args.action=='sync' else read(directory/'state.json')
        else:
            require(args.action=='status','sync requires experiment_id')
            value=[read(p) for p in sorted((Path(site['root'])/'runs').glob('*/state.json'))]
    print(json.dumps(value,ensure_ascii=False,indent=2))
    if args.action=='run' and value['status']!='SUCCEEDED':raise SystemExit(1)

if __name__=='__main__':
    try:main()
    except LabError as exc:
        print(json.dumps({'error':str(exc)},ensure_ascii=False),file=sys.stderr)
        raise SystemExit(2)
