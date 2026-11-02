"""Keep active evaluations; continue only known output-quality stops."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
from continue_evaluation import read,write,require,sha


def classify(manifest,failure,active,complete):
    if active:return 'ACTIVE'
    if complete and manifest.get('status')=='PASS':return 'COMPLETE'
    if manifest.get('status')=='FAIL' and failure and failure.get('type')=='RuntimeError' and failure.get('message','').startswith('Output quality alert;'):
        return 'RESUME'
    return 'BLOCKED'


def active_outputs():
    outputs={}
    for p in Path('/proc').iterdir():
        if not p.name.isdecimal():continue
        try:
            if p.stat().st_uid!=os.getuid():continue
            values=[s.decode() for s in (p/'cmdline').read_bytes().split(b'\0') if s]
            if 'continue_evaluation.py' in ' '.join(values) and '--run' in values:
                outputs[str(Path(values[values.index('--run')+1]).resolve())]=int(p.name)
            elif '--output' in values and 'formal' in values and any(Path(v).name=='run.py' for v in values):
                outputs[str(Path(values[values.index('--output')+1]).resolve())]=int(p.name)
        except (OSError,UnicodeError,IndexError):continue
    return outputs


def validate_jobs(jobs):
    tasks={(m,d) for m in ['llama3b','qwen3b'] for d in ['geosignal_5k','dolly_5k','gsm8k_5k']}
    require(len(jobs)==6 and {(j['model'],j['dataset']) for j in jobs}==tasks,'Require six registered trained tasks')
    require(len({j['gpu'] for j in jobs})==6 and len({j['run'] for j in jobs})==6,'Task/GPU assignments duplicated')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--jobs',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    jobs=read(args.jobs)['jobs'];validate_jobs(jobs)
    launched={};processes={};attempts=set()
    while True:
        active=active_outputs();states=[]
        for job in jobs:
            run=Path(job['run']);manifest=read(run/'run_manifest.json')
            metrics=read(run/'metrics.json') if (run/'metrics.json').exists() else {}
            failure=read(run/'failure.json') if (run/'failure.json').exists() else None
            continuation_failure=read(run/'evaluation_continuation_failure.json') if (run/'evaluation_continuation_failure.json').exists() else None
            complete=metrics.get('n')==5000 and (run/'tlm.jsonl').exists() and sha(run/'tlm.jsonl')==metrics.get('predictions_sha256')
            key=str(run.resolve());state=classify(manifest,failure,key in active,complete)
            if continuation_failure and state!='COMPLETE':state='BLOCKED'
            if state=='RESUME' and key not in attempts:
                command=[sys.executable,'-B','-u',str(Path(__file__).with_name('continue_evaluation.py')),
                    '--runtime',job['runtime'],'--run',str(run),'--model',job['model'],'--dataset',job['dataset']]
                env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(job['gpu']),CUBLAS_WORKSPACE_CONFIG=':4096:8',
                    OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
                log=args.output/(job['model']+'__'+job['dataset']+'.log')
                with log.open('xb') as stream:
                    process=subprocess.Popen(command,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT)
                processes[key]=process;launched[key]={'pid':process.pid,'log':str(log)};attempts.add(key);state='ACTIVE'
            elif state=='RESUME' and key in attempts:state='BLOCKED'
            if key in processes and processes[key].poll() not in (None,0):state='BLOCKED'
            states.append({**job,'status':state,'continuation':launched.get(key),'existing_pid':active.get(key),
                'predictions_complete':(run/'tlm.jsonl').read_bytes().count(b'\n') if (run/'tlm.jsonl').exists() else 0,
                'original_failure':failure,'continuation_failure':continuation_failure,
                'quality_latest':read(run/'quality_latest.json') if (run/'quality_latest.json').exists() else None,
                'quality_alert_seen':(run/'quality_alert.json').exists()})
        all_complete=all(s['status']=='COMPLETE' for s in states)
        running=any(s['status']=='ACTIVE' for s in states)
        write(args.output/'status.json',{'status':'PASS' if all_complete else ('RUNNING' if running else 'BLOCKED'),
            'stage':'complete' if all_complete else 'evaluation','jobs':states,'quality_policy':'record_and_continue',
            'training_performed':False,'completion_does_not_imply_quality_pass':True,'time':time.time()})
        if all_complete:return
        if not running:raise SystemExit(1)
        time.sleep(20)


if __name__=='__main__':main()
