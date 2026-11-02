"""Revalidate only blocked Qwen/Dolly; keep all five existing formal runs."""
import argparse,json,os
from pathlib import Path
import subprocess,sys,time

CODE='b6da067b55f5c65241ae6fbd8c706b88523f64b27dd9568ad5661255b89d7999'
EXPECTED_BASE='644c521c2e650bb46c5bfcedea082bbc3a02b9a5c0f67297a44633cb035384bc'
LLAMA_BASE='c80d597af8e0f5bb8e99a6545680edb580ed2fe59c9cba4af0f39b568c15be77'
NAME='qwen3b__dolly_5k'


def require_canonical_base(value):
    contract=value.get('contract',value)
    if contract.get('frozen_sha256')!=EXPECTED_BASE:
        raise ValueError('Qwen base does not match independently verified canonical weights')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime',type=Path,required=True)
    parser.add_argument('--gate',type=Path,required=True)
    parser.add_argument('--original-root',type=Path,required=True)
    parser.add_argument('--preflights',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--gpu',type=int,default=6)
    args=parser.parse_args()
    # Verify the reused controller before importing its validation functions.
    import hashlib
    if hashlib.sha256((args.gate/'gate.py').read_bytes()).hexdigest()!='6899118e59c9f8f3a2deee4c6b4c633db4e46d8b00038e9c4b692e19cb9b853b':
        raise ValueError('Use the independently reviewed frozen controller')
    for row in (args.gate/'SHA256SUMS').read_text().splitlines():
        digest,name=row.split(maxsplit=1);path=(args.gate/name).resolve()
        if not path.is_relative_to(args.gate.resolve()) or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
            raise ValueError('Frozen controller changed')
    sys.path.insert(0,str(args.gate))
    from gate import read,write,sha,require,verify_code,validate_preflight,validate_pilot,run_barriers
    verify_code(args.runtime,CODE)
    args.output.mkdir(parents=True,exist_ok=False)
    inputs=Path('/data/chuwenjing/tta_offline/tlm_inputs_v1/dolly_5k')
    prepared=read(inputs/'manifest.json')
    expected=dict(code_id=CODE,model='qwen3b',dataset='dolly_5k',seed=42,lora_init='gaussian',
                  chunk_size=128,input_sha256=prepared['files']['inputs.jsonl'])
    receipt=args.output/'preflight'/NAME/'receipt.json'
    receipt_hash=None
    registry=args.original_root/'recovery_qwen3b__dolly_5k.json'

    def verify_other_five():
        for model in ['llama3b','qwen3b']:
            for dataset in ['geosignal_5k','dolly_5k','gsm8k_5k']:
                name=model+'__'+dataset
                if name==NAME:continue
                r=read(args.preflights/name/'receipt.json')
                base=LLAMA_BASE if model=='llama3b' else EXPECTED_BASE
                data=read(inputs.parent/dataset/'manifest.json')
                validate_preflight(r,dict(code_id=CODE,model=model,dataset=dataset,seed=42,
                    lora_init='gaussian',chunk_size=128,input_sha256=data['files']['inputs.jsonl']))
                require(r['code_id']==CODE and r['status']=='PASS' and r['contract']['frozen_sha256']==base,
                        'An existing task has a different base/code identity')
                contract=read(args.original_root/'formal'/name/'model_contract.json')
                require(contract==r['contract'] and contract['frozen_sha256']==base,'Existing formal model contract changed')

    def before_training():
        verify_code(args.runtime,CODE);verify_other_five()
        require(sha(receipt)==receipt_hash,'Recovery receipt changed')
        r=read(receipt);validate_preflight(r,expected);require_canonical_base(r)
        require(all(sha(inputs/name)==digest for name,digest in prepared['files'].items()),'Prepared data changed')

    def stage(name):
        nonlocal receipt_hash
        if name!='preflight':before_training()
        if name=='formal':
            validate_pilot(args.output/'pilot'/NAME,expected,receipt)
            require_canonical_base(read(args.output/'pilot'/NAME/'model_contract.json'))
            write(args.output/'ready_for_formal.json',{'status':'PASS','code_id':CODE,
                'canonical_base':EXPECTED_BASE,'receipt_sha256':receipt_hash,
                'fresh_initialization':'base; Gaussian A/zero B; seed42','time':time.time()})
        out=args.output/name/NAME;log=args.output/(NAME+'.'+name+'.log')
        write(args.output/'status.json',{'status':'RUNNING','stage':name,'time':time.time()})
        command=[sys.executable,'-B','-u',str(args.runtime/'run.py'),name,'--model','qwen3b','--dataset','dolly_5k',
            '--lora-init','gaussian','--seed','42','--chunk-size','128','--inputs',str(inputs),'--output',str(out)]
        if name!='preflight':command+=['--receipt',str(receipt)]
        if name=='pilot':command+=['--samples','100']
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(args.gpu),CUBLAS_WORKSPACE_CONFIG=':4096:8',
            OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false',
            HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1')
        with log.open('xb') as stream:
            result=subprocess.run(command,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT)
        require(result.returncode==0,name+' failed; results retained: '+str(log))
        if name=='preflight':
            r=read(receipt);validate_preflight(r,expected);require_canonical_base(r);receipt_hash=sha(receipt)
        else:
            require_canonical_base(read(out/'model_contract.json'))
            require(read(out/'run_manifest.json')['status']=='PASS','Task did not complete')
            if name=='pilot':validate_pilot(out,expected,receipt)
            else:require(read(out/'metrics.json')['n']==5000,'Formal result incomplete')
        return True

    try:
        verify_other_five()
        old=args.original_root/'formal'/NAME
        require(read(old/'run_manifest.json')['status']=='FAIL' and not (old/'training_complete.json').exists()
                and not list(old.glob('trainer/checkpoint-*')),'Original failed launch is no longer an empty blocked run')
        require(not registry.exists(),'Recovery already registered; inspect instead of relaunching')
        write(registry,{'status':'REGISTERED','code_id':CODE,'canonical_base':EXPECTED_BASE,
            'recovery_root':str(args.output),'gpu':args.gpu,'retired_formal':str(old),
            'retired_receipt':str(args.preflights/NAME/'receipt.json'),
            'retired_receipt_sha256':sha(args.preflights/NAME/'receipt.json'),
            'reason':'Historical Dolly base identity is unexplained and noncanonical; preserve, do not reuse',
            'preserved_existing_formal_tasks':5,'time':time.time()})
        run_barriers(stage)
        write(args.output/'status.json',{'status':'PASS','stage':'complete','time':time.time()})
    except BaseException as error:
        state=read(args.output/'status.json') if (args.output/'status.json').exists() else {}
        write(args.output/'status.json',{**state,'status':'BLOCKED','error':str(error),'time':time.time()})
        raise


if __name__=='__main__':main()
