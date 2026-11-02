"""FP32 head diagnostic only. No optimizer updates or formal receipts."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

V4='f1b5c7026a9429c4008b923a42b459ae47070585d51ec28b70498806a85a459b'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--code',type=Path,required=True)
    parser.add_argument('--model',choices=['llama3b','qwen3b'],required=True)
    parser.add_argument('--dataset',choices=['geosignal_5k','dolly_5k','gsm8k_5k'],required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():parser.error('Use a new output')
    sys.path.insert(0,str(args.code))
    from runtime import code_identity,LoadedModel,specifications,read_inputs,atomic_json,gpu_lock,frozen_digest
    assert code_identity()==V4
    from launch import verify_delivery
    verify_delivery(args.code)
    from run import probe_training_arguments,PROBE_LAMB,precision_feature
    from train_core import TLMTrainer,InputCollator
    from precision import compare_precision_paths
    from tlm_loss import model_input_cross_entropy,weighted_input_loss
    from projection import fp32_head
    import torch
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
    os.environ.setdefault('HF_HUB_OFFLINE','1')
    os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
    result={'model':args.model,'dataset':args.dataset,'code_id':V4,
            'diagnostic_only':True,'not_equivalent_to_original_fp16_numerics':True,
            'ready_for_formal':False,'optimizer_steps':0,'policy':'FP32 frozen head; FP16 decoder; original LoRA dtype'}
    with gpu_lock():
        args.output.mkdir(parents=True)
        try:
            _,spec=specifications(args.model,args.dataset)
            rows,_=read_inputs(Path('/data/chuwenjing/tta_offline/tlm_inputs_v1')/args.dataset,spec['sha256'])
            loaded=LoadedModel(args.model,args.dataset,42,initialization='gaussian')
            encoded=loaded.tokens(rows)
            base=loaded.model.get_base_model()
            head=base.get_output_embeddings()
            trainer=TLMTrainer(model=loaded.model,args=probe_training_arguments(args.output,42),
                train_dataset=[{'source_index':0,'input_ids':encoded[0][0]}],data_collator=InputCollator(loaded.pad_id),
                threshold=0.,lamb=PROBE_LAMB,chunk_size=128)
            feature=precision_feature(encoded,128)
            # Isolate raw logits and dCE/dhidden, with exactly the same hidden states.
            loaded.model.eval()
            ids=torch.tensor([feature['input_ids']],device=loaded.device)
            with torch.no_grad(),trainer.accelerator.autocast():
                hidden=base.model(input_ids=ids,use_cache=False,return_dict=True).last_hidden_state[:,:-1].detach()
            labels=ids[:,1:]
            def isolated():
                values=[]
                for chunk in [hidden.shape[1],128]:
                    h=hidden.clone().requires_grad_(True)
                    pieces=[]
                    with trainer.accelerator.autocast():
                        for start in range(0,h.shape[1],chunk):
                            state=h[:,start:start+chunk]
                            if chunk==128 and state.shape[1]<chunk:
                                state=torch.nn.functional.pad(state,(0,0,0,chunk-state.shape[1]))
                            pieces.append(head(state)[:,:min(chunk,h.shape[1]-start)].float())
                    logits=torch.cat(pieces,dim=1)
                    loss=torch.nn.functional.cross_entropy(logits.reshape(-1,logits.shape[-1]),labels.reshape(-1))
                    loss.backward()
                    values.append((logits.detach().cpu(),h.grad.detach().cpu(),float(loss)))
                output={}
                for index,name in [(0,'logits'),(1,'hidden_gradient')]:
                    a,b=values[0][index],values[1][index]
                    output[name]={'max_abs':float((a-b).abs().max()),
                                  'outside_tolerance':int((~torch.isclose(a,b,atol=1e-5,rtol=1e-2)).sum())}
                output['losses']=[v[2] for v in values]
                return output
            result['isolated_fp16']=isolated()
            torch.cuda.reset_peak_memory_stats()
            started=time.monotonic()
            with fp32_head(head):
                result['isolated_fp32']=isolated()
                loaded.model.zero_grad(set_to_none=True)
                result['precision']=compare_precision_paths(trainer,feature)
                # Real longest input; no optimizer step and no full-vocabulary long reference.
                loaded.model.zero_grad(set_to_none=True)
                longest=max(range(len(encoded)),key=lambda i:len(encoded[i][0]))
                longids=torch.tensor([encoded[longest][0]],device=loaded.device)
                loaded.model.eval()
                with torch.no_grad(),loaded.model.disable_adapter(),trainer.accelerator.autocast():
                    basece=model_input_cross_entropy(loaded.model,longids,chunk_size=128)
                loaded.model.train()
                with trainer.accelerator.autocast():
                    ce=model_input_cross_entropy(loaded.model,longids,chunk_size=128)
                    loss=weighted_input_loss(ce,basece,threshold=0.,lamb=PROBE_LAMB)
                loss.backward()
                result['longest']={'source_index':longest,'tokens':longids.shape[1],
                    'all_gradients_finite':all(p.grad is not None and bool(torch.isfinite(p.grad).all())
                        for p in loaded.model.parameters() if p.requires_grad),
                    'base_ce':float(basece),'current_ce':float(ce.detach())}
                result['peak_allocated_mib']=torch.cuda.max_memory_allocated()/1024**2
                result['peak_reserved_mib']=torch.cuda.max_memory_reserved()/1024**2
            loaded.model.zero_grad(set_to_none=True)
            result['base_unchanged']=frozen_digest(loaded.model)==loaded.contract['frozen_sha256']
            result['seconds']=time.monotonic()-started
            result['status']='PASS' if result['precision']['status']=='PASS' and result['longest']['all_gradients_finite'] and result['base_unchanged'] else 'FAIL'
        except BaseException as error:
            result['status']='FAIL';result['error']={'type':type(error).__name__,'message':str(error)}
            raise
        finally:
            atomic_json(args.output/'report.json',result)
            print(json.dumps({k:v for k,v in result.items() if k!='precision'},allow_nan=False),flush=True)


if __name__=='__main__':main()
