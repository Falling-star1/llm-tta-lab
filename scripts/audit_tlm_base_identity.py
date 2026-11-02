"""Read-only fresh-load tensor audit against the pinned safetensors files."""
import argparse,hashlib,json,os,sys
from pathlib import Path
parser=argparse.ArgumentParser();parser.add_argument('--output',required=True,type=Path);args=parser.parse_args()
sys.path.insert(0,'/home/chuwenjing/TTA_LLM/TLM/local_repro/tlm_offline_b6da067b55f5')
from runtime import LoadedModel,atomic_json,gpu_lock,code_identity
import torch
from safetensors import safe_open
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
with gpu_lock():
 args.output.mkdir(parents=True,exist_ok=False)
 loaded=LoadedModel('qwen3b','dolly_5k',42,initialization='gaussian')
 root=Path(loaded.profile['model']);weight_map=json.loads((root/'model.safetensors.index.json').read_text())['weight_map']
 result={'code_id':code_identity(),'physical_gpu':os.environ['CUDA_VISIBLE_DEVICES'],
   'python_hash_seed':os.environ.get('PYTHONHASHSEED'),'contract':loaded.contract,
   'parameters':[],'not_a_preflight':True,'optimizer_steps':0}
 for name,p in loaded.model.named_parameters():
  if 'lora_' in name:continue
  key=name.removeprefix('base_model.model.').replace('.base_layer.','.')
  if key=='lm_head.weight' and key not in weight_map:key='model.embed_tokens.weight'
  assert key in weight_map,key
  actual_sha=hashlib.sha256();expected_sha=hashlib.sha256();different=0;first=[]
  with safe_open(str(root/weight_map[key]),framework='pt',device='cpu') as store:
   expected=store.get_tensor(key).reshape(-1);flat=p.detach().reshape(-1)
   for start in range(0,p.numel(),1024*1024):
    got=flat[start:start+1024*1024].cpu().contiguous()
    ref=expected[start:start+1024*1024].to(dtype=p.dtype).contiguous()
    actual_sha.update(got.view(torch.uint8).numpy().tobytes());expected_sha.update(ref.view(torch.uint8).numpy().tobytes())
    if not torch.equal(got,ref):
     mask=got!=ref;different+=int(mask.sum())
     if len(first)<5:
      for i in mask.nonzero().flatten()[:5-len(first)].tolist():
       first.append({'index':start+i,'loaded':float(got[i]),'file_cast':float(ref[i])})
  result['parameters'].append({'name':name,'file_key':key,'shape':list(p.shape),'dtype':str(p.dtype),
   'loaded_sha256':actual_sha.hexdigest(),'file_fp16_sha256':expected_sha.hexdigest(),
   'different_elements':different,'examples':first})
 base=loaded.model.get_base_model();embed=base.get_input_embeddings().weight;head=base.get_output_embeddings().weight
 result['embedding_head_tied']=embed is head
 result['frozen_parameter_count']=len(result['parameters'])
 result['canonical_parameter_hash']=hashlib.sha256(json.dumps(sorted(result['parameters'],key=lambda r:r['name']),sort_keys=True).encode()).hexdigest()
 result['all_parameters_match_files']=all(v['loaded_sha256']==v['file_fp16_sha256'] for v in result['parameters'])
 atomic_json(args.output/'report.json',result)
 print(json.dumps({'frozen_sha256':loaded.contract['frozen_sha256'],'all_parameters_match_files':result['all_parameters_match_files'],
  'embedding_head_tied':result['embedding_head_tied'],'canonical_parameter_hash':result['canonical_parameter_hash'],
  'different_parameters':[r for r in result['parameters'] if r['different_elements']],
  'parameters':len(result['parameters'])}),flush=True)
