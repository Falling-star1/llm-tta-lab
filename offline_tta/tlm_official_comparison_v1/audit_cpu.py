"""Read-only comparison with the pinned official processor and loss methods."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                  TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='2')
import hashlib
import json
from pathlib import Path
import sys
from datetime import datetime, timezone

REPO=Path('/home/chuwenjing/TTA_LLM/TLM')
RUNTIME=REPO/'local_repro/tlm_offline_b6da067b55f5'
sys.path[:0]=[str(RUNTIME),str(REPO/'src')]
import torch
torch.set_num_threads(2)
from llamafactory.hparams import ModelArguments, DataArguments
from llamafactory.model.loader import load_tokenizer
from llamafactory.data.template import get_template_and_fix_tokenizer
from llamafactory.data.processor.self_supervised import SelfSupervisedDatasetProcessor
from llamafactory.train.ttl.trainer import CustomSeq2SeqTrainer
from input_tokens import encode_input
from tlm_loss import input_cross_entropy, weighted_input_loss

def emit(value):
    print(json.dumps(value,ensure_ascii=False,allow_nan=False),flush=True)

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

emit({'kind':'environment','time_utc':datetime.now(timezone.utc).isoformat(),
      'read_only':True,'gpu_used':False})
profiles=json.loads((RUNTIME/'profiles.json').read_text())
specs=json.loads((RUNTIME/'datasets.json').read_text())['datasets']
for model,profile in profiles.items():
    tokenizer=load_tokenizer(ModelArguments(model_name_or_path=profile['model'],trust_remote_code=False))['tokenizer']
    data_args=DataArguments(template=profile['template'],cutoff_len=4096)
    template=get_template_and_fix_tokenizer(tokenizer,data_args)
    processor=SelfSupervisedDatasetProcessor(template,tokenizer,None,data_args)
    for spec in specs:
        dataset=spec['alias']
        original=REPO/'data/AdaptEval'/spec['filename']
        assert digest(original)==spec['sha256']
        raw=json.loads(original.read_text())
        prepared=Path('/data/chuwenjing/tta_offline/tlm_inputs_v1')/dataset/'inputs.jsonl'
        rows=[json.loads(s) for s in prepared.read_text().splitlines()]
        assert len(raw)==len(rows)==5000
        examples={'_prompt':[],'_response':[],'_system':[],'_tools':[],
                  '_images':[],'_videos':[],'_audios':[]}
        current=[]
        for i,(reference,row) in enumerate(zip(raw,rows)):
            assert reference['instruction']==row['instruction'] and reference['input']==row['input']
            content=reference['instruction']+ ('\n'+reference['input'] if reference['input'] else '')
            examples['_prompt'].append([{'role':'user','content':content}])
            examples['_response'].append([{'role':'assistant','content':reference['output']}])
            for key in ['_system','_tools','_images','_videos','_audios']:
                examples[key].append(None)
            current.append(encode_input(row,tokenizer,template,None)[0])
        upstream=processor.preprocess_dataset(examples)
        mismatches=[{'source_index':i,'official_tokens':len(a),'current_tokens':len(b)}
                    for i,(a,b) in enumerate(zip(upstream['input_ids'],current)) if a!=b]
        assert len(upstream['input_ids'])==5000
        emit({'kind':'input_comparison','model':model,'dataset':dataset,'n':5000,
              'official_processor_called':True,'raw_dataset_sha256':digest(original),
              'prepared_inputs_sha256':digest(prepared),'equal_token_sequences':5000-len(mismatches),
              'different_token_sequences':len(mismatches),'differences':mismatches,
              'official_training_labels_equal_inputs':upstream['labels']==upstream['input_ids'],
              'scope':'native template and original reference-length truncation; no model forward'})

# The actual upstream methods, with random logits solely to isolate loss algebra.
for batch,padding in [(1,False),(2,True)]:
    torch.manual_seed(20261028+batch)
    initial=torch.randn(batch,9,31)
    ids=torch.randint(0,31,(batch,9))
    mask=torch.ones_like(ids)
    if padding:mask[1,-3:]=0
    labels=ids.masked_fill(mask==0,-100)
    base=torch.tensor([3.25]*batch)
    reference=initial.clone().requires_grad_(True)
    got=initial.clone().requires_grad_(True)
    ce=CustomSeq2SeqTrainer.cal_ce(None,reference,labels)
    kl=CustomSeq2SeqTrainer.cal_kl(None,reference,labels)
    select=base>3
    weight=.1*torch.exp(base-3)
    official=(kl*weight*select).sum()/select.sum().clamp_min(1)
    actual=weighted_input_loss(input_cross_entropy(got,ids,mask),base)
    official.backward();actual.backward()
    emit({'kind':'loss_algebra','batch':batch,'right_padding':padding,
          'official_loss':float(official),'current_loss':float(actual),
          'loss_absolute_difference':abs(float(official-actual)),
          'gradient_max_absolute_difference':float((reference.grad-got.grad).abs().max()),
          'ce_max_absolute_difference':float((ce-input_cross_entropy(got,ids,mask)).abs().max()),
          'scope':'CPU FP32 random logits; not BF16/full training equivalence'})
emit({'kind':'complete','time_utc':datetime.now(timezone.utc).isoformat()})
