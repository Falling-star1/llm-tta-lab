"""Audit saved greedy token prefixes; never rewrite a running experiment."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

from protocol import official_stop_ids,decode_official_prefix,answer_flags


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--runtime',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.resolve().is_relative_to(args.run.resolve()):raise ValueError('Audit output must be separate')
    sys.path.insert(0,str(args.runtime))
    from runtime import code_identity,atomic_json,specifications,REPOSITORY
    sys.path.insert(0,str(REPOSITORY/'src'))
    from llamafactory.data import get_template_and_fix_tokenizer
    from llamafactory.hparams import DataArguments
    from transformers import AutoTokenizer
    code='b6da067b55f5c65241ae6fbd8c706b88523f64b27dd9568ad5661255b89d7999'
    if code_identity()!=code:raise ValueError('Frozen greedy generation code changed')
    manifest=json.loads((args.run/'run_manifest.json').read_text())
    if manifest['config']['code_id']!=code:raise ValueError('Unknown generation policy')
    model=manifest['config']['model'];dataset=manifest['config']['dataset']
    profile,_=specifications(model,dataset)
    for item in profile['files']:
        if 'tokenizer' in item['filename']:
            if hashlib.sha256((Path(profile['model'])/item['filename']).read_bytes()).hexdigest()!=item['sha256']:
                raise ValueError('Tokenizer identity changed')
    tokenizer=AutoTokenizer.from_pretrained(profile['model'],local_files_only=True)
    template=get_template_and_fix_tokenizer(tokenizer,DataArguments(template=profile['template']))
    raw=(args.run/'tlm.jsonl').read_bytes();end=len(raw) if raw.endswith(b'\n') else raw.rfind(b'\n')+1
    rows=[json.loads(line) for line in raw[:end].splitlines()]
    if any(r['source_index']!=i for i,r in enumerate(rows)):raise ValueError('Prediction indexes are not contiguous')
    old_flags=[];new_flags=[];changed=[]
    for r in rows:
        if tokenizer.decode(r['generated_ids'],skip_special_tokens=True)!=r['predict']:raise ValueError('Stored text and token IDs differ')
        revised=decode_official_prefix(r['generated_ids'],tokenizer)
        old_flags.append(answer_flags(r['predict']));new_flags.append(answer_flags(revised['predict']))
        if revised['shortened']:changed.append({'source_index':r['source_index'],**revised})
    args.output.mkdir(parents=True,exist_ok=False)
    summary={'model':model,'dataset':dataset,'n':len(rows),'original_run':str(args.run),
        'complete_prefix_sha256':hashlib.sha256(raw[:end]).hexdigest(),'uncommitted_tail_bytes':len(raw)-end,
        'current_stop_ids':template.get_stop_token_ids(tokenizer),'official_stop_ids':official_stop_ids(tokenizer),
        'changed_tlm_rows':len(changed),'old_flags':{k:sum(r[k] for r in old_flags) for k in ['empty','title_only','no_substantive_answer']},
        'official_prefix_flags':{k:sum(r[k] for r in new_flags) for k in ['empty','title_only','no_substantive_answer']},
        'original_files_modified':False,'source_official_policy_verified':False,
        'note':'Source lacks generated token IDs; this audit cannot certify a complete official-policy Source/TLM comparison.'}
    atomic_json(args.output/'summary.json',summary)
    if changed:atomic_json(args.output/'changed_prefixes.json',changed)
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
