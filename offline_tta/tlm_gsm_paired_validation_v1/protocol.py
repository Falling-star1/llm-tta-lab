"""Blind selection and conservative paired checks for inference-only validation."""
import random
import re
from pathlib import Path,PurePosixPath

SEED=20261027


def select_indices(total,excluded,count=200,seed=SEED):
    if type(total) is not int or total<1 or type(count) is not int or count<1 or type(seed) is not int or seed<0:
        raise ValueError('Invalid population, count or seed')
    if not isinstance(excluded,list) or any(type(i) is not int or not 0<=i<total for i in excluded) or len(set(excluded))!=len(excluded):
        raise ValueError('Invalid exclusion indexes')
    pool=sorted(set(range(total))-set(excluded))
    if len(pool)<count:raise ValueError('Not enough uninspected cases')
    return sorted(random.Random(seed).sample(pool,count))


def is_stub(text):
    return bool(re.fullmatch(r"\s*(?:#{1,6}\s*Response:\s*)?Let[’']s think step by step\.?\s*",text,re.I))


def paired_counts(baseline,control):
    def check(rows):
        if not rows or any(type(r.get('source_index')) is not int or r['source_index']<0 or
                           type(r.get('score')) not in (int,float) or r['score'] not in [0,1] or
                           type(r.get('stub_only')) is not bool for r in rows):
            raise ValueError('Invalid paired observation')
        ids=[r['source_index'] for r in rows]
        if len(ids)!=len(set(ids)) or ids!=sorted(ids):raise ValueError('Invalid paired indexes')
        return ids
    if check(baseline)!=check(control):raise ValueError('Paired indexes differ')
    return {'n':len(baseline),'baseline_correct':sum(int(r['score']) for r in baseline),'control_correct':sum(int(r['score']) for r in control),
            'wrong_to_right':sum(b['score']<c['score'] for b,c in zip(baseline,control)),
            'right_to_wrong':sum(b['score']>c['score'] for b,c in zip(baseline,control)),
            'both_correct':sum(b['score']==c['score']==1 for b,c in zip(baseline,control)),
            'both_wrong':sum(b['score']==c['score']==0 for b,c in zip(baseline,control)),
            'stub_removed':sum(b['stub_only'] and not c['stub_only'] for b,c in zip(baseline,control)),
            'stub_introduced':sum(c['stub_only'] and not b['stub_only'] for b,c in zip(baseline,control))}


def require_control(summary,manifest,mode,original):
    expected={'status':'COMPLETE','mode':mode,'steps':2500,'diagnostic_only':True,'lr':1e-6,'original_runs_modified':False}
    if mode not in ['hint_masked','eot_masked','both_masked'] or any(summary.get(k)!=v for k,v in expected.items()):
        raise ValueError('Incorrect completed control identity or recipe')
    if manifest.get('mode')!=mode or manifest.get('original')!=original or manifest.get('fresh_initial_adapter_exact') is not True:
        raise ValueError('Incorrect control provenance')
    proof=summary.get('controls',{});opt=summary.get('optimizer',{})
    if (proof.get('sample_order_exact') is not True or proof.get('learning_rates_exact') is not True or
        proof.get('max_base_ce_difference')!=0 or proof.get('max_weight_difference')!=0 or
        opt.get('successful_optimizer_steps')!=2500 or opt.get('skipped_optimizer_steps')!=0 or
        opt.get('adapter_parameters_finite') is not True or opt.get('optimizer_state_finite') is not True):
        raise ValueError('Control trajectory was not verified')


def artifact_paths(reg):
    original=PurePosixPath(reg['original']);prior=PurePosixPath(reg['replay_root']);control=PurePosixPath(reg['control_root'])
    inputs=PurePosixPath(reg['inputs'])
    paths=[original/name for name in ['run_manifest.json','metrics.json','tlm.jsonl','input_tokens.json','resolved_config.json',
           'source_reuse.json','training_complete.json','final_adapter/adapter_model.safetensors','trainer/milestones/step-2500/adapter_model.safetensors']]
    paths.extend(inputs/name for name in ['manifest.json','inputs.jsonl','references.jsonl']);paths.append(PurePosixPath(reg['source_path']))
    for mode in ['original_step2500','hint_masked','eot_masked','both_masked']:
        paths.extend(prior/mode/name for name in ['answers.json','summary.json','selection.json'])
        if mode!='original_step2500':
            paths.extend(control/mode/name for name in ['summary.json','manifest.json','final_adapter/adapter_model.safetensors','trainer/checkpoint-2500/commit.json'])
    if any(not p.is_absolute() or '..' in p.parts for p in paths):raise ValueError('Asset paths must be absolute canonical paths')
    return {str(p) for p in paths}


def require_assets(assets,required):
    if not isinstance(assets,dict) or not required or not set(required)<=set(assets) or any(
            not isinstance(value,str) or re.fullmatch(r'[0-9a-f]{64}',value) is None for value in assets.values()):
        raise ValueError('Incomplete required artifact registration or invalid hashes')


def protect_output(output,protected):
    if any(Path(output).resolve().is_relative_to(Path(root).resolve()) for root in protected):
        raise ValueError('Output is inside a protected source, input, code or experiment tree')


def replay_references(rows,expected_indexes):
    if (len(expected_indexes)<2 or len(expected_indexes)!=len(set(expected_indexes)) or expected_indexes!=sorted(expected_indexes)
        or len(rows)!=len(expected_indexes) or [r.get('source_index') for r in rows]!=expected_indexes):
        raise ValueError('Incomplete or mismatched replay references')
    for row in rows:
        ids=row.get('generated_ids')
        if (type(row.get('source_index')) is not int or not isinstance(row.get('predict'),str) or not isinstance(ids,list)
            or not ids or any(type(token) is not int or token<0 for token in ids)):
            raise ValueError('Invalid saved generation reference')
    return rows[:2]
