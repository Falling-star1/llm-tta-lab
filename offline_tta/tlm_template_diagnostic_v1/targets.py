"""Loss-position controls; input tokens and frozen-base selection never change."""
import re

PREFIX='Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n'
SUFFIX='\n\n### Response:\n'
MODES=('all','role_masked','wrapper_masked','both_masked')


def target_groups(rendered,offsets,user_end,truncated):
    start=rendered.find(PREFIX)
    if start<0 or not start+len(PREFIX)<user_end<=len(rendered):
        raise ValueError('Canonical Dolly scaffold or user boundary missing')
    body_start=start+len(PREFIX)
    if rendered[start:user_end].endswith(SUFFIX):
        body_end=user_end-len(SUFFIX)
    elif truncated:
        body_end=user_end
    else:
        raise ValueError('Canonical Dolly response suffix missing')
    if body_end<=body_start:raise ValueError('No body targets')
    groups=[]
    for a,b in offsets:
        if not 0<=a<=b<=len(rendered):raise ValueError('Invalid token offsets')
        if b>body_start and a<body_end:groups.append('body')
        elif b>start and a<user_end:groups.append('wrapper')
        else:groups.append('role')
    return groups


def loss_mask(groups,mode):
    if mode not in MODES or not set(groups)<={'role','wrapper','body'}:
        raise ValueError('Unknown loss mode or target group')
    excluded={'all':set(),'role_masked':{'role'},'wrapper_masked':{'wrapper'},
              'both_masked':{'role','wrapper'}}[mode]
    return [g not in excluded for g in groups]


def title_only(text):
    return bool(re.fullmatch(r'\s*(?:#{1,6}[ \t]*(?:response|answer)[ \t]*:[ \t\r\n]*)+',text,re.I))
