"""Separate the reasoning-prefill targets from the following user EOT target."""
HINT="Let's think step by step."
MODES=('hint_masked','eot_masked','both_masked')


def transition_mask(rendered,offsets,eot_index,mode):
    if mode not in MODES or not 1<=eot_index<len(offsets):raise ValueError('Invalid transition control')
    end=offsets[eot_index][0];prefix=rendered[:end].rstrip();start=len(prefix)-len(HINT)
    if not prefix.endswith(HINT) or not rendered[:start].endswith('\n\n### Response: '):
        raise ValueError('Canonical GSM reasoning prefill missing')
    mask=[True]*len(offsets)
    if mode in ['eot_masked','both_masked']:mask[eot_index]=False
    if mode in ['hint_masked','both_masked']:
        for i,(a,b) in enumerate(offsets):
            if b>start and a<len(prefix):mask[i]=False
    return mask
