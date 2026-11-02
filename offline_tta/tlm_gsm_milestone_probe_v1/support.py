"""Fixed probes and conservative descriptive output classification."""
from pathlib import Path
import random
import re


def probe_indices():
    return list(range(20))+sorted(random.Random(271828).sample(range(20,5000),80))


def reasoning_stub_only(text):
    return bool(re.fullmatch(r"\s*(?:#{1,6}\s*Response:\s*)?Let[’']s think step by step\.?\s*",text,re.I))


def milestone_path(original,step):
    if step not in [0,100,1000,2500,5000]:raise ValueError('Unknown saved milestone')
    if step==0:raise ValueError('Base stage disables adapters and has no checkpoint path')
    root=Path(original)
    return root/'final_adapter/adapter_model.safetensors' if step==5000 else root/('trainer/milestones/step-'+str(step))/'adapter_model.safetensors'
