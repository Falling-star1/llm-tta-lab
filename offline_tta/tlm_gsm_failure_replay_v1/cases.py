"""Select a fixed subset of previously observed failures, without consulting gold."""
import random
import re


def is_stub(text):
    return bool(re.fullmatch(r"\s*(?:#{1,6}\s*Response:\s*)?Let[’']s think step by step\.?\s*",text,re.I))


def failure_indices(rows,expected_rows=5000,expected_failures=350,count=16):
    if len(rows)!=expected_rows or any(type(r.get('source_index')) is not int or r['source_index']!=i for i,r in enumerate(rows)):
        raise ValueError('Prediction indexes or count changed')
    failed=[r['source_index'] for r in rows if is_stub(r['predict'])]
    if len(failed)!=expected_failures or type(count) is not int or not 1<=count<=len(failed):
        raise ValueError('Observed failure population or selection size changed')
    return sorted(random.Random(314159).sample(failed,count))
