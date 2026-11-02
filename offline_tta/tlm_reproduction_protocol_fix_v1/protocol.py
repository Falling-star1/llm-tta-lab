"""Official decoding policy and descriptive failure flags, independent of scores."""
import re


def official_stop_ids(tokenizer):
    values=[tokenizer.eos_token_id]+list(tokenizer.additional_special_tokens_ids)
    if any(type(v) is not int or v<0 for v in values):raise ValueError('Invalid official stop token IDs')
    return list(dict.fromkeys(values))


def decode_official_prefix(ids,tokenizer):
    if not isinstance(ids,list) or any(type(v) is not int or v<0 for v in ids):raise ValueError('Invalid generated token IDs')
    stops=set(official_stop_ids(tokenizer))
    position=next((i for i,v in enumerate(ids) if v in stops),None)
    prefix=ids if position is None else ids[:position+1]
    return {'generated_ids':prefix,'predict':tokenizer.decode(prefix,skip_special_tokens=True),
            'shortened':len(prefix)<len(ids),'first_stop_position':position}


def answer_flags(text):
    empty=not text.strip()
    title=bool(re.fullmatch(r'\s*(?:#{1,6}[ \t]*(?:response|answer)[ \t]*:[ \t\r\n]*)+',text,re.I))
    return {'empty':empty,'title_only':title,'no_substantive_answer':empty or title}
