"""Frozen test-only Source prompt reference; never imported by runtime.

Source: offline_tta/tlm_tent_taskqueue_bundle/tent_text.py
Source file SHA256: 6f0032f486604b517ff3a047d0f34b1afc7cad6c94b16b3b76c1fddcedd8e3d2
Functions copied verbatim; all torch/PEFT code excluded.
"""
import hashlib


def prompt_text(row):
    instruction, context = row['instruction'], row['input']
    if not isinstance(instruction, str) or not isinstance(context, str):
        raise ValueError('instruction/input must be strings')
    return instruction + ('\n' + context if context else '')


def encode_prompt(row, tokenizer, template, system, max_input):
    """Match Source: trim only user content, then apply the complete template."""
    original = prompt_text(row)
    used = original

    def encode(content):
        return template.encode_oneturn(tokenizer, [
            {'role': 'user', 'content': content},
            {'role': 'assistant', 'content': ''}], system=system)[0]

    ids = encode(used)
    original_length = len(ids)
    if original_length > max_input:
        content_ids = tokenizer.encode(original, add_special_tokens=False)
        while len(ids) > max_input:
            content_ids = content_ids[:-max(1, len(ids) - max_input)]
            if not content_ids:
                raise ValueError('Prompt budget too small for the TLM template')
            used = tokenizer.decode(content_ids, skip_special_tokens=False)
            ids = encode(used)
    if not ids:
        raise ValueError('Empty encoded prompt')
    return used, ids, {'original_prompt_tokens': original_length,
                      'used_prompt_tokens': len(ids),
                      'input_truncated': original_length > len(ids),
                      'original_prompt_sha256': hashlib.sha256(original.encode('utf-8')).hexdigest()}
