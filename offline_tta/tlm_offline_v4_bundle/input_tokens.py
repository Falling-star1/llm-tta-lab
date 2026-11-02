"""Strict answer-free records and Source-compatible prompt tokenization.

Only preparation may call sanitize_record on raw AdaptEval rows. Training must
call require_input_record instead, which rejects rows containing any extra keys.
"""
import hashlib
import json


INPUT_KEYS = frozenset({'source_index', 'instruction', 'input'})


def require_input_record(record):
    """Reject answers, unknown keys and invalid input before tokenization."""
    if not isinstance(record, dict) or set(record) != INPUT_KEYS:
        raise ValueError('Input record requires exactly source_index, instruction, input')
    index = record['source_index']
    if type(index) is not int or index < 0:
        raise ValueError('source_index must be a nonnegative integer')
    if not isinstance(record['instruction'], str) or not isinstance(record['input'], str):
        raise ValueError('instruction/input must be strings')
    if not (record['instruction'] + '\n' + record['input']).strip():
        raise ValueError('Input text must not be empty or whitespace')
    return record


def sanitize_record(raw, source_index):
    """Extract input fields; reference fields are never read here."""
    if not isinstance(raw, dict) or 'instruction' not in raw or 'input' not in raw:
        raise ValueError('Raw row requires instruction and input')
    record = {'source_index': source_index, 'instruction': raw['instruction'],
              'input': raw['input']}
    return require_input_record(record)


def prompt_text(record):
    require_input_record(record)
    return record['instruction'] + ('\n' + record['input'] if record['input'] else '')


def encode_input(record, tokenizer, template, system, max_input=4096):
    """Trim user-content tokens from the tail and reapply the full template.

    The loop deliberately matches frozen tent_text.encode_prompt, including its
    rejection of budgets that leave no user-content tokens. Never slice rendered
    template tokens: doing so could remove role delimiters or the system prompt.
    token_sha256 hashes the UTF-8 compact JSON array of the final token IDs.
    """
    if type(max_input) is not int or max_input < 1:
        raise ValueError('max_input must be a positive integer')
    if system is not None and not isinstance(system, str):
        raise ValueError('system must be a string or None for the template default')
    original = prompt_text(record)

    def encode(content):
        ids = list(template.encode_oneturn(tokenizer, [
            {'role': 'user', 'content': content},
            {'role': 'assistant', 'content': ''}], system=system)[0])
        if not ids or any(type(token) is not int or token < 0 for token in ids):
            raise ValueError('Template must return a nonempty sequence of integer token IDs')
        return ids

    ids = encode(original)
    original_length = len(ids)
    if original_length > max_input:
        content_ids = tokenizer.encode(original, add_special_tokens=False)
        while len(ids) > max_input:
            content_ids = content_ids[:-max(1, len(ids) - max_input)]
            if not content_ids:
                raise ValueError('Prompt budget too small for the TLM template')
            used = tokenizer.decode(content_ids, skip_special_tokens=False)
            ids = encode(used)
    return ids, {
        'source_index': record['source_index'],
        'original_prompt_tokens': original_length,
        'used_prompt_tokens': len(ids),
        'input_truncated': original_length > len(ids),
        'original_prompt_sha256': hashlib.sha256(original.encode('utf-8')).hexdigest(),
        'token_sha256': hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode('utf-8')).hexdigest(),
    }
