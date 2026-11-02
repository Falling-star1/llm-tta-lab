"""Input-token CE and frozen-base weighting from the pinned offline TLM code."""
import math

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def _targets(input_ids, attention_mask=None):
    if input_ids.ndim != 2 or input_ids.dtype != torch.long or input_ids.shape[1] < 2:
        raise ValueError('Expected batch x sequence int64 IDs with at least two tokens')
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    if attention_mask.shape != input_ids.shape or not bool(((attention_mask == 0) | (attention_mask == 1)).all()):
        raise ValueError('Attention mask must be binary and match input IDs')
    if not bool((attention_mask[:, 0] == 1).all()) or bool((attention_mask[:, 1:] > attention_mask[:, :-1]).any()):
        raise ValueError('TLM training requires nonempty right-padded inputs')
    labels = input_ids[:, 1:].clone()
    labels.masked_fill_(attention_mask[:, 1:].to(labels.device) == 0, -100)
    counts = (labels != -100).sum(-1)
    if bool((counts == 0).any()):
        raise ValueError('Each input must have a valid next-token target')
    return labels, counts


def _check_finite(value, name):
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError('Nonfinite ' + name)


def input_cross_entropy(logits, input_ids, attention_mask=None):
    """Per-input mean CE; padding is identified by its mask, never its token ID."""
    labels, counts = _targets(input_ids, attention_mask)
    if logits.ndim != 3 or logits.shape[:2] != input_ids.shape:
        raise ValueError('Logits and input sequence shapes differ')
    selected = logits[:, :-1].float()
    _check_finite(selected, 'input logits')
    labels = labels.to(selected.device)
    losses = F.cross_entropy(selected.reshape(-1, selected.shape[-1]),
                             labels.reshape(-1), reduction='none', ignore_index=-100)
    result = losses.reshape(labels.shape).sum(-1) / counts.to(selected.device)
    _check_finite(result, 'input CE')
    return result


def sample_weights(base_ce, threshold=3., lamb=.1):
    """Return strict selection and detached weights, including unselected weights."""
    if base_ce.ndim != 1 or not base_ce.numel():
        raise ValueError('Expected one base CE per input')
    if not math.isfinite(threshold) or not math.isfinite(lamb) or lamb <= 0:
        raise ValueError('Invalid threshold or lambda')
    base = base_ce.detach().float()
    _check_finite(base, 'base CE')
    if bool((base < 0).any()):
        raise ValueError('Base CE cannot be negative')
    weight = lamb * torch.exp(base - threshold)
    _check_finite(weight, 'sample weights')
    return base > threshold, weight


def weighted_input_loss(current_ce, base_ce, threshold=3., lamb=.1):
    """Keep a zero-gradient graph when nothing is selected, as upstream does."""
    if current_ce.shape != base_ce.shape:
        raise ValueError('Current and base CE shapes differ')
    _check_finite(current_ce, 'current CE')
    mask, weights = sample_weights(base_ce, threshold, lamb)
    weighted = current_ce * weights.to(current_ce.device) * mask.to(current_ce.device)
    result = weighted.sum() / mask.sum().clamp_min(1).to(current_ce.device)
    _check_finite(result, 'weighted input loss')
    return result


def model_input_cross_entropy(model, input_ids, attention_mask=None, chunk_size=128):
    """Project every input position in chunks; checkpoint projections in training.

    Supports the pinned Llama/Qwen decoder architecture. No suffix is dropped.
    Checkpointing avoids retaining all chunk logits for the backward pass. The
    full decoder hidden states still exist, so GPU peak memory needs preflight.
    """
    labels, counts = _targets(input_ids, attention_mask)
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError('Projection chunk size must be a positive integer')
    base = model.get_base_model() if hasattr(model, 'get_base_model') else model
    if not hasattr(base, 'model') or not hasattr(base, 'get_output_embeddings'):
        raise ValueError('Expected pinned Llama/Qwen causal LM decoder architecture')
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    hidden = base.model(input_ids=input_ids, attention_mask=attention_mask,
                        use_cache=False, return_dict=True).last_hidden_state
    head = base.get_output_embeddings()

    tile_size = min(chunk_size, labels.shape[1])

    def project_tokens(states, targets):
        valid = targets.shape[1]
        padding = tile_size - valid
        if padding:
            # Keep the final projection on the same matrix shape as the other
            # tiles. Dummy positions are zero states with ignored CE labels.
            states = F.pad(states, (0, 0, 0, padding))
            targets = F.pad(targets, (0, padding), value=-100)
        logits = head(states).float()
        _check_finite(logits, 'chunk logits')
        token_losses = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                       targets.to(logits.device).reshape(-1),
                                       reduction='none', ignore_index=-100)
        return token_losses.reshape(targets.shape)[:, :valid]

    token_parts = []
    for start in range(0, labels.shape[1], chunk_size):
        stop = min(start + chunk_size, labels.shape[1])
        states = hidden[:, start:stop]
        targets = labels[:, start:stop]
        if torch.is_grad_enabled() and states.requires_grad:
            part = checkpoint(project_tokens, states, targets, use_reentrant=False,
                              preserve_rng_state=True)
        else:
            part = project_tokens(states, targets)
        token_parts.append(part)
    # Match the full-forward reduction order; retaining one FP32 loss per
    # token is small, while checkpointing still avoids retaining full logits.
    losses = torch.cat(token_parts, dim=1)
    result = losses.sum(-1) / counts.to(losses.device)
    _check_finite(result, 'chunked input CE')
    return result
