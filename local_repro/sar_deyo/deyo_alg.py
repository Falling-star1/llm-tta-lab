"""PLPD (Pseudo-Label Probability Difference) for DeYO on autoregressive text.

Reference: https://github.com/Jhyun17/DeYO  (methods/deyo.py)
commit 3d7d2897d864907571ef75edf31f0028d53f1432.

DeYO's extra signal over Tent/EATA is Eqn. (9):

    PLPD_theta(x, x') = ( p_theta(x) - p_theta(x') )_yhat,   yhat = argmax p_theta(x)

where x' is an "object-destructive" transform of x.  On images that is patch-shuffling
(deyo.py:120-128): patch interiors untouched, patch order randomised -- local texture
kept, global shape destroyed.

Text has no shape factor, so two things are redefined here and nowhere else:

  x'  = block_shuffle(text): the input document's tokens are cut into n_chunks
        contiguous blocks and the block order is permuted.  Local n-grams survive,
        document/paragraph order and the coreference chain do not -- the closest
        analogue of "destroy the global structure, keep the local evidence".
        The instruction and the chat template are never touched.

  yhat = per-position argmax over the vocabulary, taken from the *unshuffled* forward
        (deyo.py:139).  The scalar is the mean over continuation positions of
        p_theta(yhat_t | x_<t), which is DeYO's "probability of the pseudo-label" with
        the project's per-token averaging convention.  x' is scored by teacher-forcing
        the *same* continuation tokens, so the two probabilities differ only because
        the prompt changed -- exactly like scoring the same image twice.

Both are our design decisions; the paper and the official code say nothing about text.
"""
from __future__ import annotations

import random

import torch

from eata_alg import _gather_continuation


def block_shuffle(text: str, n_chunks: int, seed: int) -> str:
    """Permute contiguous token blocks of `text`; block contents keep their order."""
    toks = (text or "").split()
    if len(toks) < 2:
        return text
    k = max(2, min(int(n_chunks), len(toks)))
    size = -(-len(toks) // k)
    chunks = [toks[i:i + size] for i in range(0, len(toks), size)]
    random.Random(int(seed)).shuffle(chunks)
    return " ".join(t for c in chunks for t in c)


def per_sample_mean(values, owner, valid_index, n_batch):
    """Average per-position `values` into per-sample scalars, keeping only valid samples."""
    sums = torch.zeros(n_batch, device=values.device, dtype=values.dtype)
    cnts = torch.zeros(n_batch, device=values.device, dtype=values.dtype)
    sums.index_add_(0, owner, values)
    cnts.index_add_(0, owner, torch.ones_like(values))
    return sums[valid_index] / cnts[valid_index]


def pos_stats(body, head, ids, mask, prompt_lens, cont_lens, ignore_ids, chunk: int = 128):
    """Per-position entropy (differentiable) and pseudo-label probability.

    Returns (pos_ent, pos_maxp, pos_argmax, owner, valid_index, n_batch).
    Position order is the same as `_gather_continuation`, so a second call with the same
    continuation tokens yields positionally aligned tensors.
    """
    out = body(input_ids=ids, attention_mask=mask, use_cache=False)
    hidden = out.last_hidden_state
    device = hidden.device
    n_batch = len(prompt_lens)

    sel_h, owner = _gather_continuation(hidden, ids, prompt_lens, cont_lens, ignore_ids, device)

    ents, maxps, argmaxes = [], [], []
    for start in range(0, sel_h.shape[0], chunk):
        logits = head(sel_h[start:start + chunk]).float()
        logp = torch.log_softmax(logits, dim=-1)
        ents.append(-(logp.exp() * logp).sum(-1))
        am = logits.argmax(dim=-1)
        maxps.append(logp.gather(1, am.unsqueeze(1)).squeeze(1).exp())
        argmaxes.append(am)

    pos_ent = torch.cat(ents, 0)
    pos_maxp = torch.cat(maxps, 0)
    pos_argmax = torch.cat(argmaxes, 0)

    cnts = torch.zeros(n_batch, device=device, dtype=pos_ent.dtype)
    cnts.index_add_(0, owner, torch.ones_like(pos_ent))
    valid_index = torch.nonzero(cnts > 0, as_tuple=False).squeeze(1)
    return pos_ent, pos_maxp, pos_argmax, owner, valid_index, n_batch


@torch.no_grad()
def prob_at(body, head, ids, mask, prompt_lens, cont_lens, ignore_ids, targets,
            chunk: int = 128):
    """p(target) at every continuation position, positionally aligned with pos_stats."""
    out = body(input_ids=ids, attention_mask=mask, use_cache=False)
    hidden = out.last_hidden_state
    device = hidden.device
    n_batch = len(prompt_lens)

    sel_h, owner = _gather_continuation(hidden, ids, prompt_lens, cont_lens, ignore_ids, device)
    assert sel_h.shape[0] == targets.shape[0], (
        f"position count changed between forwards: {sel_h.shape[0]} vs {targets.shape[0]}")

    parts = []
    for start in range(0, sel_h.shape[0], chunk):
        logits = head(sel_h[start:start + chunk]).float()
        logp = torch.log_softmax(logits, dim=-1)
        tgt = targets[start:start + chunk]
        parts.append(logp.gather(1, tgt.unsqueeze(1)).squeeze(1).exp())

    cnts = torch.zeros(n_batch, device=device, dtype=torch.float32)
    cnts.index_add_(0, owner, torch.ones(owner.numel(), device=device, dtype=torch.float32))
    valid_index = torch.nonzero(cnts > 0, as_tuple=False).squeeze(1)
    return torch.cat(parts, 0), owner, valid_index, n_batch
