"""Batched, masked token entropy for Tent-on-text.

Design notes
------------
* Only the *continuation* positions contribute (the 80-token self-generated prefix),
  which is TLM's text adaptation for LLMs; the prompt never contributes.
* Positions at/after the first stop or pad token are excluded (same rule as the
  team's `tent_text.generated_positions`, verified by a unit test).
* Logits are produced only at the selected continuation positions
  (body -> gather -> lm_head). Materialising [B, L, V] logits is what makes
  b>=2 impossible on 11 GB cards: 8 x 4176 x 152k x 2 B = 10.1 GB.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def resolve_body_head(model):
    """Return (transformer_body, lm_head) for a HF causal LM (Llama / Qwen2)."""
    body = getattr(model, "model", None)
    head = getattr(model, "lm_head", None)
    if body is None or head is None:
        raise RuntimeError("cannot resolve model.model / model.lm_head for this architecture")
    return body, head


def build_batch(prompts, conts, pad_id):
    """Right-pad prompt+continuation. Returns ids, mask, prompt_lens, cont_lens."""
    lengths = [len(p) + len(c) for p, c in zip(prompts, conts)]
    L = max(lengths)
    B = len(prompts)
    ids = torch.full((B, L), int(pad_id), dtype=torch.long)
    mask = torch.zeros((B, L), dtype=torch.long)
    for i, (p, c) in enumerate(zip(prompts, conts)):
        n = len(p) + len(c)
        ids[i, :n] = torch.tensor(list(p) + list(c), dtype=torch.long)
        mask[i, :n] = 1
    return ids, mask, [len(p) for p in prompts], [len(c) for c in conts]


def valid_token_mask(cont, ignore_ids):
    """1 for continuation tokens that are real (before the first stop/pad)."""
    keep = torch.ones(len(cont), dtype=torch.bool)
    ignored = set(int(i) for i in ignore_ids)
    for j, tok in enumerate(cont):
        if int(tok) in ignored:
            keep[j:] = False
            break
    return keep


def batch_entropy(body, head, ids, mask, prompt_lens, cont_lens, ignore_ids,
                  chunk: int = 128):
    """Mean token entropy, computed per sample then averaged over the batch.

    Returns (loss, info) where info carries per-sample entropies and valid counts.
    """
    out = body(input_ids=ids, attention_mask=mask, use_cache=False)
    hidden = out.last_hidden_state                      # [B, L, D]

    sel_h, owner = [], []
    for i, (lp, lc) in enumerate(zip(prompt_lens, cont_lens)):
        if lc <= 0:
            continue
        cont_ids = ids[i, lp:lp + lc]
        keep = valid_token_mask(cont_ids.tolist(), ignore_ids).to(hidden.device)
        if not bool(keep.any()):
            continue
        pos = torch.arange(lp - 1, lp - 1 + lc, device=hidden.device)[keep]
        sel_h.append(hidden[i].index_select(0, pos))
        owner.append(torch.full((pos.numel(),), i, device=hidden.device))
    if not sel_h:
        raise RuntimeError("no valid continuation token in this batch")

    sel_h = torch.cat(sel_h, 0)                         # [N, D]
    owner = torch.cat(owner, 0)                         # [N]

    ents, counts = [], []
    for start in range(0, sel_h.shape[0], chunk):
        logits = head(sel_h[start:start + chunk]).float()
        logp = torch.log_softmax(logits, dim=-1)
        ents.append(-(logp.exp() * logp).sum(-1))       # [n]
    ent_all = torch.cat(ents, 0)                        # [N], differentiable

    B = len(prompt_lens)
    sums = torch.zeros(B, device=ent_all.device, dtype=ent_all.dtype)
    cnts = torch.zeros(B, device=ent_all.device, dtype=ent_all.dtype)
    sums.index_add_(0, owner, ent_all)
    cnts.index_add_(0, owner, torch.ones_like(ent_all))
    valid_samples = cnts > 0
    per_sample_t = sums[valid_samples] / cnts[valid_samples]     # per-sample mean entropy
    loss = per_sample_t.mean()                                   # paper: batch mean
    per_sample = [float(v) if c > 0 else float("nan")
                  for v, c in zip((sums / cnts.clamp_min(1)).tolist(), cnts.tolist())]
    info = {"per_sample_entropy": per_sample, "valid_tokens": [int(c) for c in cnts.tolist()],
            "token_entropy_mean": float(ent_all.mean()), "num_positions": int(ent_all.numel())}
    return loss, info


def entropy_backward(body, head, ids, mask, prompt_lens, cont_lens, ignore_ids):
    """Forward + backward on the mean continuation entropy."""
    loss, info = batch_entropy(body, head, ids, mask, prompt_lens, cont_lens, ignore_ids)
    before = float(loss.detach())
    loss.backward()
    return before, info, loss
