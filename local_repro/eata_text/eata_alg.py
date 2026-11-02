"""EATA (Niu et al., ICML 2022) ported to token-level text adaptation.

Reference implementation: https://github.com/mr-eggplant/EATA  (eata.py, main)
commit f739b3668cc7617e9b9f1979c1a358497a3472c3.

Mapping from the closed-set classifier code to an autoregressive LM:

  official (eata.py)                     | here
  ---------------------------------------|---------------------------------------
  outputs = model(x)  -> logits [B, C]   | next-token logits at the 80 continuation
                                         | positions only (gather -> lm_head)
  softmax_entropy(outputs) -> [B]        | per-sample mean token entropy  E_i
  outputs.softmax(1) -> [B, C]           | per-sample mean next-token distribution
                                         | p_i over the V=128k vocabulary
  filter 1: entropys < e_margin          | same
  filter 2: |cos(p_i, m_{t-1})| < d_margin,  m = EMA(0.9 * old + 0.1 * new)
  coeff = 1 / exp(E - E0)  -> exp(E0 - E) | same (detached)
  loss = mean over SELECTED of E * coeff | same
  grad step only if |selected| > 0       | same (counted for #Backwards)
  diagonal Fisher over BN affine params  | over RMSNorm gamma (same surface choice as
                                         | the Tent reproduction on this server)
  e_margin default = 0.4*log(num_classes) | 0.4*log(V)  (--e0-mode ratio)
  optimizer: SGD(lr=2.5e-4, momentum=.9)  | same by default

The paper's E0 is the coefficient 0.4 applied to log C; the code's default
e_margin is math.log(1000) * 0.40. For a V-way next-token distribution the
analog is 0.4*log(V).
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from entropy import valid_token_mask


def _gather_continuation(hidden, ids, prompt_lens, cont_lens, ignore_ids, device):
    """Hidden states of the positions that predict a *real* continuation token."""
    sel_h, owner = [], []
    for i, (lp, lc) in enumerate(zip(prompt_lens, cont_lens)):
        if lc <= 0:
            continue
        cont_ids = ids[i, lp:lp + lc]
        keep = valid_token_mask(cont_ids.tolist(), ignore_ids).to(device)
        if not bool(keep.any()):
            continue
        pos = torch.arange(lp - 1, lp - 1 + lc, device=device)[keep]
        sel_h.append(hidden[i].index_select(0, pos))
        owner.append(torch.full((pos.numel(),), i, device=device))
    if not sel_h:
        raise RuntimeError("no valid continuation token in this batch")
    return torch.cat(sel_h, 0), torch.cat(owner, 0)


def forward_entropy_probs(body, head, ids, mask, prompt_lens, cont_lens, ignore_ids,
                          chunk: int = 128, want_probs: bool = True):
    """Per-sample mean token entropy (differentiable) and mean next-token distribution.

    Returns (per_sample_entropy [n_valid], probs [n_valid, V] or None, valid_index).
    `valid_index` are the batch indices that carried at least one valid position.
    """
    out = body(input_ids=ids, attention_mask=mask, use_cache=False)
    hidden = out.last_hidden_state
    device = hidden.device
    B = len(prompt_lens)

    sel_h, owner = _gather_continuation(hidden, ids, prompt_lens, cont_lens,
                                       ignore_ids, device)

    V = int(head.weight.shape[0])
    prob_acc = torch.zeros((B, V), device=device, dtype=torch.float32) if want_probs else None

    ent_parts = []
    for start in range(0, sel_h.shape[0], chunk):
        logits = head(sel_h[start:start + chunk]).float()
        logp = torch.log_softmax(logits, dim=-1)
        ent_parts.append(-(logp.exp() * logp).sum(-1))
        if want_probs:
            prob_acc.index_add_(0, owner[start:start + chunk], logp.exp())
    ent = torch.cat(ent_parts, 0)

    sums = torch.zeros(B, device=device, dtype=ent.dtype)
    cnts = torch.zeros(B, device=device, dtype=ent.dtype)
    sums.index_add_(0, owner, ent)
    cnts.index_add_(0, owner, torch.ones_like(ent))
    valid = cnts > 0
    per_sample = sums[valid] / cnts[valid]
    probs = None
    if want_probs:
        probs = prob_acc[valid] / cnts[valid].unsqueeze(1)
    return per_sample, probs, torch.nonzero(valid, as_tuple=False).squeeze(1)


def eata_select(per_sample, probs, e_margin, d_margin, ema):
    """Two-stage sample selection of Eqn. (3) and Eqn. (5).

    Returns (selected_entropies, selected_probs, new_ema, stats). Any of the
    returned tensors may be empty -> the caller must skip the update.
    """
    stats = {"n_samples": int(per_sample.numel())}

    idx1 = torch.nonzero(per_sample < e_margin, as_tuple=False).squeeze(1)
    stats["n_reliable"] = int(idx1.numel())
    stats["ema_before"] = None if ema is None else float(ema.abs().sum())

    if idx1.numel() == 0:
        return None, None, ema, stats

    e1, p1 = per_sample[idx1], probs[idx1]
    # d_margin=None or <=0 means "S_div disabled" -- Eqn (5) is not applied at all.
    # A large finite threshold such as 1.0 is NOT equivalent: once the model collapses to a
    # one-hot next-token distribution |cos| reaches exactly 1.0, the test |cos| < 1.0 turns
    # false and the adaptation silently freezes (observed on llama3b/geo at step 1964).
    use_div = ema is not None and d_margin is not None and float(d_margin) > 0.0
    if use_div:
        cos = F.cosine_similarity(ema.unsqueeze(0), p1, dim=1)
        keep = torch.nonzero(cos.abs() < d_margin, as_tuple=False).squeeze(1)
        e_sel, p_sel = e1[keep], p1[keep]
        ema_update_src = p1[keep] if keep.numel() else None
        stats["cos_abs_mean"] = float(cos.abs().mean())
    else:
        e_sel, p_sel = e1, p1
        ema_update_src = p1

    stats["n_selected"] = int(e_sel.numel())

    if ema_update_src is not None and ema_update_src.numel() > 0:
        new_ema = (ema_update_src.detach().mean(0) if ema is None
                   else 0.9 * ema + 0.1 * ema_update_src.detach().mean(0))
    else:
        new_ema = ema
    return e_sel, p_sel, new_ema, stats


def eata_loss(e_sel, e_margin):
    """Eqn. (3) weighting: mean over selected of E * exp(E0 - E)."""
    coeff = torch.exp(e_margin - e_sel.detach().clone())
    return (e_sel * coeff).mean(0), float(coeff.mean())


def fisher_penalty(pairs, fishers, fisher_alpha):
    """Eqn. (8) second term: sum_i omega_i (theta_i - theta0_i)^2."""
    if not fishers:
        return None
    total = None
    for name, prm in pairs:
        if name not in fishers:
            continue
        f, p0 = fishers[name]
        term = fisher_alpha * (f * (prm - p0) ** 2).sum()
        total = term if total is None else total + term
    return total


def compute_fisher(body, head, ids, mask, prompt_lens, cont_lens, ignore_ids,
                   pairs, n_batches, chunk: int = 128, fisher_clip_by_norm: float = 10.0):
    """Diagonal Fisher over the adapted parameters with the model's own hard pseudo-labels.

    Official EATA uses 2000 unlabeled in-distribution samples, takes the model's
    argmax as the label, averages (grad of CE)^2 over batches, and normalises by
    the global norm (arg --fisher_clip_by_norm).
    """
    out = body(input_ids=ids, attention_mask=mask, use_cache=False)
    hidden = out.last_hidden_state
    device = hidden.device
    sel_h, owner = _gather_continuation(hidden, ids, prompt_lens, cont_lens, ignore_ids, device)

    # labels: the token that follows each selected prediction position
    labels = []
    for i, (lp, lc) in enumerate(zip(prompt_lens, cont_lens)):
        if lc <= 0:
            continue
        cont_ids = ids[i, lp:lp + lc]
        keep = valid_token_mask(cont_ids.tolist(), ignore_ids).to(device)
        if not bool(keep.any()):
            continue
        nxt = torch.cat([cont_ids[1:], cont_ids[-1:]])
        labels.append(nxt[keep])
    if not labels:
        return {}
    labels = torch.cat(labels, 0)

    logits = head(sel_h).float()
    loss = F.cross_entropy(logits, labels, reduction="mean")
    loss.backward()
    out_f = {}
    total_sq = 0.0
    for name, prm in pairs:
        if prm.grad is None:
            continue
        g = prm.grad.detach().clone() ** 2
        out_f[name] = g
        total_sq += float(g.sum())
    if total_sq <= 0:
        return {}
    scale = min(1.0, fisher_clip_by_norm / math.sqrt(total_sq))
    for name, prm in pairs:
        prm.grad = None
    return {name: (g * scale, prm.detach().clone()) for name, prm in pairs
            if name in out_f}
