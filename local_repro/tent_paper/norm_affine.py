"""Tent-style adaptation surface for RMSNorm LLMs: gamma only, ZERO new parameters.

Paper mapping (Tent, 2006.10726):
  "collect the affine transformation parameters {gamma, beta} in the source model;
   the remaining parameters are fixed" -> source model has only `weight` (gamma),
   so only gamma is trained. No beta exists in RMSNorm, nothing is created.

Why there is no test-batch statistic here
----------------------------------------
Tent replaces BatchNorm's *running* statistics (estimated on source data) with the
statistics of the current test batch. That step exists only because BatchNorm
caches mu/sigma from training and keeps reusing them at test time.

RMSNorm has no running statistics. Its normalizer is the per-token RMS of the
input it is currently given, recomputed on every forward pass. There is nothing
cached to replace, so the source RMSNorm computation is kept unchanged; the only
thing Tent adapts on this architecture is gamma.

The forward below is the HF RMSNorm computation verbatim (LlamaRMSNorm /
Qwen2RMSNorm). The single deviation is a dtype bridge: gamma is kept in fp32
because fp16 spacing at 1.0 is 2^-10 ~ 9.8e-4 while an Adam step at lr=1.5e-5 is
~1e-7, so an fp16 gamma would never move. Gamma therefore takes part in the
multiply at full fp32 precision and only the product is cast back to the
activation dtype -- casting gamma itself first would round it onto the fp16 grid
and throw away most of the adaptation.
"""
from __future__ import annotations

import torch


def rmsnorm_forward(self, hidden_states):
    """LlamaRMSNorm/Qwen2RMSNorm.forward with an fp32 gamma kept in fp32."""
    input_dtype = hidden_states.dtype
    x = hidden_states.to(torch.float32)
    var = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(var + float(getattr(self, "variance_epsilon", 1e-6)))
    return (self.weight.float() * x).to(input_dtype)


def install(model):
    """Bind `rmsnorm_forward` to every normalization layer. Returns the patched names."""
    patched = []
    for name, mod in model.named_modules():
        if not hasattr(mod, "variance_epsilon") or not hasattr(mod, "weight"):
            continue
        mod.forward = rmsnorm_forward.__get__(mod, type(mod))
        patched.append(name)
    return patched


def trainable_norm_params(model):
    """(name, param) of every normalization weight; nothing else is trainable."""
    return [(n, p) for n, p in model.named_parameters()
            if n.endswith("layernorm.weight") or n.endswith("norm.weight")]


def expected_norm_param_count(model) -> int:
    return int(sum(p.numel() for _, p in trainable_norm_params(model)))


def freeze_all(model) -> None:
    for p in model.parameters():
        p.requires_grad_(False)


def enable_norm_gamma(model) -> list:
    freeze_all(model)
    pairs = trainable_norm_params(model)
    for _, p in pairs:
        p.requires_grad_(True)
    return pairs


def gamma_drift(pairs) -> dict:
    """mean |gamma - 1| and max |gamma - 1| (gamma starts at 1 for both RMSNorm impls)."""
    diffs = [(p.detach().float() - 1.0).abs() for _, p in pairs]
    if not diffs:
        return {"mean_abs_delta": 0.0, "max_abs_delta": 0.0}
    mean = float(sum(float(d.mean()) for d in diffs) / len(diffs))
    mx = float(max(float(d.max()) for d in diffs))
    return {"mean_abs_delta": mean, "max_abs_delta": mx}


def snapshot(pairs):
    return {n: p.detach().cpu().clone() for n, p in pairs}


def restore(pairs, snap):
    with torch.no_grad():
        for n, p in pairs:
            p.copy_(snap[n].to(p.device))


def drift_vs(pairs, snap):
    """Deviation of gamma from its initial snapshot: mean and max |delta|."""
    if not pairs:
        return {"mean_abs_delta": 0.0, "max_abs_delta": 0.0}
    tot, mx = 0.0, 0.0
    for n, p in pairs:
        ref = snap[n].to(p.device).float()
        d = (p.detach().float() - ref).abs()
        tot += float(d.mean())
        mx = max(mx, float(d.max()))
    return {"mean_abs_delta": tot / len(pairs), "max_abs_delta": mx}
