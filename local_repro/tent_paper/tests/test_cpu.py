"""CPU-only unit tests for the strict Tent surface (no GPU, no model download).

    python tests/test_cpu.py        # in the tent_paper directory
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

import torch  # noqa: E402

import norm_affine as na  # noqa: E402
from entropy import batch_entropy, build_batch, resolve_body_head, valid_token_mask  # noqa: E402

FAIL = []


def check(name, cond, extra=""):
    print(("  PASS  " if cond else "  FAIL  ") + name + (f"   {extra}" if extra else ""))
    if not cond:
        FAIL.append(name)


def tiny_model(hidden=32, layers=2, vocab=64):
    from transformers import LlamaConfig, LlamaForCausalLM

    cfg = LlamaConfig(vocab_size=vocab, hidden_size=hidden, intermediate_size=hidden * 2,
                      num_hidden_layers=layers, num_attention_heads=4, num_key_value_heads=2,
                      max_position_embeddings=1024, tie_word_embeddings=True)
    m = LlamaForCausalLM(cfg).eval()
    return m


def test_gamma_count():
    m = tiny_model(hidden=32, layers=2)
    n = len(na.install(m))
    pairs = na.enable_norm_gamma(m)
    expect = (2 * 2 + 1) * 32                      # (attn + mlp) * layers + final, x hidden
    check("patched norm layers == 2*layers+1", n == 2 * 2 + 1, f"n={n}")
    check("gamma params == (2L+1)*D", sum(p.numel() for _, p in pairs) == expect,
          f"{sum(p.numel() for _, p in pairs)} vs {expect}")
    others = [p for nm, p in m.named_parameters()
              if not (nm.endswith("layernorm.weight") or nm.endswith("norm.weight"))]
    check("no other parameter is trainable", all(not p.requires_grad for p in others))
    check("all gamma start at 1.0", all(bool((p.detach() - 1).abs().max() < 1e-6) for _, p in pairs))


def test_forward_matches_native():
    """The patched forward must be the native RMSNorm computation, nothing else."""
    from transformers.models.llama.modeling_llama import LlamaRMSNorm

    d = 16
    torch.manual_seed(0)
    w = torch.randn(d).abs() + 0.5
    x = torch.randn(3, 5, d)

    # (1) fp32 gamma -- the training setup. The patched output must be exactly the
    # native fp32 result rounded to the activation dtype. Rounding gamma to fp16
    # first would break this equality, so this also proves gamma is used at fp32.
    native32 = LlamaRMSNorm(d)
    native32.weight.data = w.clone()
    patched32 = LlamaRMSNorm(d)
    patched32.weight.data = w.clone()
    na.install(patched32)
    ref32 = native32(x)                                  # fp32 in -> fp32 out
    out32 = patched32(x)
    check("fp32 gamma: output dtype follows the input", out32.dtype == x.dtype, str(out32.dtype))
    check("fp32 gamma: identical to the native fp32 computation",
          torch.equal(out32, ref32),
          f"max diff {float((out32 - ref32).abs().max()):.2e}")

    # (2) fp16 gamma and fp16 activations: same computation, one fp32 multiply then round
    native16 = LlamaRMSNorm(d).half()
    native16.weight.data = w.half()
    patched16 = LlamaRMSNorm(d).half()
    patched16.weight.data = w.half()
    na.install(patched16)
    x16 = x.half()
    check("fp16: patched matches native within fp16 rounding",
          torch.allclose(patched16(x16).float(), native16(x16).float(), atol=1e-3, rtol=1e-3),
          f"max diff {float((patched16(x16).float() - native16(x16).float()).abs().max()):.2e}")

    # (3) padding must not change the normalizer: RMSNorm is per token, per position
    patched32_pad = LlamaRMSNorm(d)
    patched32_pad.weight.data = w.clone()
    na.install(patched32_pad)
    x_pad = x.clone()
    x_pad[1, 3:] = 0.0
    out_pad = patched32_pad(x_pad)
    check("padding does not leak across positions",
          torch.allclose(out_pad[1, :3], out32[1, :3], atol=1e-6),
          f"max diff {float((out_pad[1, :3] - out32[1, :3]).abs().max()):.2e}")


def test_valid_token_mask():
    keep = valid_token_mask([5, 6, 7, 2, 9, 9], [2, 0])
    check("valid mask stops at first stop token", keep.tolist() == [True, True, True, False, False, False],
          str(keep.tolist()))
    keep = valid_token_mask([5, 6], [2, 0])
    check("valid mask keeps all when no stop", keep.tolist() == [True, True])


def test_padding_invariance_and_entropy():
    torch.manual_seed(0)
    m = tiny_model(hidden=32, layers=2, vocab=64)
    na.install(m)
    body, head = resolve_body_head(m)
    pad_id, stop_id = 0, 63
    p1, c1 = [1, 2, 3, 4, 5], [6, 7, 8]
    p2, c2 = [1, 2, 3], [9, 10]
    ids, mask, plens, clens = build_batch([p1, p2], [c1, c2], pad_id)
    loss, info = batch_entropy(body, head, ids, mask, plens, clens, [stop_id, pad_id])

    with torch.no_grad():
        logits = m(input_ids=ids, attention_mask=mask).logits
    per = []
    for i, (lp, lc) in enumerate(zip(plens, clens)):
        sel = logits[i, lp - 1:lp - 1 + lc].float()
        logp = torch.log_softmax(sel, dim=-1)
        per.append(float((-(logp.exp() * logp).sum(-1)).mean()))
    ref = sum(per) / len(per)
    check("batch entropy matches per-sample reference", abs(loss.item() - ref) < 1e-4,
          f"{loss.item():.6f} vs {ref:.6f}")
    check("valid token counts", info["valid_tokens"] == clens, str(info["valid_tokens"]))

    # gradient reaches gamma only, and is non-zero
    na.enable_norm_gamma(m)
    m.zero_grad(set_to_none=True)
    loss2, _ = batch_entropy(body, head, ids, mask, plens, clens, [stop_id, pad_id])
    loss2.backward()
    pairs = na.trainable_norm_params(m)
    gs = [p.grad for _, p in pairs if p.grad is not None]
    check("gradient present on gamma", len(gs) == len(pairs) and all(float(g.abs().sum()) > 0 for g in gs))
    check("all gamma grads finite", all(bool(torch.isfinite(g).all()) for g in gs))


def test_stop_truncation():
    m = tiny_model(hidden=32, layers=2, vocab=64)
    na.install(m)
    body, head = resolve_body_head(m)
    pad_id, stop_id = 0, 63
    ids, mask, plens, clens = build_batch([[1, 2, 3, 4]], [[5, 6, stop_id, 8, 9]], pad_id)
    _, info = batch_entropy(body, head, ids, mask, plens, clens, [stop_id, pad_id])
    check("tokens after stop are excluded", info["valid_tokens"] == [2], str(info["valid_tokens"]))


def test_no_statistics_surface_left():
    """The module must not expose any test-batch-statistics entry point."""
    for attr in ("set_active_mask", "get_active_mask", "batch_stats_forward", "_ACTIVE_MASK", "DIAG"):
        check(f"norm_affine has no {attr}", not hasattr(na, attr))


if __name__ == "__main__":
    print("strict Tent surface -- CPU unit tests")
    test_gamma_count()
    test_forward_matches_native()
    test_valid_token_mask()
    test_padding_invariance_and_entropy()
    test_stop_truncation()
    test_no_statistics_surface_left()
    print()
    if FAIL:
        print("FAILED:", ", ".join(FAIL))
        sys.exit(1)
    print("ALL CPU TESTS PASSED")
