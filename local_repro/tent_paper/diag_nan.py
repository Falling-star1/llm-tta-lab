"""Locate the NaN source in one adaptation step (diagnostic only)."""
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, "/home/chuwenjing/TTA_LLM/TLM/src")
sys.path.insert(0, "/home/chuwenjing/TTA_LLM/TLM/scripts/eval")

import torch

import norm_affine as na
from entropy import batch_entropy, build_batch, resolve_body_head
from run import MODELS, INPUTS, encode, generate, build_runtime, read_jsonl


def stats(t, name):
    with torch.no_grad():
        f = t.detach().float()
        print(f"    {name}: shape={tuple(f.shape)} min={f.min().item():.4g} "
              f"max={f.max().item():.4g} mean={f.mean().item():.4g} nan={bool(torch.isnan(f).any())}")


def main():
    model, tok, template, system, stop_ids, pad_id, pairs = build_runtime("llama3b", 0)
    body, head = resolve_body_head(model)
    rows = read_jsonl(INPUTS / "gsm8k_5k" / "inputs.jsonl")[:2]
    prompts, conts = [], []
    na.set_active_mask(None)
    for r in rows:
        _, ids = encode(template, tok, r, system)
        prompts.append(ids)
        conts.append(generate(model, tok, ids, 80, stop_ids, pad_id) or [pad_id])

    ids_t, mask_t, plens, clens = build_batch(prompts, conts, pad_id)
    ids_t, mask_t = ids_t.to("cuda:0"), mask_t.to("cuda:0")
    print(f"batch: B={ids_t.shape[0]} L={ids_t.shape[1]} plens={plens} clens={clens}")

    # --- 1) source-mode forward (mask None) ---
    na.set_active_mask(None)
    model.eval()
    with torch.no_grad():
        out = body(input_ids=ids_t, attention_mask=mask_t, use_cache=False)
    print("  [source mode] hidden:")
    stats(out.last_hidden_state, "hidden")

    # --- 2) batch-stat forward, eval mode, no grad ---
    na.set_active_mask(mask_t)
    with torch.no_grad():
        out2 = body(input_ids=ids_t, attention_mask=mask_t, use_cache=False)
    print("  [batch-stat, eval] hidden:")
    stats(out2.last_hidden_state, "hidden")

    # --- 3) train mode + backward ---
    model.train()
    opt = torch.optim.Adam([p for _, p in pairs], lr=1.5625e-5)
    opt.zero_grad(set_to_none=True)
    loss, info = batch_entropy(body, head, ids_t, mask_t, plens, clens, stop_ids + [pad_id])
    print(f"  loss={float(loss):.6f} valid={info['valid_tokens']}")
    loss.backward()
    print("  grads:")
    for n, p in pairs[:3] + pairs[-1:]:
        stats(p.grad if p.grad is not None else torch.zeros(1), f"grad {n[-40:]}")
    gn = torch.nn.utils.clip_grad_norm_([p for _, p in pairs], 1e9)
    print(f"  grad_norm={float(gn):.4f}")
    print("  gamma before step:")
    stats(pairs[0][1], "gamma0")
    opt.step()
    print("  gamma after step:")
    stats(pairs[0][1], "gamma0")
    for n, p in pairs:
        if bool(torch.isnan(p).any()):
            print(f"  !! NaN gamma in {n}")
            break

    # --- 4) forward after step ---
    with torch.no_grad():
        out3 = body(input_ids=ids_t, attention_mask=mask_t, use_cache=False)
    print("  [after step] hidden:")
    stats(out3.last_hidden_state, "hidden")
    l2, _ = batch_entropy(body, head, ids_t, mask_t, plens, clens, stop_ids + [pad_id])
    print(f"  loss_after_step={float(l2):.6f}")


if __name__ == "__main__":
    main()
