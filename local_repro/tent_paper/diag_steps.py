"""Multi-step diagnostic: find which step/batch introduces NaN."""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, "/home/chuwenjing/TTA_LLM/TLM/src")
sys.path.insert(0, "/home/chuwenjing/TTA_LLM/TLM/scripts/eval")

import torch

import norm_affine as na
from entropy import batch_entropy, build_batch, resolve_body_head
from run import INPUTS, build_runtime, encode, generate, read_jsonl


def gstats(pairs):
    lo = min(float(p.detach().min()) for _, p in pairs)
    hi = max(float(p.detach().max()) for _, p in pairs)
    bad = [n for n, p in pairs if bool((~torch.isfinite(p.detach())).any())]
    return lo, hi, bad


def main():
    n_samples = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    batch = 2
    lr = float(sys.argv[2]) if len(sys.argv) > 2 else 1.5625e-5
    model, tok, template, system, stop_ids, pad_id, pairs = build_runtime("llama3b", 0)
    body, head = resolve_body_head(model)
    cand = read_jsonl(INPUTS / "gsm8k_5k" / "inputs.jsonl")[:32]
    rows = [r for _, r in sorted(((len(encode(template, tok, x, system)[1]), x) for x in cand),
                                 key=lambda t: -t[0])[:n_samples]]
    prompts, conts = [], []
    na.set_active_mask(None)
    for r in rows:
        _, ids = encode(template, tok, r, system)
        prompts.append(ids)
        conts.append(generate(model, tok, ids, 80, stop_ids, pad_id) or [pad_id])
    print("lens:", [(len(p), len(c)) for p, c in zip(prompts, conts)])

    snap = {n: p.detach().clone() for n, p in pairs}
    opt = torch.optim.Adam([p for _, p in pairs], lr=lr)
    na.set_active_mask(None)
    for step, start in enumerate(range(0, len(prompts), batch)):
        sp, sc = prompts[start:start + batch], conts[start:start + batch]
        if len(sp) < 2:
            break
        ids_t, mask_t, plens, clens = build_batch(sp, sc, pad_id)
        ids_t, mask_t = ids_t.to("cuda:0"), mask_t.to("cuda:0")
        na.set_active_mask(mask_t)
        model.train()
        opt.zero_grad(set_to_none=True)
        loss, info = batch_entropy(body, head, ids_t, mask_t, plens, clens, stop_ids + [pad_id])
        b = float(loss.detach())
        loss.backward()
        gn = float(torch.nn.utils.clip_grad_norm_([p for _, p in pairs], 1e9))
        opt.step()
        lo, hi, bad = gstats(pairs)
        with torch.no_grad():
            a = float(batch_entropy(body, head, ids_t, mask_t, plens, clens, stop_ids + [pad_id])[0])
        print(json.dumps({"step": step, "lr": lr, "L": int(ids_t.shape[1]), "plens": plens, "clens": clens,
                          "loss_before": round(b, 4), "loss_after": round(a, 4),
                          "gnorm": round(gn, 2), "gamma_min": round(lo, 5),
                          "gamma_max": round(hi, 5), "bad_gamma": bad[:2], "delta_mean": round(sum(float((p.detach()-snap[n]).abs().mean()) for n,p in pairs)/len(pairs), 9),
                          "delta_max": round(max(float((p.detach()-snap[n]).abs().max()) for n,p in pairs), 9),
                          "diag": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in na.DIAG.items()}}))


if __name__ == "__main__":
    main()
