"""EATA text adaptation: drop-in companion to ../tent_paper/run.py.

Everything that determines comparability with the existing Tent numbers --
the input encode, the source pass, the R-Lsum / EM scoring, the manifest --
is imported from run.py, not re-implemented. Only the adaptation loop is new.

Usage (server, after `source .../stage1/activate_tlm.sh`):

    PY=/data/chuwenjing/conda-envs/tta-llm/bin/python
    CUDA_VISIBLE_DEVICES=0 $PY -B run_eata.py formal \
        --model llama3b --dataset geosignal_5k --batch 2 --gpu 0 \
        --reuse-source /data/chuwenjing/results/archives/tent_lora_b1_v3_20261020_001210/run_20261020_001210_900089/llama3b__geosignal_5k \
        --out /data/chuwenjing/results/eata_text/llama3b__geosignal_5k/smoke
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
TENT_DIR = HERE.parent / "tent_paper"
sys.path.insert(0, str(TENT_DIR))
sys.path.insert(0, str(HERE))


def _pre_scan(flag: str, default: str) -> str:
    argv = sys.argv[1:]
    for i, a in enumerate(argv):
        if a == flag and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith(flag + "="):
            return a.split("=", 1)[1]
    return default


# run.py pins CUDA_VISIBLE_DEVICES with setdefault at import time, so it must be
# set before the import or --gpu is silently ignored (same trap as _start_runs.sh).
os.environ["CUDA_VISIBLE_DEVICES"] = _pre_scan("--gpu", "0")

import torch  # noqa: E402

import run as tent_run  # noqa: E402
import norm_affine  # noqa: E402
import eata_alg  # noqa: E402
from entropy import build_batch, resolve_body_head  # noqa: E402

log = tent_run.log
read_jsonl = tent_run.read_jsonl


def adapt_eata(task, rt, batch, lr, opt_name, e_margin, d_margin,
               fisher_samples=0, fisher_alpha=2000.0, ckpt_every=25, resume=True):
    model, tokenizer, template, system, stop_ids, pad_id, pairs = rt
    body, head = resolve_body_head(model)
    device = next(model.parameters()).device
    base_snap = norm_affine.snapshot(pairs)
    params = [p for _, p in pairs]

    journal = task.journal("adaptation_eata.jsonl")
    start = task.done_count("adaptation_eata.jsonl") if resume else 0

    if opt_name == "sgd":
        opt = torch.optim.SGD(params, lr=lr, momentum=0.9)
    else:
        opt = torch.optim.Adam(params, lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    total_steps = len(range(0, len(task.rows), batch))

    fishers = {}
    if fisher_samples > 0:
        log("fisher_start", n=fisher_samples)
        for bstart in range(0, fisher_samples, batch):
            chunk = task.rows[bstart:bstart + batch]
            if not chunk:
                break
            prompts, conts = [], []
            for row in chunk:
                _, ids = tent_run.encode(template, tokenizer, row, system)
                gen = tent_run.generate(model, tokenizer, ids, tent_run.ADAPT_MAX_NEW,
                                        stop_ids, pad_id) or [pad_id]
                prompts.append(ids)
                conts.append(gen)
            ids_t, mask_t, plens, clens = build_batch(prompts, conts, pad_id)
            model.train()
            opt.zero_grad(set_to_none=True)
            f_batch = eata_alg.compute_fisher(body, head, ids_t.to(device), mask_t.to(device),
                                              plens, clens, stop_ids + [pad_id], pairs,
                                              n_batches=1)
            for name, (f, p0) in f_batch.items():
                if name in fishers:
                    fishers[name] = ((fishers[name][0] * 0.5 + f * 0.5), fishers[name][1])
                else:
                    fishers[name] = (f * 0.5, p0)
            opt.zero_grad(set_to_none=True)
        log("fisher_done", params=len(fishers))

    ema = None
    idx = list(range(len(task.rows)))
    steps = 0
    backwards = 0
    skipped = 0
    degenerate = 0
    t0 = time.time()
    with open(journal, "a", encoding="utf-8") as f:
        for bstart in range(start, len(idx), batch):
            lr_now = tent_run.set_lr(opt, lr, bstart // batch, total_steps)
            chunk = idx[bstart:bstart + batch]
            prompts, conts, metas = [], [], []
            for j in chunk:
                row = task.rows[j]
                _, ids = tent_run.encode(template, tokenizer, row, system)
                gen = tent_run.generate(model, tokenizer, ids, tent_run.ADAPT_MAX_NEW,
                                        stop_ids, pad_id)
                if not gen:
                    gen = [pad_id]
                prompts.append(ids)
                conts.append(gen)
                metas.append(row["source_index"])
            ids_t, mask_t, plens, clens = build_batch(prompts, conts, pad_id)
            ids_t, mask_t = ids_t.to(device), mask_t.to(device)

            model.train()                       # enable gradient checkpointing
            opt.zero_grad(set_to_none=True)

            per_sample, probs, _ = eata_alg.forward_entropy_probs(
                body, head, ids_t, mask_t, plens, clens, stop_ids + [pad_id])
            entropy_before = float(per_sample.detach().mean())

            e_sel, p_sel, new_ema, sel_stats = eata_alg.eata_select(
                per_sample, probs, e_margin, d_margin, ema)

            loss_value = None
            coeff_mean = None
            if e_sel is not None and e_sel.numel() > 0:
                loss, coeff_mean = eata_alg.eata_loss(e_sel, e_margin)
                pen = eata_alg.fisher_penalty(pairs, fishers, fisher_alpha)
                if pen is not None:
                    loss = loss + pen
                loss.backward()
                gnorm = float(torch.nn.utils.clip_grad_norm_(params, 1e9))
                if not (gnorm == gnorm):
                    raise RuntimeError(f"nan gradient (gnorm={gnorm}) at step {steps}")
                if gnorm <= 0.0:
                    # The next-token distribution is exactly one-hot, so the entropy gradient
                    # is exactly zero: there is nothing to learn. Skip the step instead of
                    # treating it as a failure (this is what killed qwen3b/geo at step 125).
                    skipped += 1
                    degenerate += 1
                else:
                    opt.step()
                    backwards += 1
                    loss_value = float(loss.detach())
            else:
                skipped += 1
                gnorm = 0.0
            opt.zero_grad(set_to_none=True)
            ema = new_ema

            steps += 1
            drift = norm_affine.drift_vs(pairs, base_snap)
            rec = {"step": steps, "samples": metas, "e_margin": e_margin,
                   "d_margin": d_margin, "entropy_before": entropy_before,
                   "loss": loss_value, "coeff_mean": coeff_mean,
                   "grad_norm": gnorm, "did_backward": bool(loss_value is not None),
                   "mean_abs_gamma_delta": drift["mean_abs_delta"],
                   "max_abs_gamma_delta": drift["max_abs_delta"],
                   "lr": lr_now, "opt": opt_name,
                   "wall_seconds": round(time.time() - t0, 1)}
            rec.update({f"sel_{k}": v for k, v in sel_stats.items() if v is not None})
            f.write(json.dumps(rec) + "\n")
            f.flush()
            if steps % ckpt_every == 0:
                torch.save({"gamma": norm_affine.snapshot(pairs), "opt": opt.state_dict(),
                            "step": steps, "cursor": bstart + len(chunk)},
                           task.out / "ckpt_eata.pt")
            if steps % 10 == 0:
                log("adapt", step=steps, entropy=round(entropy_before, 4),
                    loss=loss_value, gnorm=round(gnorm, 3),
                    n_sel=sel_stats.get("n_selected"), backwards=backwards, skipped=skipped,
                    gamma_delta=round(drift["mean_abs_delta"], 6))

    recs = read_jsonl(journal)
    task.state["adaptation"] = {
        "method": "eata", "steps": len(recs), "batch": batch, "lr": lr, "opt": opt_name,
        "sched": tent_run.SCHED, "warmup_ratio": tent_run.WARMUP_RATIO,
        "e_margin": e_margin, "d_margin": d_margin, "fisher_samples": fisher_samples,
        "fisher_alpha": fisher_alpha,
        "backwards": backwards, "skipped_batches": skipped,
        "degenerate_zero_grad": degenerate,
        "backwards_per_sample": backwards / max(1, len(idx)),
        "entropy_first": recs[0]["entropy_before"] if recs else None,
        "entropy_last": recs[-1]["entropy_before"] if recs else None,
        "gamma_delta_last": recs[-1]["mean_abs_gamma_delta"] if recs else None,
        "wall_seconds": round(time.time() - t0, 1),
    }
    task._save_state()
    log("adapt_done", **task.state["adaptation"])
    return task.state["adaptation"]


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("formal", "eval_only"):
        p = sub.add_parser(name)
        p.add_argument("--model", default="llama3b", choices=list(tent_run.MODELS))
        p.add_argument("--dataset", default="geosignal_5k")
        p.add_argument("--gpu", default="0")
        p.add_argument("--limit", type=int, default=None)
        p.add_argument("--batch", type=int, default=2)
        p.add_argument("--lr", type=float, default=2.5e-4)
        p.add_argument("--opt", default="sgd", choices=["sgd", "adam"])
        p.add_argument("--e-margin", type=float, default=None,
                       help="absolute E0 in nats; default 0.4*log(V)")
        p.add_argument("--e0-ratio", type=float, default=0.4)
        p.add_argument("--e0-mode", default="ratio", choices=["ratio", "abs"])
        p.add_argument("--d-margin", type=float, default=0.05)
        p.add_argument("--fisher", type=int, default=0)
        p.add_argument("--fisher-alpha", type=float, default=2000.0)
        p.add_argument("--out", default=None)
        p.add_argument("--ctx", type=int, default=4096)
        p.add_argument("--grad-ckpt", default="on", choices=["on", "off"])
        p.add_argument("--reuse-source", default=None)
        p.add_argument("--reuse-adapt", default=None)
        p.add_argument("--no-resume", action="store_true")
        p.add_argument("--sched", default="cosine", choices=["cosine", "const"])
        p.add_argument("--warmup-ratio", type=float, default=0.1)
    a = ap.parse_args()

    tent_run.MAX_INPUT = a.ctx
    tent_run.GRAD_CKPT = (a.grad_ckpt == "on")
    tent_run.SCHED = a.sched
    tent_run.WARMUP_RATIO = a.warmup_ratio
    torch.use_deterministic_algorithms(True, warn_only=False)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.benchmark = False

    if a.out is None:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        a.out = f"/data/chuwenjing/results/eata_text/{a.model}__{a.dataset}/{stamp}"

    task = tent_run.Task(a.model, a.dataset, a.out, a.limit, reuse_source=a.reuse_source)
    rt = tent_run.build_runtime(a.model, a.gpu)

    V = int(rt[0].lm_head.weight.shape[0])
    e_margin = (a.e_margin if a.e_margin is not None
                else (a.e0_ratio * math.log(V) if a.e0_mode == "ratio" else a.e0_ratio))
    log("eata_config", vocab=V, e_margin=e_margin, d_margin=a.d_margin,
        lr=a.lr, opt=a.opt, batch=a.batch, sched=a.sched, warmup_ratio=a.warmup_ratio)

    task.state.update({"method": "eata", "surface": "norm_gamma_only", "new_parameters": 0,
                       "stats_mode": "rms_per_token", "ctx": tent_run.MAX_INPUT,
                       "adapt_max_new": tent_run.ADAPT_MAX_NEW,
                       "eval_max_new": tent_run.EVAL_MAX_NEW,
                       "lr": a.lr, "batch": a.batch, "opt": a.opt,
                       "sched": a.sched, "warmup_ratio": a.warmup_ratio,
                       "vocab": V, "e_margin": e_margin, "d_margin": a.d_margin})
    task._save_state()

    if a.cmd == "formal":
        if a.reuse_source:
            tent_run.import_source(task, a.reuse_source)
        else:
            tent_run.run_source(task, rt)
        adapt_eata(task, rt, a.batch, a.lr, a.opt, e_margin, a.d_margin,
                   fisher_samples=a.fisher, fisher_alpha=a.fisher_alpha,
                   resume=not a.no_resume)

    value = tent_run.run_eval(task, rt)
    if "tent" in task.state:
        task.state["eata"] = task.state.pop("tent")
        task._save_state()
    log("eata_done", value=value)
    log("task_done", out=str(task.out), manifest=str(task.manifest_path))


if __name__ == "__main__":
    main()
