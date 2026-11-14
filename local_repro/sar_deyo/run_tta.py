"""SAR (ICLR 2023) and DeYO (ICLR 2024) on the same text protocol as Tent/EATA.

Both methods are ports of the official image-classification implementations onto the
project's frozen protocol: the input encode, the source pass, the ROUGE-Lsum / EM
scoring and the manifest all come from `../tent_paper/run.py`, and the adaptation
surface is one of the two already-frozen choices (`norm_gamma_only` / `lora_qv`).
Only the adaptation loop is new.

What is taken from each paper verbatim
--------------------------------------
SAR  (github.com/mr-eggplant/SAR, sar.py + sam.py)
  * selective entropy minimisation: keep samples with E(x;Theta) < E0, mean over kept.
  * SAM: epsilon = rho * g/||g||_2, gradient taken at Theta+epsilon, applied at Theta.
  * the filter is re-applied at Theta+epsilon before the second backward.
  * model recovery: EMA(0.9) of the second loss; reset to source weights when EMA < e0.
  * predictions come from the pre-update forward (sar.py:77, returned at :106).

DeYO (github.com/Jhyun17/DeYO, methods/deyo.py)
  * two-stage AND filter: Ent < tau_Ent (deyo.py:103) and PLPD > tau_PLPD (:145).
  * weights alpha = 1/exp(Ent-Ent0) + 1/exp(-PLPD), both terms detached (:168-171).
  * loss = mean over the intersection of alpha * Ent (:172).
  * predictions come from the pre-update forward (deyo.py:96).

What is our decision, because the papers have no text answer
------------------------------------------------------------
  * DeYO's x' is a block-shuffle of the input document, and PLPD averages the
    per-position pseudo-label probability over continuation positions.  See deyo_alg.py.
  * SAR's rho is passed as a relative perturbation of the adapted surface
    (`--rho-rel`), not as the paper's absolute 0.05, which was tuned on a ~5e4-parameter
    GroupNorm surface.
  * SAR's reset threshold is `--reset-ratio * E0` (paper: 0.2 / 2.76 = 0.072), because
    an absolute nats threshold does not transfer between 1000-way logits and a 15万-way
    next-token distribution.
  * on reset we follow the paper (`ema = None`) rather than the official code, where
    sar.py:52 writes the triggering low EMA back and the model resets for many steps.
  * the second forward is teacher-forced on the same continuation tokens, so the only
    difference between the two entropies is the weight perturbation (no re-decoding).
  * optimizer and lr stay on the project protocol (Adam, 5e-5 / 1e-6) instead of the
    papers' SGD(0.9), so the Delta against Tent/EATA does not mix in an optimizer change.

Usage (server, after `source .../stage1/activate_tlm.sh`, then cd back):

    PY=/data/chuwenjing/conda-envs/tta-llm/bin/python
    CUDA_VISIBLE_DEVICES=0 $PY -B run_tta.py formal --method sar --surface lora \
        --model llama3b --dataset geosignal_5k --gpu 0 --batch 1 --lr 5e-5 \
        --reuse-source <archive>/llama3b__geosignal_5k \
        --out /data/chuwenjing/results/sar_deyo/sar_lora_llama3b_geo
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
EATA_DIR = HERE.parent / "eata_text"
for p in (str(TENT_DIR), str(EATA_DIR), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)


def _pre_scan(flag: str, default: str) -> str:
    argv = sys.argv[1:]
    for i, a in enumerate(argv):
        if a == flag and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith(flag + "="):
            return a.split("=", 1)[1]
    return default


os.environ["CUDA_VISIBLE_DEVICES"] = _pre_scan("--gpu", "0")

import torch  # noqa: E402

import run as tent_run  # noqa: E402
import run_eata  # noqa: E402
import run_eata_lora  # noqa: E402
import eata_alg  # noqa: E402
import norm_affine  # noqa: E402
import sam  # noqa: E402
import deyo_alg  # noqa: E402
from entropy import build_batch  # noqa: E402

log = tent_run.log
read_jsonl = tent_run.read_jsonl


# --------------------------------------------------------------------------- #
# shared plumbing
# --------------------------------------------------------------------------- #
def collect_batch(task, rt, chunk):
    """Encode each row and greedily generate the 80-token adaptation continuation."""
    model, tokenizer, template, system, stop_ids, pad_id, _ = rt
    prompts, conts, metas, rows = [], [], [], []
    for j in chunk:
        row = task.rows[j]
        _, ids = tent_run.encode(template, tokenizer, row, system)
        gen = tent_run.generate(model, tokenizer, ids, tent_run.ADAPT_MAX_NEW, stop_ids, pad_id)
        if not gen:
            gen = [pad_id]
        prompts.append(ids)
        conts.append(gen)
        metas.append(row["source_index"])
        rows.append(row)
    ids_t, mask_t, plens, clens = build_batch(prompts, conts, pad_id)
    device = next(model.parameters()).device
    return ids_t.to(device), mask_t.to(device), plens, clens, metas, rows, conts


def new_optimizer(opt_name, params, lr):
    if opt_name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9)
    return torch.optim.Adam(params, lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)


def adapt_loop(task, rt, batch, peak_lr, method, make_step, ckpt_every=25, resume=True):
    model, pairs = rt[0], rt[6]
    base_snap = norm_affine.snapshot(pairs)
    name = f"adaptation_{method}.jsonl"
    journal = task.journal(name)
    start = task.done_count(name) if resume else 0
    idx = list(range(len(task.rows)))
    total_steps = len(range(0, len(idx), batch))
    steps = 0
    t0 = time.time()
    with open(journal, "a", encoding="utf-8") as f:
        for bstart in range(start, len(idx), batch):
            chunk = idx[bstart:bstart + batch]
            ids_t, mask_t, plens, clens, metas, rows, conts = collect_batch(task, rt, chunk)
            rec = make_step(ids_t, mask_t, plens, clens, rows, conts, bstart // batch, total_steps)
            steps += 1
            drift = norm_affine.drift_vs(pairs, base_snap)
            rec.update({"step": steps, "samples": metas,
                        "mean_abs_param_delta": drift["mean_abs_delta"],
                        "max_abs_param_delta": drift["max_abs_delta"],
                        "wall_seconds": round(time.time() - t0, 1)})
            f.write(json.dumps(rec) + "\n")
            f.flush()
            if steps % ckpt_every == 0:
                torch.save({"params": norm_affine.snapshot(pairs),
                            "step": steps, "cursor": bstart + len(chunk)},
                           task.out / f"ckpt_{method}.pt")
            if steps % 10 == 0:
                shown = {k: v for k, v in rec.items() if k not in ("samples", "wall_seconds")}
                log("adapt", **shown)
    return read_jsonl(journal)


# --------------------------------------------------------------------------- #
# SAR
# --------------------------------------------------------------------------- #
def make_sar_step(rt, peak_lr, opt_name, e_margin, rho, reset_e0):
    model = rt[0]
    body, head = run_eata.resolve_body_head(model)
    ignore = rt[4] + [rt[5]]
    params = [p for _, p in rt[6]]
    init = [p.detach().clone() for p in params]
    ctx = {"opt": sam.SAM(params, torch.optim.SGD if opt_name == "sgd" else torch.optim.Adam,
                          rho=rho, lr=peak_lr,
                          **(dict(momentum=0.9) if opt_name == "sgd"
                             else dict(betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0))),
           "ema": None, "resets": 0, "updates": 0, "skipped": 0}

    def step(ids_t, mask_t, plens, clens, rows, conts, step_index, total_steps):
        opt = ctx["opt"]
        lr_now = tent_run.set_lr(opt, peak_lr, step_index, total_steps)
        model.train()
        opt.zero_grad(set_to_none=True)

        ent1, _, _ = eata_alg.forward_entropy_probs(body, head, ids_t, mask_t, plens, clens, ignore)
        ent_before = float(ent1.detach().mean())
        keep1 = ent1 < e_margin
        n_kept1 = int(keep1.sum())

        gnorm = 0.0
        ent_after = None
        n_kept2 = 0
        if n_kept1:
            ent1[keep1].mean().backward()
            gnorm = float(torch.nn.utils.clip_grad_norm_(params, 1e9))
            if not (gnorm == gnorm):
                raise RuntimeError(f"nan gradient (gnorm={gnorm}) at step {step_index}")
            if gnorm > 0.0:
                opt.first_step(zero_grad=True)
                # same continuation tokens -> only the weights differ from the first pass
                ent2, _, _ = eata_alg.forward_entropy_probs(body, head, ids_t, mask_t,
                                                            plens, clens, ignore)
                keep2 = ent2 < e_margin
                n_kept2 = int(keep2.sum())
                if n_kept2:
                    loss2 = ent2[keep2].mean()
                    ent_after = float(loss2.detach())
                    ctx["ema"] = ent_after if ctx["ema"] is None else 0.9 * ctx["ema"] + 0.1 * ent_after
                    loss2.backward()
                    opt.second_step(zero_grad=True)
                    ctx["updates"] += 1
                else:
                    # restores Theta+eps -> Theta; the base step then has no gradient to apply
                    opt.second_step(zero_grad=True)
                    ctx["skipped"] += 1
            else:
                ctx["skipped"] += 1
        else:
            ctx["skipped"] += 1

        reset = bool(ctx["ema"] is not None and ctx["ema"] < reset_e0)
        if reset:
            with torch.no_grad():
                for p, p0 in zip(params, init):
                    p.copy_(p0)
            ctx["opt"] = sam.SAM(params, torch.optim.SGD if opt_name == "sgd" else torch.optim.Adam,
                                  rho=rho, lr=peak_lr,
                                  **(dict(momentum=0.9) if opt_name == "sgd"
                                     else dict(betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)))
            ctx["ema"] = None          # paper: e_m reset on recovery (official code does not)
            ctx["resets"] += 1

        return {"entropy_before": ent_before, "entropy_after": ent_after,
                "ema": ctx["ema"], "grad_norm": gnorm, "lr": lr_now,
                "n_kept_first": n_kept1, "n_kept_second": n_kept2,
                "reset": reset, "resets_total": ctx["resets"],
                "updates": ctx["updates"], "skipped": ctx["skipped"]}

    return step, ctx


# --------------------------------------------------------------------------- #
# DeYO
# --------------------------------------------------------------------------- #
def make_deyo_step(rt, peak_lr, opt_name, tau_ent, tau_plpd, ent0, chunks, seed):
    model, tokenizer, template, system, stop_ids, pad_id = rt[:6]
    body, head = run_eata.resolve_body_head(model)
    device = next(model.parameters()).device
    ignore = stop_ids + [pad_id]
    params = [p for _, p in rt[6]]
    opt = new_optimizer(opt_name, params, peak_lr)
    ctx = {"updates": 0, "skipped": 0}

    def step(ids_t, mask_t, plens, clens, rows, conts, step_index, total_steps):
        lr_now = tent_run.set_lr(opt, peak_lr, step_index, total_steps)
        model.train()
        opt.zero_grad(set_to_none=True)

        pos_ent, pos_maxp, pos_argmax, owner, valid, nb = deyo_alg.pos_stats(
            body, head, ids_t, mask_t, plens, clens, ignore)
        ent_ps = deyo_alg.per_sample_mean(pos_ent, owner, valid, nb)
        p_ps = deyo_alg.per_sample_mean(pos_maxp, owner, valid, nb).detach()
        ent_before = float(ent_ps.detach().mean())

        keep1 = ent_ps < tau_ent
        n1 = int(keep1.sum())
        rec = {"entropy_before": ent_before, "lr": lr_now, "n_pass_ent": n1,
               "n_pass_plpd": 0, "plpd_mean": None, "loss": None, "grad_norm": 0.0,
               "alpha_ent": None, "alpha_plpd": None}
        if n1 == 0:
            ctx["skipped"] += 1
            return rec

        # x' = block-shuffled document, scored on the same continuation tokens
        valid_list = valid.tolist()
        kept = [valid_list[i] for i, k in enumerate(keep1.tolist()) if k]   # batch indices
        sh_prompts = []
        for bi in kept:
            row = rows[bi]
            text = row["instruction"] + ("\n" + row["input"] if row.get("input") else "")
            shuffled = deyo_alg.block_shuffle(text, chunks, seed + int(row["source_index"]))
            _, ids_sh = tent_run.encode(template, tokenizer,
                                        {"instruction": shuffled, "input": ""}, system)
            sh_prompts.append(ids_sh)
        conts_kept = [conts[bi] for bi in kept]
        ids2, mask2, plens2, clens2 = build_batch(sh_prompts, conts_kept, pad_id)
        ids2, mask2 = ids2.to(device), mask2.to(device)

        keep_pos = torch.zeros(nb, dtype=torch.bool, device=owner.device)
        keep_pos[kept] = True
        targets = pos_argmax[keep_pos[owner]]
        p2_pos, owner2, valid2, nb2 = deyo_alg.prob_at(
            body, head, ids2, mask2, plens2, clens2, ignore, targets)
        p2_ps = deyo_alg.per_sample_mean(p2_pos, owner2, valid2, nb2)

        plpd = p_ps[keep1] - p2_ps
        keep2 = plpd > tau_plpd
        n2 = int(keep2.sum())
        rec["n_pass_plpd"] = n2
        rec["plpd_mean"] = float(plpd.mean())
        if n2 == 0:
            ctx["skipped"] += 1
            return rec

        ent_sel = ent_ps[keep1][keep2]
        a_ent = 1.0 / torch.exp(ent_sel.detach() - ent0)
        a_plpd = 1.0 / torch.exp(-plpd[keep2].detach())
        loss = ((a_ent + a_plpd) * ent_sel).mean()
        loss.backward()
        rec["grad_norm"] = float(torch.nn.utils.clip_grad_norm_(params, 1e9))
        if not (rec["grad_norm"] == rec["grad_norm"]):
            raise RuntimeError(f"nan gradient at step {step_index}")
        opt.step()
        opt.zero_grad(set_to_none=True)
        ctx["updates"] += 1
        rec["loss"] = float(loss.detach())
        rec["alpha_ent"] = float(a_ent.mean())
        rec["alpha_plpd"] = float(a_plpd.mean())
        return rec

    return step, ctx


@torch.no_grad()
def calibrate_deyo(task, rt, n_rows, chunks, seed, pad_id, log_fn=None):
    """Fix Ent0 and tau_PLPD from the *source* model's own output distribution.

    The paper's 0.4*lnV and 0.2 are calibrated for a 1000-way classifier whose entropy
    sits near 0.4*lnC.  On next-token distributions the per-sample entropy is 0.3-1.5
    nats, so 0.4*lnV = 4.70 puts exp(Ent0-Ent) near 80 instead of ~1, and PLPD averaged
    over 80 positions shrinks to a median of 0.066, so 0.2 filters ~87% of samples.
    Both constants are therefore replaced by the median of the same quantities measured
    on unadapted source predictions.  No reference answer is read: only inputs.
    """
    model, tokenizer, template, system, stop_ids = rt[:5]
    body, head = run_eata.resolve_body_head(model)
    device = next(model.parameters()).device
    ignore = stop_ids + [pad_id]
    ents, plpds = [], []
    n = min(int(n_rows), len(task.rows))
    for i in range(n):
        row = task.rows[i]
        text, ids = tent_run.encode(template, tokenizer, row, system)
        gen = tent_run.generate(model, tokenizer, ids, tent_run.ADAPT_MAX_NEW, stop_ids, pad_id)
        if not gen:
            gen = [pad_id]
        ids_t, mask_t, plens, clens = build_batch([ids], [gen], pad_id)
        ids_t, mask_t = ids_t.to(device), mask_t.to(device)
        pos_ent, pos_maxp, pos_argmax, owner, valid, nb = deyo_alg.pos_stats(
            body, head, ids_t, mask_t, plens, clens, ignore)
        p1 = float(deyo_alg.per_sample_mean(pos_maxp, owner, valid, nb)[0])
        ents.append(float(deyo_alg.per_sample_mean(pos_ent, owner, valid, nb)[0]))

        shuffled = deyo_alg.block_shuffle(text, chunks, seed + int(row["source_index"]))
        _, ids_sh = tent_run.encode(template, tokenizer,
                                    {"instruction": shuffled, "input": ""}, system)
        ids2, mask2, plens2, clens2 = build_batch([ids_sh], [gen], pad_id)
        ids2, mask2 = ids2.to(device), mask2.to(device)
        p2_pos, owner2, valid2, nb2 = deyo_alg.prob_at(
            body, head, ids2, mask2, plens2, clens2, ignore, pos_argmax)
        plpds.append(p1 - float(deyo_alg.per_sample_mean(p2_pos, owner2, valid2, nb2)[0]))
        if log_fn and (i + 1) % 20 == 0:
            log_fn("calib", done=i + 1, total=n)
    ents.sort()
    plpds.sort()
    med_e = ents[len(ents) // 2]
    med_p = plpds[len(plpds) // 2]
    stats = {"n": n, "ent_min": ents[0], "ent_med": med_e, "ent_max": ents[-1],
             "plpd_min": plpds[0], "plpd_med": med_p, "plpd_max": plpds[-1]}
    return med_e, med_p, stats


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("formal", "eval_only"):
        p = sub.add_parser(name)
        p.add_argument("--method", required=True, choices=["sar", "deyo"])
        p.add_argument("--surface", default="lora", choices=["lora", "gamma"])
        p.add_argument("--model", default="llama3b", choices=list(tent_run.MODELS))
        p.add_argument("--dataset", default="geosignal_5k")
        p.add_argument("--gpu", default="0")
        p.add_argument("--limit", type=int, default=None)
        p.add_argument("--batch", type=int, default=1)
        p.add_argument("--lr", type=float, default=5e-5)
        p.add_argument("--opt", default="adam", choices=["adam", "sgd"])
        p.add_argument("--lora-r", type=int, default=8)
        p.add_argument("--lora-alpha", type=int, default=16)
        p.add_argument("--lora-dropout", type=float, default=0.0)
        p.add_argument("--targets", default="q_proj,v_proj")
        p.add_argument("--e0-ratio", type=float, default=0.4,
                       help="SAR filter E0 = ratio*lnV (same form as the EATA group)")
        p.add_argument("--rho", type=float, default=-1.0,
                       help="SAR absolute L2 budget; <=0 means auto from --rho-rel")
        p.add_argument("--rho-rel", type=float, default=2.0e-4,
                       help="relative perturbation; paper 0.05 on a ~5e4-param GroupNorm "
                            "surface is about 2e-4 of ||theta||")
        p.add_argument("--reset-ratio", type=float, default=0.072,
                       help="SAR reset when EMA < ratio*E0 (paper 0.2/2.76=0.072)")
        p.add_argument("--tau-ent-ratio", type=float, default=0.5)
        p.add_argument("--tau-plpd", type=float, default=0.2)
        p.add_argument("--ent0-ratio", type=float, default=0.4)
        p.add_argument("--calib", default="auto", choices=["auto", "none"],
                       help="auto: replace Ent0 and tau_PLPD with the median of the same "
                            "quantities on unadapted source predictions (see calibrate_deyo)")
        p.add_argument("--calib-n", type=int, default=100)
        p.add_argument("--plpd-chunks", type=int, default=4)
        p.add_argument("--seed", type=int, default=42)
        p.add_argument("--out", default=None)
        p.add_argument("--ctx", type=int, default=4096)
        p.add_argument("--grad-ckpt", default="on", choices=["on", "off"])
        p.add_argument("--reuse-source", default=None)
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
        a.out = f"/data/chuwenjing/results/sar_deyo/{a.method}_{a.surface}_{a.model}__{a.dataset}/{stamp}"

    task = tent_run.Task(a.model, a.dataset, a.out, a.limit, reuse_source=a.reuse_source)
    if a.surface == "lora":
        rt = run_eata_lora.build_rt_lora(a.model, a.gpu, a.lora_r, a.lora_alpha,
                                        a.lora_dropout, a.targets)
    else:
        rt = tent_run.build_runtime(a.model, a.gpu)

    V = int(run_eata.resolve_body_head(rt[0])[1].weight.shape[0])
    e_margin = a.e0_ratio * math.log(V)
    tau_ent = a.tau_ent_ratio * math.log(V)
    ent0 = a.ent0_ratio * math.log(V)
    params = [p for _, p in rt[6]]
    theta_norm = float(torch.sqrt(sum((p.detach().float() ** 2).sum() for p in params)))
    rho = a.rho if a.rho > 0 else a.rho_rel * theta_norm
    reset_e0 = a.reset_ratio * e_margin

    log(f"{a.method}_config", vocab=V, e_margin=e_margin, tau_ent=tau_ent, tau_plpd=a.tau_plpd,
        ent0=ent0, rho=rho, theta_norm=theta_norm, rho_rel=rho / max(theta_norm, 1e-9),
        reset_e0=reset_e0, lr=a.lr, opt=a.opt, batch=a.batch, surface=a.surface,
        sched=a.sched, warmup_ratio=a.warmup_ratio)

    task.state.update({"method": a.method, "surface": a.surface,
                       "new_parameters": int(sum(p.numel() for p in params)),
                       "ctx": tent_run.MAX_INPUT,
                       "adapt_max_new": tent_run.ADAPT_MAX_NEW,
                       "eval_max_new": tent_run.EVAL_MAX_NEW,
                       "lr": a.lr, "batch": a.batch, "opt": a.opt,
                       "sched": a.sched, "warmup_ratio": a.warmup_ratio,
                       "vocab": V, "e_margin": e_margin})
    if a.surface == "lora":
        task.state.update({"lora_r": a.lora_r, "lora_alpha": a.lora_alpha,
                           "targets": a.targets})
    if a.method == "sar":
        task.state.update({"rho": rho, "rho_rel": rho / max(theta_norm, 1e-9),
                           "reset_e0": reset_e0})
    else:
        task.state.update({"tau_ent": tau_ent, "tau_plpd": a.tau_plpd, "ent0": ent0,
                           "plpd_chunks": a.plpd_chunks, "plpd_seed": a.seed,
                           "calib": a.calib, "calib_n": a.calib_n})
    task._save_state()

    if a.cmd == "formal":
        if a.reuse_source:
            tent_run.import_source(task, a.reuse_source)
        else:
            tent_run.run_source(task, rt)
        if a.method == "deyo" and a.calib == "auto":
            ent0, tau_plpd, cstats = calibrate_deyo(task, rt, a.calib_n, a.plpd_chunks,
                                                    a.seed, rt[5], log_fn=log)
            task.state["calibration"] = cstats
            task.state["tau_plpd"] = tau_plpd
            task.state["ent0"] = ent0
            task._save_state()
            log("calibrated", ent0=ent0, tau_plpd=tau_plpd, **cstats)
        else:
            tau_plpd = a.tau_plpd
        if a.method == "sar":
            step, ctx = make_sar_step(rt, a.lr, a.opt, e_margin, rho, reset_e0)
        else:
            step, ctx = make_deyo_step(rt, a.lr, a.opt, tau_ent, tau_plpd,
                                       ent0, a.plpd_chunks, a.seed)
        recs = adapt_loop(task, rt, a.batch, a.lr, a.method, step, resume=not a.no_resume)
        task.state["adaptation"] = {
            "method": a.method, "steps": len(recs), "batch": a.batch, "lr": a.lr,
            "sched": a.sched, "warmup_ratio": a.warmup_ratio,
            "entropy_first": recs[0]["entropy_before"] if recs else None,
            "entropy_last": recs[-1]["entropy_before"] if recs else None,
            "wall_seconds": recs[-1]["wall_seconds"] if recs else None,
        }
        task.state["adaptation"].update({k: v for k, v in ctx.items() if isinstance(v, (int, float))})
        task._save_state()

    value = tent_run.run_eval(task, rt)
    if "tent" in task.state:
        task.state[a.method] = task.state.pop("tent")
        task._save_state()
    log(f"{a.method}_done", value=value)
    log("task_done", out=str(task.out), manifest=str(task.manifest_path))


if __name__ == "__main__":
    main()
