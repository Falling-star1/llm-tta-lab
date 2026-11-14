"""Strict Tent reproduction on RMSNorm LLMs (gamma only, source RMS statistic).

RMSNorm keeps no running statistics: its normalizer is the per-token RMS of the
input at hand, recomputed every forward. Tent's "replace the source statistics
with the current test batch's statistics" step therefore has nothing to act on
here, so the source RMSNorm forward is used unchanged and only gamma is adapted.

Usage (on the server, after `source activate_tlm.sh`):

    python run.py probe     --model llama3b --dataset gsm8k_5k --limit 8
    python run.py preflight --model llama3b --dataset gsm8k_5k --limit 8
    python run.py formal    --model llama3b --dataset geosignal_5k --batch 4 --gpu 0 --out DIR
    python run.py resume    --out DIR

Phases of `formal`, per Tent's offline setting:
    source  : frozen model (gamma untouched) generates the answer
    adapt   : 1 epoch over the stream; per batch -> self-generated 80-token prefix ->
              entropy at continuation positions -> 1 Adam step on gamma
    eval    : adapted model regenerates the answer
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
import contextlib
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent

GRAD_CKPT = True
DEBUG = False
LOSS_SCALE = 1.0      # loss/K with lr*K: same maths, smaller fp16 backward intermediates
SCHED = "cosine"      # official offline_ttl.yaml: cosine decay, warmup_ratio 0.1
WARMUP_RATIO = 0.1
TLM = Path("/home/chuwenjing/TTA_LLM/TLM")
sys.path.insert(0, str(TLM / "src"))
sys.path.insert(0, str(TLM / "scripts/eval"))

import os as _os  # noqa: E402
_os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import torch  # noqa: E402

import norm_affine  # noqa: E402
from norm_affine import drift_vs as _drift_vs
from entropy import batch_entropy, build_batch, entropy_backward, resolve_body_head  # noqa: E402

MODELS = {
    "llama3b": {"path": "/data/chuwenjing/models/Llama-3.2-3B-Instruct", "template": "llama3",
                "params": 175104},
    "qwen3b": {"path": "/data/chuwenjing/models/Qwen2.5-3B-Instruct", "template": "qwen",
               "params": 149504},
}
INPUTS = Path("/data/chuwenjing/tta_offline/tlm_inputs_v1")
ADAPT_MAX_NEW = 80
EVAL_MAX_NEW = 512
MAX_INPUT = 4096
ROUGE_MAX_CHARS = 100000


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def code_id():
    return hashlib.sha256(b"".join(sha256_file(p).encode() for p in
                                   sorted(HERE.glob("*.py")))).hexdigest()


def log(phase, **kw):
    print(json.dumps({"at": time.strftime("%Y-%m-%dT%H:%M:%S"), "phase": phase, **kw}), flush=True)


@contextlib.contextmanager
def eval_mode(model):
    was_training = model.training
    model.eval()
    try:
        yield model
    finally:
        model.train(was_training)


def read_jsonl(p):
    out = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def set_lr(opt, peak_lr, step, total_steps):
    """Official offline_ttl.yaml schedule: linear warmup, then cosine decay to zero."""
    if SCHED == "const" or total_steps <= 1:
        lr = peak_lr
    else:
        warm = max(1, int(total_steps * WARMUP_RATIO))
        if step < warm:
            lr = peak_lr * step / warm
        else:
            prog = (step - warm) / max(1, total_steps - warm)
            lr = peak_lr * 0.5 * (1.0 + math.cos(math.pi * prog))
    for group in opt.param_groups:
        group["lr"] = lr
    return lr


def repeated(text):
    """Same 4-gram rule as the TLM pipeline's quality.py."""
    tok = re.findall(r"\w+|[^\w\s]", (text or "").lower())
    if len(tok) < 32:
        return False
    grams = [tuple(tok[i:i + 4]) for i in range(len(tok) - 3)]
    return 1.0 - len(set(grams)) / len(grams) >= 0.5


def output_quality(recs, baseline=()):
    """Mode-collapse gate. Thresholds are engineering limits, not paper hyperparameters.

    Calibrated on the archived runs: healthy groups stay at dominant share <=0.011 and
    repetition excess <=0.001; every collapsed group exceeds one of the three limits.
    """
    n = max(1, len(recs))
    texts = Counter(" ".join((r.get("predict") or "").lower().split()) for r in recs)
    if not texts:
        return {"n": 0, "unique_answers": 0, "dominant_answer_share": 1.0,
                "dominant_answer": "", "repetition_rate": 1.0,
                "source_repetition_rate": None, "collapsed": True, "status": "COLLAPSED"}
    dominant, count = texts.most_common(1)[0]
    rep = sum(repeated(r.get("predict")) for r in recs) / n
    base = sum(repeated(r.get("predict")) for r in baseline) / max(1, len(baseline))
    q = {"n": len(recs), "unique_answers": len(texts),
         "dominant_answer_share": round(count / n, 4), "dominant_answer": dominant[:180],
         "repetition_rate": round(rep, 4),
         "source_repetition_rate": round(base, 4) if len(baseline) else None}
    q["collapsed"] = bool(count / n >= 0.5 or len(texts) < 0.05 * n
                          or (len(baseline) and rep - base > 0.05))
    q["status"] = "COLLAPSED" if q["collapsed"] else "OK"
    return q


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #
def build_runtime(model_key, gpu):
    """Load the model directly with transformers (ChatModel's device_map offloads to CPU)."""
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(gpu))
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from llamafactory.data.template import TEMPLATES

    spec = MODELS[model_key]
    tokenizer = AutoTokenizer.from_pretrained(spec["path"], trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        spec["path"], torch_dtype=torch.float16, trust_remote_code=False,
        attn_implementation="sdpa")
    assert not any(str(p.device) == "meta" for p in model.parameters()), "meta parameters present"
    model = model.to("cuda:0")
    model.config.use_cache = False
    template = TEMPLATES[spec["template"]]
    system = ""
    stop_ids = list(template.get_stop_token_ids(tokenizer))
    pad_id = tokenizer.pad_token_id
    model.eval()
    for prm in model.parameters():
        prm.requires_grad_(False)
    if GRAD_CKPT:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    n_patched = norm_affine.install(model)
    pairs = norm_affine.enable_norm_gamma(model)
    for _, prm in pairs:
        prm.data = prm.data.to(torch.float32)
    got = int(sum(p.numel() for _, p in pairs))
    assert got == spec["params"], f"trainable gamma count {got} != expected {spec['params']}"
    log("runtime", model=model_key, patched_norms=n_patched, gamma_params=got,
        stop_ids=stop_ids, pad_id=pad_id, dtype=str(next(model.parameters()).dtype),
        device=str(next(model.parameters()).device))
    return model, tokenizer, template, system, stop_ids, pad_id, pairs


def encode(template, tokenizer, row, system):
    text = row["instruction"] + ("\n" + row["input"] if row.get("input") else "")
    ids = template.encode_oneturn(tokenizer, [{"role": "user", "content": text},
                                              {"role": "assistant", "content": ""}],
                                  system=system)[0]
    ids = list(ids)[-MAX_INPUT:]                     # 固定尾部截断（与现有 pipeline 同口径）
    return text, ids


@torch.no_grad()
def generate(model, tokenizer, ids, limit, stop_ids, pad_id):
    from transformers import GenerationConfig

    inputs = torch.tensor([ids], dtype=torch.long, device=next(model.parameters()).device)
    cfg = GenerationConfig(max_new_tokens=limit, do_sample=False, num_beams=1,
                           repetition_penalty=1.0, eos_token_id=stop_ids, pad_token_id=pad_id)
    with eval_mode(model):
        seq = model.generate(input_ids=inputs, attention_mask=torch.ones_like(inputs), generation_config=cfg)
    gen = seq[0, len(ids):].tolist()
    return gen


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def rouge_setup():
    import nltk
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rougeLsum"], use_stemmer=True, split_summaries=True)

    def prep(text):
        text = (text or "").replace("<n>", " ")
        try:
            return " ".join(nltk.sent_tokenize(text))
        except Exception:
            return text

    def score(pred, ref):
        return float(scorer.score(prep(ref), prep(pred))["rougeLsum"].fmeasure)

    return score


def em_setup():
    import eval_utils

    def score(pred, ref):
        got = eval_utils.extract_gsm8k_answer_number(pred)
        return 1.0 if got is not None and str(got) == str(ref).strip().strip("\n") else 0.0

    return score


# --------------------------------------------------------------------------- #
# one task
# --------------------------------------------------------------------------- #
class Task:
    def __init__(self, model_key, dataset, out, limit=None, reuse_source=None):
        self.model_key, self.dataset = model_key, dataset
        self.out = Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        d = INPUTS / dataset
        self.rows = read_jsonl(d / "inputs.jsonl")
        self.refs = {r["source_index"]: r["output"] for r in read_jsonl(d / "references.jsonl")}
        if limit:
            self.rows = self.rows[:limit]
        self.metric = "exact_match" if dataset.startswith("gsm8k") else "rougeLsum"
        self.score = em_setup() if self.metric == "exact_match" else rouge_setup()
        self.manifest_path = self.out / "run_manifest.json"
        self.state = json.loads(self.manifest_path.read_text()) if self.manifest_path.exists() else {}
        self.state.update({"dataset": dataset, "model": model_key, "metric": self.metric,
                           "records": len(self.rows), "code_id": code_id()})
        self._save_state()
        self.reuse_source = reuse_source

    def _save_state(self):
        tmp = self.manifest_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, indent=2, ensure_ascii=False))
        tmp.replace(self.manifest_path)

    def journal(self, name):
        return self.out / name

    def done_count(self, name):
        p = self.journal(name)
        if not p.exists():
            return 0
        with open(p, encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())


def import_source(task, src_dir):
    """Reuse a validated frozen-model source pass (same protocol as the team's source_reuse.json)."""
    src = Path(src_dir) / "source.jsonl"
    assert src.exists(), f"source to reuse not found: {src}"
    their = {}
    for line in open(src, encoding="utf-8"):
        r = json.loads(line)
        their[r["source_index"]] = r
    missing = [r["source_index"] for r in task.rows if r["source_index"] not in their]
    assert not missing, f"{len(missing)} records missing in the reused source"
    out = task.journal("source.jsonl")
    with open(out, "w", encoding="utf-8") as f:
        for row in task.rows:
            f.write(json.dumps(their[row["source_index"]], ensure_ascii=False) + "\n")
    recs = [their[r["source_index"]] for r in task.rows]
    sc = [r["score"] for r in recs if r.get("score") is not None]
    task.state["source"] = {"n": len(recs), "value": sum(sc) / max(1, len(sc)),
                            "limit_hits": sum(bool(r.get("hit_generation_limit")) for r in recs),
                            "reused": True}
    task.state["source_reuse"] = {"path": str(src), "sha256": sha256_file(src),
                                  "matched_records": len(recs), "protocol": "A"}
    task._save_state()
    log("source_reused", value=task.state["source"]["value"], sha256=task.state["source_reuse"]["sha256"][:16])
    return task.state["source"]["value"]


def run_source(task, rt):
    model, tokenizer, template, system, stop_ids, pad_id, _ = rt
    path = task.journal("source.jsonl")
    done = task.done_count("source.jsonl")
    t0 = time.time()
    with open(path, "a", encoding="utf-8") as f:
        for i in range(done, len(task.rows)):
            row = task.rows[i]
            prompt, ids = encode(template, tokenizer, row, system)
            gen = generate(model, tokenizer, ids, EVAL_MAX_NEW, stop_ids, pad_id)
            pred = tokenizer.decode(gen, skip_special_tokens=True)
            ref = task.refs[row["source_index"]]
            rec = {"source_index": row["source_index"], "prompt_tokens": len(ids),
                   "predict": pred, "label": ref, "generated_tokens": len(gen),
                   "hit_generation_limit": len(gen) >= EVAL_MAX_NEW,
                   "score": task.score(pred, ref)}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            if (i + 1) % 100 == 0:
                log("source", done=i + 1, total=len(task.rows), secs=round(time.time() - t0, 1))
    recs = read_jsonl(path)
    task.state["source"] = {"n": len(recs), "value": sum(r["score"] for r in recs) / max(1, len(recs)),
                            "limit_hits": sum(r["hit_generation_limit"] for r in recs),
                            "wall_seconds": round(time.time() - t0, 1)}
    task._save_state()
    log("source_done", value=task.state["source"]["value"])
    return task.state["source"]["value"]


def adapt_batches(task, rt, batch, lr, ckpt_every=25, resume=True):
    """1 epoch; one Adam step per batch of self-generated prefixes."""
    model, tokenizer, template, system, stop_ids, pad_id, pairs = rt
    body, head = resolve_body_head(model)
    base_snap = norm_affine.snapshot(pairs)
    opt = torch.optim.Adam([p for _, p in pairs], lr=lr, betas=(0.9, 0.999),
                           eps=1e-8, weight_decay=0.0)
    journal = task.journal("adaptation.jsonl")
    start = task.done_count("adaptation.jsonl") if resume else 0
    idx = list(range(len(task.rows)))
    total_steps = len(range(0, len(idx), batch))
    steps = 0
    t0 = time.time()
    device = next(model.parameters()).device
    with open(journal, "a", encoding="utf-8") as f:
        for bstart in range(start, len(idx), batch):
            lr_now = set_lr(opt, lr, bstart // batch, total_steps)
            chunk = idx[bstart:bstart + batch]
            prompts, conts, metas = [], [], []
            for j in chunk:
                row = task.rows[j]
                _, ids = encode(template, tokenizer, row, system)
                gen = generate(model, tokenizer, ids, ADAPT_MAX_NEW, stop_ids, pad_id)
                if not gen:
                    gen = [pad_id]
                prompts.append(ids)
                conts.append(gen)
                metas.append(row["source_index"])
            ids_t, mask_t, plens, clens = build_batch(prompts, conts, pad_id)
            ids_t, mask_t = ids_t.to(device), mask_t.to(device)
            model.train()                                    # enable checkpointing
            opt.zero_grad(set_to_none=True)
            before, info, loss = entropy_backward(
                body, head, ids_t, mask_t, plens, clens, stop_ids + [pad_id])
            gnorm = float(torch.nn.utils.clip_grad_norm_([p for _, p in pairs], 1e9))
            if not (gnorm == gnorm) or gnorm <= 0:
                raise RuntimeError(f"bad gamma gradient (gnorm={gnorm}) at step {steps}")
            opt.step()
            after = None
            with torch.no_grad():
                after = float(batch_entropy(body, head, ids_t, mask_t, plens, clens,
                                            stop_ids + [pad_id])[0])
            steps += 1
            drift = _drift_vs(pairs, base_snap)
            rec = {"step": steps, "samples": metas, "entropy_before": before,
                   "entropy_after": after, "grad_norm": gnorm,
                   "mean_abs_gamma_delta": drift["mean_abs_delta"],
                   "max_abs_gamma_delta": drift["max_abs_delta"],
                   "valid_tokens": sum(info["valid_tokens"]), "lr": lr_now,
                   "wall_seconds": round(time.time() - t0, 1)}
            f.write(json.dumps(rec) + "\n")
            f.flush()
            if steps % ckpt_every == 0:
                torch.save({"gamma": norm_affine.snapshot(pairs), "opt": opt.state_dict(),
                            "step": steps, "cursor": bstart + len(chunk)},
                           task.out / "ckpt.pt")
            if steps % 10 == 0:
                log("adapt", step=steps, entropy_before=round(before, 4),
                    entropy_after=round(after, 4), gnorm=round(gnorm, 3),
                    gamma_delta=round(drift["mean_abs_delta"], 6))
    recs = read_jsonl(journal)
    task.state["adaptation"] = {"steps": len(recs), "batch": batch, "lr": lr,
                                "sched": SCHED, "warmup_ratio": WARMUP_RATIO,
                                "entropy_first": recs[0]["entropy_before"] if recs else None,
                                "entropy_last": recs[-1]["entropy_after"] if recs else None,
                                "gamma_delta_last": recs[-1]["mean_abs_gamma_delta"] if recs else None,
                                "wall_seconds": round(time.time() - t0, 1)}
    task._save_state()
    log("adapt_done", **{k: v for k, v in task.state["adaptation"].items()})


def run_eval(task, rt):
    model, tokenizer, template, system, stop_ids, pad_id, _ = rt
    path = task.journal("tent.jsonl")
    done = task.done_count("tent.jsonl")
    t0 = time.time()
    with open(path, "a", encoding="utf-8") as f:
        for i in range(done, len(task.rows)):
            row = task.rows[i]
            prompt, ids = encode(template, tokenizer, row, system)
            gen = generate(model, tokenizer, ids, EVAL_MAX_NEW, stop_ids, pad_id)
            pred = tokenizer.decode(gen, skip_special_tokens=True)
            ref = task.refs[row["source_index"]]
            rec = {"source_index": row["source_index"], "prompt_tokens": len(ids),
                   "predict": pred, "label": ref, "generated_tokens": len(gen),
                   "hit_generation_limit": len(gen) >= EVAL_MAX_NEW,
                   "score": task.score(pred, ref)}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            if (i + 1) % 100 == 0:
                log("eval", done=i + 1, total=len(task.rows), secs=round(time.time() - t0, 1))
    recs = read_jsonl(path)
    src_path = task.journal("source.jsonl")
    baseline = read_jsonl(src_path) if src_path.exists() else []
    quality = output_quality(recs, baseline)
    task.state["tent"] = {"n": len(recs), "value": sum(r["score"] for r in recs) / max(1, len(recs)),
                          "limit_hits": sum(r["hit_generation_limit"] for r in recs),
                          "wall_seconds": round(time.time() - t0, 1)}
    task.state["quality"] = quality
    task.state["complete"] = True
    task.state["status"] = "COLLAPSED" if quality["collapsed"] else "PASS"
    task._save_state()
    log("eval_done", value=task.state["tent"]["value"], quality=quality["status"],
        unique_answers=quality["unique_answers"],
        dominant_share=quality["dominant_answer_share"],
        repetition_rate=quality["repetition_rate"])
    return task.state["tent"]["value"]


def _adapt_events(task):
    p = task.journal("adaptation.jsonl")
    return read_jsonl(p) if p.exists() else []


# --------------------------------------------------------------------------- #
# probe / preflight
# --------------------------------------------------------------------------- #
def probe(model_key, dataset, gpu, limit, ctx_tokens, synth=False, only_b=None):
    model, tokenizer, template, system, stop_ids, pad_id, pairs = build_runtime(model_key, gpu)
    body, head = resolve_body_head(model)
    cand = read_jsonl(INPUTS / dataset / "inputs.jsonl")[:64]
    enc = sorted(((len(encode(template, tokenizer, r, system)[1]), r) for r in cand),
                 key=lambda t: -t[0])[:limit]
    task_rows = [r for _, r in enc]
    prompts, conts = [], []
    for row in task_rows:
        _, ids = encode(template, tokenizer, row, system)
        gen = generate(model, tokenizer, ids, ADAPT_MAX_NEW, stop_ids, pad_id) or [pad_id]
        prompts.append(ids); conts.append(gen)
    if synth:
        base = max(prompts, key=len)
        body_ids = list(base)
        target = max(64, (ctx_tokens or MAX_INPUT) - ADAPT_MAX_NEW)
        while len(body_ids) < target:
            body_ids = body_ids + list(base)
        body_ids = body_ids[:target]
        prompts = [body_ids] * max(1, limit)
        log("probe_synth", prompt_tokens=target, rows=len(prompts))
    longest = max(len(p) for p in prompts)
    log("probe_start", longest_prompt=longest, rows=len(prompts))
    opt = torch.optim.Adam([p for _, p in pairs], lr=1e-4)
    results = []
    plan = [only_b] if only_b else [8, 6, 4, 3, 2, 1]
    for b in plan:
        if b is None or b > len(prompts):
            continue
        try:
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            sub_p, sub_c = prompts[:b], conts[:b]
            ids_t, mask_t, plens, clens = build_batch(sub_p, sub_c, pad_id)
            ids_t, mask_t = ids_t.to("cuda:0"), mask_t.to("cuda:0")
            model.train()
            opt.zero_grad(set_to_none=True)
            loss, _ = batch_entropy(body, head, ids_t, mask_t, plens, clens, stop_ids + [pad_id])
            loss.backward()
            opt.step()
            peak = torch.cuda.max_memory_allocated() / 1024**3
            results.append({"batch": b, "peak_gib": round(peak, 2), "tokens": int(mask_t.sum())})
            log("probe_ok", batch=b, peak_gib=round(peak, 2))
        except RuntimeError as exc:
            results.append({"batch": b, "error": str(exc)[:200]})
            log("probe_fail", batch=b, error=str(exc)[:120])
    ok = [r["batch"] for r in results if "error" not in r]
    print(json.dumps({"probe": results, "max_batch": max(ok) if ok else None}, indent=2))
    return results


def preflight(model_key, dataset, gpu, limit, batch, lr):
    """Multi-step lr gate: does gamma actually move, and does entropy drop?"""
    model, tokenizer, template, system, stop_ids, pad_id, pairs = build_runtime(model_key, gpu)
    body, head = resolve_body_head(model)
    cand = read_jsonl(INPUTS / dataset / "inputs.jsonl")[:64]
    rows = [r for _, r in sorted(((len(encode(template, tokenizer, x, system)[1]), x) for x in cand),
                                 key=lambda t: -t[0])[:limit]]
    prompts, conts = [], []
    for row in rows:
        _, ids = encode(template, tokenizer, row, system)
        gen = generate(model, tokenizer, ids, ADAPT_MAX_NEW, stop_ids, pad_id) or [pad_id]
        prompts.append(ids); conts.append(gen)

    base_snap = norm_affine.snapshot(pairs)
    opt = torch.optim.Adam([p for _, p in pairs], lr=lr)
    steps, drift_hist, ent_hist, gnorms = 0, [], [], []
    for bstart in range(0, len(prompts), batch):
        seg_p = prompts[bstart:bstart + batch]
        seg_c = conts[bstart:bstart + batch]
        if len(seg_p) < 2:
            continue
        ids_t, mask_t, plens, clens = build_batch(seg_p, seg_c, pad_id)
        ids_t, mask_t = ids_t.to("cuda:0"), mask_t.to("cuda:0")
        model.train()
        opt.zero_grad(set_to_none=True)
        loss, _ = batch_entropy(body, head, ids_t, mask_t, plens, clens, stop_ids + [pad_id])
        before = float(loss.detach())
        (loss / LOSS_SCALE).backward()
        gnorm = float(torch.nn.utils.clip_grad_norm_([p for _, p in pairs], 1e9))
        opt.step()
        with torch.no_grad():
            after = float(batch_entropy(body, head, ids_t, mask_t, plens, clens,
                                        stop_ids + [pad_id])[0])
        steps += 1
        gnorms.append(gnorm); ent_hist.append((before, after))
        if DEBUG:
            g0 = pairs[0][1].detach()
            print(json.dumps({"dbg_step": steps,
                              "ids_hash": hashlib.sha256(ids_t.cpu().numpy().tobytes()).hexdigest()[:12],
                              "mask_sum": int(mask_t.sum()), "L": int(ids_t.shape[1]),
                              "loss": round(before, 4), "gnorm": gnorm,
                              "g0_finite": int(torch.isfinite(g0).sum()), "g0_numel": g0.numel(),
                              "g0_min": float(g0.min()), "g0_max": float(g0.max())}), flush=True)
        drift_hist.append(_drift_vs(pairs, base_snap)["mean_abs_delta"])
        if steps % 5 == 0:
            log("preflight_step", step=steps, entropy_before=round(before, 4),
                entropy_after=round(after, 4), gnorm=round(gnorm, 3),
                gamma_delta=round(drift_hist[-1], 8))
    drift = drift_hist[-1] if drift_hist else 0.0
    drops = [1 - a / b for b, a in ent_hist]
    improved = sum(1 for d in drops if d > 0)
    report = {
        "model": model_key, "dataset": dataset, "batch": batch, "lr": lr, "steps": steps,
        "gamma_params": sum(p.numel() for _, p in pairs),
        "frozen_others": all(not p.requires_grad for n, p in model.named_parameters()
                             if not (n.endswith("layernorm.weight") or n.endswith("norm.weight"))),
        "grad_norm_min": min(gnorms) if gnorms else None,
        "grad_norm_max": max(gnorms) if gnorms else None,
        "gamma_drift_final": drift,
        "entropy_first": ent_hist[0] if ent_hist else None,
        "entropy_last": ent_hist[-1] if ent_hist else None,
        "steps_entropy_improved": f"{improved}/{len(drops)}",
        "peak_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 2),
        "gate_gamma_moved": drift >= 1e-4,
        "gate_no_nan": all(g == g and g > 0 for g in gnorms),
        "gate_memory_headroom": torch.cuda.max_memory_allocated() / 1024**3 < 10.0,
    }
    report["PREFLIGHT_PASS"] = bool(report["gate_gamma_moved"] and report["gate_no_nan"]
                                    and report["gate_memory_headroom"] and improved >= max(1, len(drops) // 2))
    print(json.dumps(report, indent=2))
    return report


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("probe", "preflight", "formal", "resume"):
        p = sub.add_parser(name)
        p.add_argument("--model", default="llama3b", choices=list(MODELS))
        p.add_argument("--dataset", default="gsm8k_5k")
        p.add_argument("--gpu", default="0")
        p.add_argument("--limit", type=int, default=None)
        p.add_argument("--batch", type=int, default=2)
        p.add_argument("--lr", type=float, default=None)
        p.add_argument("--out", default=None)
        p.add_argument("--ctx", type=int, default=4096)
        p.add_argument("--grad-ckpt", default="on", choices=["on", "off"])
        p.add_argument("--synth", action="store_true")
        p.add_argument("--only-b", type=int, default=None)
        p.add_argument("--loss-scale", type=float, default=1.0)
        p.add_argument("--debug", action="store_true")
        p.add_argument("--reuse-source", default=None, help="dir containing source.jsonl to reuse")
        p.add_argument("--sched", default="cosine", choices=["cosine", "const"])
        p.add_argument("--warmup-ratio", type=float, default=0.1)
    a = ap.parse_args()

    global MAX_INPUT, GRAD_CKPT, LOSS_SCALE, DEBUG, SCHED, WARMUP_RATIO
    MAX_INPUT = a.ctx
    GRAD_CKPT = (a.grad_ckpt == "on")
    LOSS_SCALE = float(a.loss_scale)
    DEBUG = bool(getattr(a, "debug", False))
    SCHED = a.sched
    WARMUP_RATIO = a.warmup_ratio
    torch.use_deterministic_algorithms(True, warn_only=False)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.benchmark = False

    if a.cmd == "probe":
        probe(a.model, a.dataset, a.gpu, a.limit or 4, a.ctx, synth=a.synth, only_b=a.only_b)
        return

    batch = a.batch
    lr = a.lr if a.lr is not None else 1e-3 * batch / 128.0

    if a.cmd == "preflight":
        preflight(a.model, a.dataset, a.gpu, a.limit or 8, batch, lr)
        return

    if a.out is None:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        a.out = f"/data/chuwenjing/results/tent_paper/{a.model}__{a.dataset}/{stamp}"
    task = Task(a.model, a.dataset, a.out, a.limit, reuse_source=a.reuse_source)
    task.state.update({"lr": lr, "batch": batch, "ctx": MAX_INPUT,
                       "adapt_max_new": ADAPT_MAX_NEW, "eval_max_new": EVAL_MAX_NEW,
                       "sched": SCHED, "warmup_ratio": WARMUP_RATIO,
                       "surface": "norm_gamma_only", "new_parameters": 0, "stats_mode": "rms_per_token"})
    task._save_state()
    rt = build_runtime(a.model, a.gpu)
    if a.cmd == "formal":
        if a.reuse_source:
            import_source(task, a.reuse_source)
        else:
            run_source(task, rt)
        adapt_batches(task, rt, batch, lr)
        run_eval(task, rt)
    else:
        run_eval(task, rt)
    log("task_done", out=str(task.out), manifest=str(task.manifest_path))


if __name__ == "__main__":
    main()
