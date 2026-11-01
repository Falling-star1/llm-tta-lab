"""EATA on the LoRA surface (q_proj / v_proj), which is what TLM's Appendix C.2 specifies.

TLM's implementation paragraph reads, verbatim:

    "For the LoRA setup, we use random Gaussian initialization for matrix A and set
     matrix B to zero, with a rank of r = 8. The LoRA is applied only to the Wq and Wv."

and the Tent/EATA descriptions that follow only say "adjust the parameters of the
LLM".  This file therefore builds the same adaptation surface as TLM, keeps EATA's
entropy-based sample selection, and leaves the rest of the evaluation protocol to
run.py so the numbers stay comparable with the frozen-source archive.

Usage:
    PY=/data/chuwenjing/conda-envs/tta-llm/bin/python
    CUDA_VISIBLE_DEVICES=2 $PY -B run_eata_lora.py formal \
        --model llama3b --dataset geosignal_5k --gpu 2 --batch 1 --lr 5e-5 \
        --reuse-source <archive>/llama3b__geosignal_5k \
        --out /data/chuwenjing/results/eata_lora/llama3b__geosignal_5k
"""
from __future__ import annotations

import argparse
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


os.environ["CUDA_VISIBLE_DEVICES"] = _pre_scan("--gpu", "0")

import torch  # noqa: E402

import run as tent_run  # noqa: E402
import run_eata  # noqa: E402
import eata_alg  # noqa: E402
import norm_affine  # noqa: E402

log = tent_run.log


def build_rt_lora(model_key, gpu, lora_r, lora_alpha, lora_dropout, target_modules):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from llamafactory.data.template import TEMPLATES
    from peft import LoraConfig, get_peft_model

    spec = tent_run.MODELS[model_key]
    tokenizer = AutoTokenizer.from_pretrained(spec["path"], trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        spec["path"], torch_dtype=torch.float16, trust_remote_code=False,
        attn_implementation="sdpa")
    model = model.to("cuda:0")
    model.config.use_cache = False
    for prm in model.parameters():
        prm.requires_grad_(False)

    cfg = LoraConfig(r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                     target_modules=[m for m in target_modules.split(",") if m],
                     bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, cfg)
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False})
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    pairs = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    n_train = int(sum(p.numel() for _, p in pairs))
    dtypes = sorted({str(p.dtype) for _, p in pairs})

    # PeftModel.__getattr__("model") forwards to LoraModel.model, i.e. the whole
    # causal LM -- not the transformer body that entropy.py expects. Resolve the
    # body/head explicitly and hand them to the adaptation loop.
    inner = model.base_model.model if hasattr(model, "base_model") else model
    body, head = inner.model, inner.lm_head
    run_eata.resolve_body_head = lambda _m: (body, head)

    template = TEMPLATES[spec["template"]]
    stop_ids = list(template.get_stop_token_ids(tokenizer))
    pad_id = tokenizer.pad_token_id
    model.eval()
    log("runtime_lora", model=model_key, trainable_params=n_train,
        trainable_tensors=len(pairs), dtypes=dtypes,
        targets=target_modules, r=lora_r, alpha=lora_alpha,
        body=type(body).__name__, head=type(head).__name__,
        stop_ids=stop_ids, pad_id=pad_id)
    return model, tokenizer, template, "", stop_ids, pad_id, pairs


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("formal", "eval_only"):
        p = sub.add_parser(name)
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
        p.add_argument("--e-margin", type=float, default=None)
        p.add_argument("--e0-ratio", type=float, default=0.4)
        p.add_argument("--e0-mode", default="ratio", choices=["ratio", "abs"])
        p.add_argument("--d-margin", type=float, default=1.0)
        p.add_argument("--fisher", type=int, default=0)
        p.add_argument("--fisher-alpha", type=float, default=2000.0)
        p.add_argument("--out", default=None)
        p.add_argument("--ctx", type=int, default=4096)
        p.add_argument("--grad-ckpt", default="on", choices=["on", "off"])
        p.add_argument("--reuse-source", default=None)
        p.add_argument("--no-resume", action="store_true")
    a = ap.parse_args()

    tent_run.MAX_INPUT = a.ctx
    tent_run.GRAD_CKPT = (a.grad_ckpt == "on")
    torch.use_deterministic_algorithms(True, warn_only=False)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.benchmark = False

    if a.out is None:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        a.out = f"/data/chuwenjing/results/eata_lora/{a.model}__{a.dataset}/{stamp}"

    task = tent_run.Task(a.model, a.dataset, a.out, a.limit, reuse_source=a.reuse_source)
    rt = build_rt_lora(a.model, a.gpu, a.lora_r, a.lora_alpha,
                       a.lora_dropout, a.targets)

    V = int(rt[0].lm_head.weight.shape[0])
    e_margin = (a.e_margin if a.e_margin is not None
                else (a.e0_ratio * math.log(V) if a.e0_mode == "ratio" else a.e0_ratio))
    # --d-margin 0 => S_div fully disabled (None), not a huge finite threshold.
    d_margin = None if a.d_margin <= 0 else a.d_margin
    log("eata_lora_config", vocab=V, e_margin=e_margin, d_margin=d_margin,
        lr=a.lr, opt=a.opt, batch=a.batch, targets=a.targets, r=a.lora_r)

    task.state.update({"method": "eata", "surface": "lora_qv", "lora_r": a.lora_r,
                       "lora_alpha": a.lora_alpha, "targets": a.targets,
                       "new_parameters": int(sum(p.numel() for _, p in rt[6])),
                       "ctx": tent_run.MAX_INPUT,
                       "adapt_max_new": tent_run.ADAPT_MAX_NEW,
                       "eval_max_new": tent_run.EVAL_MAX_NEW,
                       "lr": a.lr, "batch": a.batch, "opt": a.opt,
                       "vocab": V, "e_margin": e_margin, "d_margin": d_margin})
    task._save_state()

    if a.cmd == "formal":
        if a.reuse_source:
            tent_run.import_source(task, a.reuse_source)
        else:
            tent_run.run_source(task, rt)
        run_eata.adapt_eata(task, rt, a.batch, a.lr, a.opt, e_margin, a.d_margin,
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
