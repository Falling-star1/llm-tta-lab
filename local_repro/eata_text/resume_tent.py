"""Resume the interrupted TENT eval (qwen3b / gsm8k) by restoring gamma from ckpt.pt.

The run was killed at 4259/5000 eval records. adaptation.jsonl is complete
(step=2500) and ckpt.pt holds the adapted gamma, so the cell can be finished
exactly instead of restarted.
"""
import os
import sys

OUT = "/data/chuwenjing/results/tent_paper/qwen3b__gsm8k_5k/20261028_124530"
TENT = "/home/chuwenjing/TTA_LLM/TLM/local_repro/tent_paper"
GPU = sys.argv[1] if len(sys.argv) > 1 else "5"

sys.path.insert(0, TENT)
os.environ["CUDA_VISIBLE_DEVICES"] = GPU

import torch  # noqa: E402

import run  # noqa: E402
import norm_affine  # noqa: E402


def main():
    torch.backends.cudnn.benchmark = False
    task = run.Task("qwen3b", "gsm8k_5k", OUT, None)
    rt = run.build_runtime("qwen3b", GPU)
    pairs = rt[6]

    ck = torch.load(os.path.join(OUT, "ckpt.pt"), map_location="cpu", weights_only=False)
    norm_affine.restore(pairs, ck["gamma"])
    run.log("gamma_restored", step=ck.get("step"), cursor=ck.get("cursor"),
            tensors=len(pairs))

    done = task.done_count("tent.jsonl")
    run.log("resume_eval", done=done, total=len(task.rows))
    value = run.run_eval(task, rt)
    task.state["resumed_from_ckpt"] = {"step": ck.get("step"), "resumed_at": done}
    task._save_state()
    run.log("resume_done", value=value)


if __name__ == "__main__":
    main()
