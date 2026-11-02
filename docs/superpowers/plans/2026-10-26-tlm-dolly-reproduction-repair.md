# TLM Dolly Reproduction Repair Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans for inline execution. Keep frozen v6 and all existing run directories unchanged.

**Goal:** Verify the replay baseline and determine whether role or dataset formatting targets cause the observed Dolly failure, while correcting demonstrable protocol and diagnostic gaps.

**Architecture:** A separate diagnostic imports the frozen v6 runtime. Four loss-target variants share input IDs, frozen-base CE weights, seed, 5000-row sampler and original scheduler. Stop at step 200 with an explicit callback rather than shortening the dataset or schedule. Existing evaluation workers retain their GPUs and output directories.

**Tech Stack:** Python 3.10, pinned PyTorch / Transformers / PEFT / Accelerate, stdlib CPU tests, existing secure SSH helper.

2026-10-27更新：200步两组基线已逐值重放，但不足以复现全量退化，因此对照延长到1000步。用户暂停后已于08:08明确要求继续，已恢复原进程，未重复启动。新增GSM8K只读检查点核查见[实施记录](2026-10-27-tlm-gsm-reproduction-audit.md)。

## Tasks

- [x] Test target partitions before implementation in `offline_tta/tlm_template_diagnostic_v1/tests/test_targets.py`: every target belongs to exactly one of role/wrapper/body; unknown scaffolds are rejected; truncated tails retain body tokens; causal targets use positions 1 onward; title-only outputs are recognized separately from empty strings.
- [x] Implement `targets.py` with canonical scaffold parsing, token-offset groups, masks and title-only detection. Run `python -B -m unittest discover -s offline_tta/tlm_template_diagnostic_v1/tests -v`; first observe missing-module failure, then passing behavior.
- [x] Implement `diagnose.py`: load and verify canonical frozen base; assert reconstructed token hashes; use frozen TLMTrainer unchanged for all-target baseline; use independently masked current-input CE for the three diagnostic variants. Preserve complete-input base CE and its strict selection/weights. Stop at 200 while leaving the full schedule at 5000. Save original-vs-replay history differences and step-100 tensor equality.
- [x] Measure a fixed first-20 plus seeded-random-80 input set for next-token argmax/probability, clearly separating probabilities from counts. Generate fixed held-out samples with the same 512-token greedy settings. Store all text, scores, masks, manifests and failure evidence in fresh directories.
- [x] Validate hashes and tests on server. Check idle GPUs and acquire the existing per-GPU lock before any diagnostic. Launch a baseline first; inspect its replay proof before interpreting masked variants. Block claims of causal repair when the short baseline does not reproduce the failure or original training trajectory.
- [x] Independently review the implementation and report the evidence. All eight 1000-step cases are now evaluated, including five evaluation-only recoveries whose AMP/LR differences remain explicitly recorded. Format/first-stop probes improve, while generated samples retain quality failures; this is not a full-quality repair. Protocol corrections remain separate and all original run records are preserved. See the dated diagnostic report for evidence.

The user's request already authorizes executing this repair; no additional training restart or replacement of main results is implied by a successful diagnostic.
