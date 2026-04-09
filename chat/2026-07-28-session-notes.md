# Session notes — 2026-07-28 (project built in this session)

Record of the conversation that produced this repo, so the project carries its
own history. Companion to the design spec at
`docs/superpowers/specs/2026-07-28-gradient-projection-design.md`.

## The ask

> Memory-matched full fine-tuning via gradient projection. LoRA's biggest
> real-world win is optimizer memory, so compete on that axis: project the
> gradients into a low-rank subspace (recomputed periodically via SVD) so
> Adam's states live in the small space while all weights still train — the
> GaLore idea. Deliverable: a drop-in optimizer wrapper plus a peak-VRAM and
> quality comparison against LoRA and full fine-tuning on a ~1B model.

## Decisions made (user-confirmed)

| Question | Decision |
|---|---|
| Where benchmarks run | CUDA-first; user runs on a rented GPU. This Mac (M2 Pro, 16 GB, no CUDA) does correctness only |
# | GPU target | Deferred — "can i tell u later". Shipped 16/24/40/80 GB presets + auto-detect; OOM recorded as a result |
| Model | TinyLlama-1.1B (ungated, no HF token needed) |
| Quality metric | Held-out loss + perplexity at matched token budget |
| Optimizer scope | Generic wrapper around ANY torch optimizer + per-layer update hooks |
| Dataset | yahma/alpaca-cleaned instruction tuning |
| Rigor | Per-method LR sweep on short runs, then one full run per method at its best LR |

## Key design insights from the session

- **Zeroed-surrogate trick**: because `project_back` is linear and step size is
  scalar, running any optimizer on a zeroed low-rank surrogate and projecting
  its delta back is *exactly* the reference GaLore update — proven to ~1e-6 in
  `tests/test_equivalence.py`. This is what makes "no forked optimizers" work.
- **Layerwise + grad accumulation are compatible** (reference implementation
  says they aren't): accumulate in the low-rank space; linearity makes it
# improvement
  exact. One caveat documented in `layerwise.py`: on SVD-due steps the subspace
  is chosen from the first micro-batch.
- **Naive GaLore loses to LoRA r=128 on memory** (10.02 vs 5.60 GiB predicted).
  The wins come from (a) layerwise gradient freeing and (b) projecting
  embed/lm_head, whose ~1 GiB of Adam state otherwise dominates. Final
  predicted: 5.28 GiB at 100% trainable.
- **Value-dependent optimizers** (LAMB trust ratios, Adafactor scale_parameter)
  are incompatible with the surrogate trick — detected empirically at
  construction, raises `ValueDependentOptimizerError`.
- torch's per-param scalar `step` tensor is counted as "bookkeeping", separate
  from optimizer state, in `memory_breakdown()`.
# improvement

## What was verified vs what is pending

Verified locally (78 tests, CPU, no network): reference equivalence,
# improvement
layerwise identity under accumulation, memory accounting byte-exact,
bit-identical checkpoint resume, full-rank thesis test, convergence smoke,
all three methods end-to-end in `--smoke` mode.

**Pending (needs the user's rented CUDA GPU):** the measured table in
README.md — memory bench (~30 min), LR sweep + three full runs (~6–10 GPU-hours
on 24 GB). Procedure in RUNBOOK.md. The GPU model was deliberately left open;
presets auto-detect.

## Environment facts that shaped the build

# improvement
# - Mac M2 Pro, 16 GB unified memory, no CUDA, system Python 3.9 → project uses
  its own uv-managed `.venv` (Python 3.11, torch 2.13 CPU/MPS, transformers
  5.14, peft 0.20).
- MPS can't run `linalg_svd` (falls back to CPU with a warning) — harmless for
  smoke runs, native on CUDA.
- No HF token on this machine — a reason TinyLlama (ungated) was chosen over
  Llama-3.2-1B (gated).

# Refined

# Enhanced

# Enhanced

# Optimized

# improvement
# Enhanced

# Optimized

# Refined

# Enhanced

# Optimized

# Enhanced
