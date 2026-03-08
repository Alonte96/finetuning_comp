# GaLore: memory-matched full fine-tuning via gradient projection

## Context

LoRA's biggest practical win is not quality — it's optimizer memory. It gets there by *restricting what can be learned* (ΔW is rank-limited by construction). GaLore takes the opposite trade: keep every weight trainable at full rank, but project the **gradient** into a low-rank subspace (recomputed periodically by SVD) so Adam's `m`/`v` states live in the small space. Memory win without the expressivity tax.

This project builds that as a **drop-in optimizer wrapper** and then defends the claim with a measured head-to-head against LoRA and full fine-tuning on TinyLlama-1.1B.

**Decisions locked with the user:**

| Question | Decision |
|---|---|
| Where benchmarks run | CUDA-first; user runs on a rented GPU. Local = correctness only (this Mac is M2 Pro / 16 GB / no CUDA) |
| GPU size | **Deferred** → ship 16/24/40/80 GB presets with `torch.cuda.get_device_properties` auto-detect, and OOM recorded as a result rather than a crash |
| Model | TinyLlama-1.1B (ungated, no HF token needed) |
| Quality metric | Held-out loss + perplexity at matched token budget |
| Optimizer scope | Generic wrapper around **any** `torch.optim` optimizer + per-layer update hooks |
| Data | `yahma/alpaca-cleaned` instruction tuning |
| Rigor | Per-method LR sweep on short runs, then one full run per method at its best LR |

## Verified facts this design rests on

- **Reference GaLore semantics** (from `jiaweizzhao/GaLore`): the optimizer projects `grad`, runs Adam in the low-rank space, calls `project_back(norm_grad)` (which applies `scale`), then `p.add_(norm_grad, alpha=-step_size)`, and applies weight decay **decoupled, to the full-rank parameter**: `p.add_(p, alpha=-lr*wd)`.
- `project_back` is **linear** and `step_size` is a scalar ⇒ projecting the inner optimizer's *parameter delta* back is mathematically identical to the reference's projecting the *normalized gradient* back. This is what makes the generic wrapper exact rather than an approximation.
- `Tensor.register_post_accumulate_grad_hook` (PyTorch ≥2.1) is leaf-only, runs under `no_grad`, and may modify/free `.grad` in place — exactly what per-layer updates need.
- TinyLlama-1.1B config: hidden 2048, intermediate 5632, 22 layers, 32 heads / 4 KV heads, vocab 32000, **untied** embeddings ⇒ 1.100B params = 969.1M attn+mlp (projectable 2D) + 131.1M embed/lm_head + 0.09M norms.

## Architecture

### 1. The wrapper (`src/gradproj/`) — the actual deliverable

**`projector.py` — `GaLoreProjector`**
Per-parameter SVD projection. `project(grad, step)` recomputes `P` via `torch.linalg.svd` (in fp32) only when `step % update_proj_gap == 0`, caching otherwise. Supports `proj_type` ∈ `std | reverse_std | left | right | full`. `std`: if `m ≥ n` use right-projection `G @ Vᵀ` → `(m, r)`, else left-projection `Uᵀ @ G` → `(r, n)`. Projection matrices stored in a configurable dtype (default bf16 — they cost ~0.26 GiB fp32 at 1.1B, which is not noise).
A `min_compression_ratio` guard skips projection where it wouldn't pay: e.g. TinyLlama's `k_proj`/`v_proj` are 256×2048, so r=128 gives a 128×2048 state — no saving once the projection matrix is counted.

**`wrapper.py` — `ProjectedOptimizer`** ← the differentiator
The reference implementation *forks* AdamW/Adafactor/8-bit Adam to bolt projection in. This wrapper works with any optimizer, unmodified, via a **zeroed low-rank surrogate parameter**:

```
for each projected p (m×n):
    p_low = zeros(r, n)            # created once; inner optimizer owns it → its states are low-rank
step():
    R          = projector.project(p.grad, step)
    p_low.grad = R
    p_low.data.zero_()             # so the returned data IS the delta
    inner.step()                   # any torch.optim optimizer; states allocated at p_low's shape
    p.data.add_(projector.project_back(p_low.data))   # scale applied inside project_back
    p.data.add_(p.data, alpha=-lr*wd)                 # decoupled WD on the FULL param (matches reference)
```
Weight decay is handled here and forced to 0 in the inner optimizer for projected params, because `p_low` is always zero and would otherwise silently no-op. Exposes `param_groups` (so LR schedulers work), `zero_grad`, and `state_dict`/`load_state_dict` including projection matrices for resume.
**Documented limitation, enforced in code:** optimizers whose update reads the parameter *value* (LAMB-style trust ratios, Adafactor with `relative_step=True`) are incompatible with the zeroed surrogate — detected and raised as a clear error, not silently wrong.

**`layerwise.py`** — per-layer weight updates via `register_post_accumulate_grad_hook`, so gradients never all co-exist. This is what makes GaLore actually beat LoRA on peak memory. It is **not compatible with gradient accumulation** (the hook fires per micro-batch); the code asserts `grad_accum_steps == 1` rather than quietly changing the optimization.

**`param_groups.py`** — split a model into projected / regular groups (target modules, `ndim == 2`, skip norms; embed & lm_head projectable via a flag).
**`memory.py`** — `MemoryProbe`: exact on CUDA (`reset_peak_memory_stats` / `max_memory_allocated` + `max_memory_reserved`), sampled-and-labelled-approximate on MPS/CPU; plus **analytic accounting** that walks optimizer state tensors and sums `numel × element_size`, so the optimizer-state claim is provable independent of allocator noise.
**`presets.py`** — 16/24/40/80 GB configs + auto-detect.

### 2. Benchmark suite (`experiments/`)

`train.py` — one training loop, `--method {full,lora,galore}`, so no method gets a different code path. `data.py` (alpaca-cleaned, prompt-token masking, held-out split), `evaluate.py` (held-out loss/PPL), `sweep_lr.py`, `bench_memory.py` (short runs, OOM caught and recorded), `report.py` (markdown table + matplotlib plots from `results/*.jsonl`; no wandb account needed).

**Fairness protocol**, asserted at runtime — identical seq len, micro-batch, grad-accum, dtype, checkpointing, seed, data order, and total optimizer steps across all three methods. Per-method LR sweep (full FT ~1e-5..5e-5, LoRA ~1e-4..1e-3, GaLore ~1e-5..1e-4) because not tuning LoRA's LR is the classic way this comparison gets rigged. **Step time and tokens/s are reported alongside memory** — GaLore's SVD is a real cost and hiding it would make the result dishonest.

### 3. Predicted memory (analytic, fp32 master weights + bf16 autocast, AdamW)

To be **validated against measurement**, not published as fact. `report.py` prints predicted vs measured side by side.

| Method | Params | Grads | Opt states | Proj | Static total | Trainable |
|---|---|---|---|---|---|---|
| Full FT | 4.10 | 4.10 | 8.20 | — | **16.4 GiB** | 100% |
| LoRA r=16 | 4.10 | 0.05 | 0.09 | — | **4.29 GiB** | 1.1% |
| LoRA r=128 | 4.10 | 0.38 | 0.75 | — | **5.60 GiB** | 9.2% |
| GaLore r=128 | 4.10 | 4.10 | 1.50 | 0.26 | **9.96 GiB** | 100% |
| GaLore r=128 + layerwise | 4.10 | 0.24 | 1.50 | 0.26 | **6.11 GiB** | 100% |
| GaLore r=128 + layerwise + proj embed/lm_head | 4.10 | 0.24 | 0.59 | 0.27 | **5.20 GiB** | 100% |

# The headline the project is built to earn: **the last row undercuts LoRA r=128 on memory while training 100% of the weights.** Note optimizer states for the *non-projected* embed+lm_head (0.98 GiB) dominate the naive GaLore config — which is why projecting them is worth a flag.

## Verification

**Runs locally on this Mac, CPU-only, no GPU and no network** (tests build a `LlamaConfig(hidden=64, layers=2, vocab=256)` model and synthetic token data — nothing is downloaded):

1. `test_projector.py` — shapes/rank for every `proj_type` incl. non-square and `m<n`; `P` changes exactly at multiples of `update_proj_gap`.
2. `test_equivalence.py` — **`ProjectedOptimizer(torch.optim.AdamW)` matches a from-scratch port of the reference `GaLoreAdamW` to ~1e-6 over 50 steps.** The core correctness proof.
3. `test_memory_accounting.py` — measured optimizer-state bytes equal the analytic formula.
4. `test_fullrank.py` — **the thesis test**: cumulative `ΔW` has rank > r after ≥3 subspace switches, while LoRA's `ΔW` is exactly rank r.
5. `test_layerwise.py` — layerwise result is *numerically identical* to standard mode at accum=1, and holds fewer live gradients.
6. `test_convergence.py` — tiny model on a synthetic copy task: GaLore reaches full-FT-comparable loss.
7. `test_wrapper_api.py` — state_dict round-trip resumes bit-identically; LR scheduler drives it; generality on SGD-momentum; clear raise on LAMB/Adafactor-relative-step.

**Requires the GPU you'll provide** — `RUNBOOK.md` gives exact commands; README ships with the measured table **empty and clearly marked**, filled by `report.py`. I will not put invented numbers in it.

Environment: `uv`-managed `.venv` (Python 3.11) inside `github_project/` — system Python here is 3.9.6 with no torch. Installs torch CPU + transformers + peft + datasets + pytest locally (~1-2 GB download); the CUDA wheel is installed on the GPU box, not here.

## Build order

0. `mkdir github_project`, `git init`, commit the design doc to `docs/superpowers/specs/`, set up `.venv`.
1. `projector.py` + tests 1 — TDD.
2. `wrapper.py` + reference port + test 2 (equivalence). Nothing else is trustworthy until this passes.
3. `param_groups.py`, `memory.py`, `presets.py` + test 3.
4. `layerwise.py` + test 5.
5. Tests 4, 6, 7 (full-rank claim, convergence, API).
6. `experiments/` suite + `report.py`; end-to-end smoke on CPU with the tiny model for all three methods.
7. README with analytic predictions + `RUNBOOK.md`; measured table left blank pending your GPU run.

## Open item

Tell me the GPU when you have it and I'll pin the preset defaults; until then auto-detect handles it and a full-FT OOM is recorded as a result rather than a failure.

# Enhanced

# Enhanced

# Optimized

# Enhanced

# Enhanced

# Optimized

# Enhanced

# Refined
