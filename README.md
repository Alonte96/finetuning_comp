# gradproj — memory-matched full fine-tuning via gradient projection

LoRA's biggest real-world win is not quality — it's **optimizer memory**, and it
buys that memory by restricting what the model can learn (its total weight
change is rank-limited by construction, forever). This project competes on the
same axis without the restriction: project the **gradient** into a low-rank
subspace (recomputed periodically via SVD) so Adam's moments live in the small
space, while **every weight still trains at full rank** — the
[GaLore](https://arxiv.org/abs/2403.03507) idea.

Two deliverables:

1. **A drop-in optimizer wrapper** (`src/gradproj/`) that adds gradient
   projection to *any* `torch.optim` optimizer — unmodified. The reference
   implementation ships hand-forked copies of AdamW/Adafactor/8-bit Adam; this
   wrapper needs none of that, and is proven equivalent to the reference
   optimizer to ~1e-6 (`tests/test_equivalence.py`).
2. **A peak-VRAM + quality benchmark** (`experiments/`) against LoRA and full
   fine-tuning on TinyLlama-1.1B / alpaca-cleaned, with a fairness protocol
   enforced at runtime.

```python
import torch
from gradproj import ProjectedOptimizer, LayerwiseProjection, galore_param_groups

optimizer = ProjectedOptimizer(
    galore_param_groups(model),      # attention + MLP weights projected, rest normal
    torch.optim.AdamW,               # or SGD, RMSprop, Lion, 8-bit Adam, ...
    rank=128, update_proj_gap=200, scale=0.25, lr=1e-5,
)
layerwise = LayerwiseProjection(optimizer).attach()   # optional: free grads per-layer

for batch in data:                   # ordinary training loop, schedulers work
    model(**batch).loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

## How it works

An optimizer step maps `(state, grad) → delta`. The wrapper hands the inner
optimizer a **zeroed low-rank surrogate parameter** whose gradient is the
projected gradient `P^T G`. After `inner.step()` the surrogate *is* the low-rank
delta (it started at zero), which is projected back and added to the real
weight. The inner optimizer allocates its state at the surrogate's shape —
that's the entire memory win — and never learns projection exists. Since
`project_back` is linear, this is *exactly* the reference GaLore update, not an
approximation.

The one requirement: the inner optimizer's update must not depend on the
parameter's **value** (true for Adam/SGD/RMSprop/Lion; false for LAMB trust
ratios and Adafactor's `scale_parameter`). The wrapper verifies this empirically
at construction and raises `ValueDependentOptimizerError` rather than training
silently wrong.

### Beyond the reference implementation

- **Any optimizer, no forks** — the reference ships modified optimizer copies;
  this wraps stock ones and matches the reference to ~1e-6.
- **Layerwise + gradient accumulation together** — the reference treats
  per-layer updates as incompatible with accumulation. Projection is linear, so
  accumulating *in the low-rank space* is exact:
  `P^T(g₁+…+gₖ) = P^T g₁+…+P^T gₖ`. You get per-layer gradient freeing *and* a
  real effective batch size (`tests/test_layerwise.py` proves numeric identity).
- **Memory honesty end-to-end** — analytic predictions from parameter shapes,
  live-tensor accounting in the optimizer, exact CUDA peaks, and a report that
  prints predicted vs measured side by side and refuses to compare runs with
  mismatched settings.
- **Compression-ratio guard** — parameters where projection wouldn't pay
  (e.g. TinyLlama's GQA `k_proj`/`v_proj` at 256×2048 vs rank 128) are trained
  normally instead of carrying a useless projection.

## Predicted memory, TinyLlama-1.1B (fp32 masters + bf16 autocast, AdamW)

Static training memory (params + grads + optimizer state + projections),
computed analytically from parameter shapes — activations excluded. Reproduce
with `python -m gradproj.memory` or see `tests/test_memory_accounting.py`.

| Method | Params | Grads | Opt. state | Proj. | **Total** | Trainable |
|---|---|---|---|---|---|---|
| Full fine-tune | 4.10 | 4.10 | 8.20 | — | **16.39 GiB** | 100% |
| LoRA r=16 | 4.14 | 0.05 | 0.09 | — | **4.29 GiB** | 1.1% |
| LoRA r=128 | 4.47 | 0.38 | 0.75 | — | **5.60 GiB** | 9.2% |
| GaLore r=128 | 4.10 | 4.10 | 1.50 | 0.32 | **10.02 GiB** | 100% |
| + layerwise | 4.10 | 0.24 | 1.50 | 0.32 | **6.17 GiB** | 100% |
| + layerwise + proj. embeddings | 4.10 | 0.24 | 0.59 | 0.35 | **5.28 GiB** | 100% |

The row that matters: **GaLore with layerwise updates and projected embeddings
undercuts LoRA r=128 while training 100% of the weights.** Also visible: naive
GaLore does *not* beat LoRA — the 4.10 GiB of co-existing gradients and the
~1 GiB of optimizer state on unprojected embeddings dominate. The wins come
from the layerwise hooks and the embedding flag, which is why they exist.

## Measured results

> **This table is intentionally empty.** It requires a CUDA GPU, and inventing
> numbers would defeat the purpose. Run the benchmark (see
> [RUNBOOK.md](RUNBOOK.md)) and `experiments/report.py` fills it in at
> `results/REPORT.md`, including measured-vs-predicted deltas, tokens/sec (the
> SVD is not free, and we report it), and eval loss/perplexity per method at its
> best learning rate.

| Method | Peak VRAM (GiB) | Eval loss | Eval PPL | Tokens/s |
|---|---|---|---|---|
| Full fine-tune | *pending* | *pending* | *pending* | *pending* |
| LoRA r=128 | *pending* | *pending* | *pending* | *pending* |
| GaLore r=128 (+layerwise) | *pending* | *pending* | *pending* | *pending* |

## Repository layout

```
src/gradproj/
  projector.py     GaLoreProjector: periodic-SVD projection (reference semantics)
  wrapper.py       ProjectedOptimizer: the generic wrapper (zeroed-surrogate trick)
  layerwise.py     per-layer grad projection hooks; exact under grad accumulation
  param_groups.py  model → projected/regular param groups
  memory.py        analytic prediction + honest device probes (CUDA exact, MPS/CPU labelled approx)
  presets.py       16/24/40/80 GB benchmark presets + CUDA auto-detect
experiments/
  train.py         ONE training loop for all three methods (only the optimizer differs)
  data.py          alpaca-cleaned with prompt masking; synthetic offline smoke data
  sweep_lr.py      per-method LR sweep (not sweeping LoRA's LR is how comparisons get rigged)
  bench_memory.py  short-run peak-memory measurement; OOM recorded as a result
  report.py        results table + plots; enforces the fairness fingerprint
tests/             79 tests, CPU-only, no network — see below
```

## Tests

```bash
uv venv .venv && uv pip install -e '.[dev]'   # torch CPU is enough
.venv/bin/python -m pytest
```

No GPU, no downloads. The suite proves: reference equivalence to ~1e-6 across
subspace switches (incl. weight decay placement), layerwise ≡ standard mode
under gradient accumulation, optimizer-state bytes == analytic formula, resume
from checkpoint is bit-identical, and the thesis itself — after subspace
switches GaLore's accumulated ΔW exceeds rank r while a LoRA-style ΔW driven by
the same gradient stream stays rank-limited forever
(`tests/test_fullrank.py`).

## Relation to prior work

This is an independent implementation of the method from *GaLore: Memory-
Efficient LLM Training by Gradient Low-Rank Projection* (Zhao et al., 2024),
built for drop-in generality (any optimizer, no forks), layerwise+accumulation
support, and a memory-accounting story you can audit. Projection semantics
follow the reference implementation so numbers stay comparable.
