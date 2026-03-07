# gradproj — memory-matched full fine-tuning via gradient projection

Low-rank gradient projection — keep every weight trainable at full rank, but
project the **gradient** into a low-rank subspace so the optimizer's state lives
in the small space — is **not this project's idea**. It is
[GaLore](https://arxiv.org/abs/2403.03507) (Zhao et al.). What is here is a
different *implementation* of it, and an honest measurement of it.

**Contribution 1 — projection for any optimizer, with no forks.**
The reference implementation bolts projection *inside* the optimizer, and
therefore ships hand-modified copies of AdamW, Adafactor and 8-bit Adam: every
new optimizer needs a new fork. That limitation is not confined to the reference
repo — **HuggingFace Transformers inherits it verbatim.** Its GaLore support is
a dispatch table over those same three forks
([`trainer_optimizer.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/trainer_optimizer.py)):

```python
from galore_torch import GaLoreAdafactor, GaLoreAdamW, GaLoreAdamW8bit
optimizer_mapping = {GALORE_ADAMW: GaLoreAdamW, GALORE_ADAMW_8BIT: GaLoreAdamW8bit, ...}
```

So `optim="galore_adamw"` gives you exactly three optimizers, and a fourth
(SGD, RMSprop, Lion, NAdam, …) requires someone to write and upstream a new
fork. This wrapper needs none. It hands the inner
optimizer a **zeroed low-rank surrogate parameter**, so after the inner step
that surrogate *is* the low-rank delta, which is projected back onto the real
weight. The inner optimizer never learns projection exists, and because
`project_back` is linear this is *exactly* the reference update, not an
approximation — asserted to ~1e-6 against a port of the reference optimizer in
`tests/test_equivalence.py`, and run end-to-end through seven stock optimizers
in `experiments/bench_optimizers.py`.

**Contribution 2 — layerwise updates *and* gradient accumulation, together.**
The reference treats these as mutually exclusive: its per-layer hook applies a
weight update per micro-batch, so accumulation would apply k updates instead of
one. HuggingFace inherits this too, as a hard error
([`trainer_optimizer.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/trainer_optimizer.py)):

```python
if is_layerwise:
    if args.gradient_accumulation_steps != 1:
        raise ValueError(f"Layerwise {optimizer_name} does not support gradient accumulation!")
```

That is the single most-used training loop in open-source ML refusing the exact
configuration measured below. The restriction is unnecessary. Projection is linear and the subspace is
fixed inside an accumulation window, so accumulating **in the low-rank space** is
exact:

    P^T (g₁ + g₂ + … + gₖ)  ==  P^T g₁ + P^T g₂ + … + P^T gₖ

You get per-layer gradient freeing (the thing that makes projection actually pay
at peak memory) *and* a real effective batch size, with the accumulator at `r/n`
the size of a full gradient. Numeric identity is proven in
`tests/test_layerwise.py` and measured at model scale in
`experiments/bench_accum.py`.

Supporting both: **a peak-VRAM + quality benchmark** (`experiments/`) against
LoRA and full fine-tuning on TinyLlama-1.1B / alpaca-cleaned, with a fairness
protocol enforced at runtime — which, as the measured table below records, does
not flatter the method.

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

## Measured: the two contributions

Both on TinyLlama-1.1B, RTX 3090, exact CUDA peaks. Reproduce with
`experiments/bench_optimizers.py` and `experiments/bench_accum.py`.

### 1. One wrapper, seven stock optimizers, zero forks

Every optimizer below is unmodified `torch.optim`, driving the same 1.1B model
through the same wrapper, with all 154 eligible matrices projected at rank 128.

| Inner optimizer | Optimizer state | Projections | Loss (6 steps) |
|---|---|---|---|
| AdamW | 1539.7 MiB | 57.8 MiB | 1.3530 → 0.8386 |
| Adamax | 1539.7 MiB | 57.8 MiB | 1.3530 → 0.8390 |
| NAdam | 1539.7 MiB | 57.8 MiB | 1.3530 → 0.8377 |
| SGD + momentum | 769.9 MiB | 57.8 MiB | 1.3530 → 0.8392 |
| RMSprop | 769.9 MiB | 57.8 MiB | 1.3530 → 0.8311 |
| Adagrad | 769.9 MiB | 57.8 MiB | 1.3530 → 0.8380 |
| **8-bit AdamW** (bitsandbytes) | **391.8 MiB** | 57.8 MiB | 1.3530 → 0.8385 |

The two-moment optimizers hold **exactly 2.0×** the state of the one-moment
ones, and 8-bit AdamW holds **3.93×** less than fp32 AdamW (two moments at one
byte instead of four). That is the signature of the mechanism working: the inner
optimizer allocated its state at the *surrogate's* low-rank shape, and its
moment count and dtype are the only things that vary. Nothing here is a fork, a
subclass, or a patch — and 8-bit Adam is precisely one of the three optimizers
the reference implementation had to fork by hand.

> **A measurement bug this table caught.** The first run of it reported 8-bit
> AdamW at **7.0 MiB** — a 220× win over AdamW. That was false.
> `memory_breakdown()` skipped tensors that were not floating point, and
> bitsandbytes stores its moments in `uint8`, so the state holding the numbers
> was invisible to the accounting and only the float quantisation statistics
> were counted. The fix (count every dtype) is in `wrapper.py`, pinned by
> `tests/test_contribution_benches.py::test_memory_breakdown_counts_non_float_state`.
> The honest 3.93× is what theory predicts; the 220× was the instrument
> measuring itself.

### 2. Layerwise updates under gradient accumulation

Effective batch fixed at 32; only the split changes. Each split runs twice —
layerwise off, then on.

| micro × accum | Peak (baseline) | Peak (layerwise) | Saved | max &#124;Δloss&#124; |
|---|---|---|---|---|
| 1 × 32 | 12.23 GiB | 8.90 GiB | 3.33 GiB | 1.1e-02 |
| 2 × 16 | 12.50 GiB | 9.16 GiB | 3.34 GiB | 6.8e-03 |
| 4 × 8 | 13.00 GiB | 9.66 GiB | 3.34 GiB | 2.6e-03 |
| 8 × 4 | 14.00 GiB | 10.67 GiB | 3.33 GiB | 1.6e-03 |

A near-constant **3.33 GiB** freed at every accumulation setting — the
combination the reference implementation refuses to run at all.

**On the residual Δloss, honestly:** it is not zero, and it should not be
reported as if it were. Within an accumulation window the identity is exact.
The deviation comes from the one documented caveat in `layerwise.py`: on a step
where the subspace is recomputed, the SVD sees the *first micro-batch's*
gradient rather than the accumulated one, because the accumulated full gradient
deliberately never exists. That predicts the deviation should shrink as the
first micro-batch grows — and it does, monotonically, from 1.1e-02 at
micro-batch 1 to 1.6e-03 at micro-batch 8. So the measurement quantifies the
caveat rather than contradicting the claim: same optimiser, same data, a
different-but-equally-valid choice of subspace.

Cross-check: the baseline peak at 2×16 (12.50 GiB) and the layerwise peak
(9.16 GiB) reproduce `mem_galore_r128` (12.50) and `mem_galore_r128_layerwise`
(9.16) from the independent memory bench.

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

These are *predictions*, and the measured table below shows where they are
wrong. What the prediction gets right is the internal ordering: naive GaLore
does not beat LoRA, and the wins come from the layerwise hooks and the embedding
flag — which is why those exist. What it gets wrong is the headline: it predicts
GaLore's best config undercutting LoRA r=128 (5.28 vs 5.60 GiB). Measured, it
does not. Static accounting omits the transient allocations projection creates,
and that omission is much larger for GaLore than for LoRA.

## Measured results

Measured on an **RTX 3090 (24 GB)**, TinyLlama-1.1B on alpaca-cleaned, fp32
masters + bf16 autocast, seq_len 512, effective batch 32, identical across
methods. Peak VRAM is an exact CUDA measurement (`max_memory_allocated`) over 30
optimizer steps; every config was isolated from the previous one, so no row
inherits its predecessor's allocation.

| Config | Peak VRAM | Predicted | Tokens/s | Trainable |
|---|---|---|---|---|
| Full fine-tune | 20.52 GiB | 16.39 | 2491 | 100% |
| LoRA r=16 | 4.98 GiB | 4.29 | 2221 | 1.1% |
| **LoRA r=128** | **6.50 GiB** | 5.60 | 2162 | 9.2% |
| GaLore r=128 | 12.50 GiB | 10.02 | 1647 | 100% |
| GaLore + layerwise | 9.16 GiB | 6.17 | 1551 | 100% |
| **GaLore + layerwise + emb.** | **7.82 GiB** | 5.28 | 1536 | 100% |

**The honest read: GaLore does not beat LoRA on peak memory here.** Its best
config costs 1.32 GiB *more* than LoRA r=128 (7.82 vs 6.50) and runs ~29% slower
(1536 vs 2162 tokens/s) — the SVD every `update_proj_gap` steps and the per-layer
hooks are not free, and hiding that would misrepresent the trade. The prediction
missed because it counts only static memory: the measured-vs-predicted gap is
+2.54 GiB for GaLore against +0.90 GiB for LoRA.

What GaLore *does* win, decisively, is against full fine-tuning: **7.82 vs 20.52
GiB, a 2.6× reduction while still training 100% of the weights** — versus LoRA's
9.2%.

### Quality: the full runs

1000 optimizer steps, each method at its own swept learning rate, same held-out
500 examples, identical batch/seq/seed, three identical RTX 3090s (one method
per GPU).

Run under **two seeds**, because the method differences turned out to be smaller
than the seed effect and a single-seed ranking would have been unpublishable.

| Method | LR | Eval loss (seed 42) | Eval loss (seed 43) | Peak VRAM | Tokens/s | Trainable |
|---|---|---|---|---|---|---|
| Full fine-tune | 5e-6 | **1.1251** | **1.0876** | 20.55 GiB | 2380 | 100% |
| LoRA r=128 | 3e-5 | 1.1277 | 1.0890 | **6.50 GiB** | 1802 | 8.4% |
| GaLore r=128 + layerwise | 3e-5 | 1.1336 | 1.0962 | 9.17 GiB | 2142 | 100% |

**Read this paired, not pooled.** Changing the seed moves every method by
~0.038 — four times the largest gap *between* methods — so comparing pooled
means across seeds would drown the signal in noise. But within a seed all three
methods see the identical data order, so the within-seed differences are paired
observations, and those replicate almost exactly:

| Paired difference | seed 42 | seed 43 |
|---|---|---|
| LoRA − full | +0.0026 | +0.0014 |
| GaLore − LoRA | +0.0059 | +0.0072 |
| GaLore − full | +0.0085 | +0.0086 |

**The thesis does not survive. LoRA dominates GaLore on both axes at once** —
lower held-out loss in both seeds *and* 29% less peak memory (6.50 vs 9.17 GiB).
No axis is left on which GaLore's full-rank updates buy something LoRA's
rank-limited ones do not. The ordering (full < LoRA < GaLore) is identical in
both seeds, and the GaLore−full gap reproduces to within 0.0001, so the ranking
is not a seed artefact even though the absolute numbers are.

Read that with these caveats, which cut in both directions:

- **Two seeds is two seeds.** Consistent sign across a paired replication is
  suggestive, not a significance test. The LoRA−full gap in particular
  (+0.0026, +0.0014) is small enough that more seeds could plausibly reorder
  those two; the GaLore−LoRA gap is 3–5× larger and more robust.
- **GaLore ran at 9.17 GiB, not its best 7.82 GiB config** — the quality run
  used layerwise without projected embeddings. Even at 7.82 it still loses to
  LoRA's 6.50.
- **Scale and budget.** TinyLlama-1.1B, 1000 steps, instruction tuning. GaLore's
  original claims centre on *pre-training* at larger scale, which this does not
  test and does not refute.
- **The sweep horizon flatters low learning rates.** Every winner sat at or near
  its grid's bottom edge; grids were extended once, which improved both
  baselines and flipped the 200-step ranking. All three methods got the
  identical procedure, so the comparison is fair, but it is not proof that these
  are each method's globally best settings.

The internal progression also confirms the design: 12.50 → 9.16 → 7.82 GiB as
layerwise hooks and embedding projection are switched on. Those two features are
the entire reason naive projection becomes competitive at all.

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
  bench_accum.py   contribution 2: layerwise x accumulation, loss identity + memory saved
  bench_optimizers.py  contribution 1: N stock optimizers through one wrapper, zero forks
  run_protocol.py  runs all four phases unattended; sweep winners feed the full runs
  report.py        results table + plots; enforces the fairness fingerprint per phase
tests/             157 tests, CPU-only, no network — see below
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

# Refined

# Optimized

# Optimized

# Enhanced

# Optimized

# Optimized
