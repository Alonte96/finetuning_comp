# RUNBOOK — producing the measured table on a CUDA GPU

Everything below runs on a rented GPU box (RunPod / Lambda / vast.ai / Colab
terminal). Nothing here needs an HF token: TinyLlama and alpaca-cleaned are
ungated. Total budget for the full protocol: **roughly 6–10 GPU-hours on a
24 GB card**; the memory table alone is ~30 minutes.

## 0. Setup (once per box)

```bash
git clone <this-repo> && cd github_project     # or rsync the project across:
# rsync -az --exclude .venv --exclude .git --exclude '.runpod*' \
#   -e 'ssh -p <port>' ./ root@<host>:/workspace/github_project/
pip install uv
uv venv .venv && source .venv/bin/activate
uv pip install -e '.[experiments,dev]'          # pulls the CUDA torch wheel
python -m pytest -q                             # ~1 min, must be all green
python -m gradproj.memory                       # predicted table, sanity check
```

If `python -c "import torch; print(torch.cuda.is_available())"` is not `True`,
stop and fix the torch install first.

## Everything at once (recommended for a rented box)

```bash
python experiments/run_protocol.py 2>&1 | tee results/protocol.log
```

Runs all four phases below in order and feeds the sweep's winning LRs straight
into the full runs without anyone retyping them. Steps 1-4 are what it does;
run them by hand if you want to stop and inspect between phases.

## 1. The memory table (~30 min)

```bash
python experiments/bench_memory.py              # auto-detects GPU tier
# or pin the preset if auto-detect picks wrong: --preset 24gb
```

Runs all six configs for 30 optimizer steps each and records exact CUDA peaks.
These runs do no evaluation at all (`--eval-every` is set above `--max-steps`),
so eval activations never enter a memory measurement. Each config is isolated
from the next: the layerwise hooks are detached and the allocator cache emptied
between runs, so every row measures only itself.

**If full FT OOMs on a small card, that is a recorded result, not a failure** —
the run is written with `status: oom` and appears in the table as such. If one
config crashes outright, the remaining configs still run.

## 2. The LR sweep (~2-3 h on 24 GB)

```bash
python experiments/sweep_lr.py --steps 200
```

Nine short runs (3 methods × 3 LRs), winners written to
`results/sweep_winners.jsonl` and printed. Expected neighbourhoods: full ~2e-5,
LoRA ~3e-4, GaLore ~3e-5. Diverged (NaN) candidates are excluded rather than
allowed to win on an IEEE comparison accident. If a winner lands on a grid edge
the sweep says so explicitly — extend the grid one step in that direction and
re-run that method:

```bash
python experiments/sweep_lr.py --methods galore --steps 200   # after editing DEFAULT_LR_GRIDS
```

Any flag `sweep_lr.py` does not recognise is forwarded verbatim to `train.py`,
so the sweep can run under exactly the settings the full runs will use — a
winner found under different settings is not a winner. Typos fail immediately
rather than 40 minutes in.

## 3. Full runs at the winning LRs (~4-6 h on 24 GB)
# improvement

```bash
python experiments/train.py --method full   --lr <winner> --run-name final_full
python experiments/train.py --method lora   --lr <winner> --run-name final_lora
python experiments/train.py --method galore --lr <winner> --run-name final_galore
```

1000 optimizer steps × effective batch 32 each, eval every 100 steps on the
same held-out 500 examples. The three runs MUST use identical `--seq-len /
--micro-batch / --grad-accum` (defaults already are identical; the report
enforces this and will refuse mixed settings).

Every run records which `--phase` it belongs to (`mem` / `sweep` / `final`).
The report checks fairness *within* each phase and, across phases, on
everything except step count — so the three phases legitimately differing in
`--max-steps` is not reported as unfairness, while a real mismatch (a different
seq_len or dtype anywhere) still is.

## 4. The report

```bash
python experiments/report.py
cat results/REPORT.md
```

Produces the measured table (peak GiB, predicted GiB, eval loss/ppl, tokens/s,
status), training/eval curves, and the measured-vs-predicted memory bar chart.
Copy the table into README.md's "Measured results" section.

## 5. The contribution benches (~30 min, optional but recommended)

The four phases above measure GaLore *as a method*. These two measure what this
implementation adds on top of the reference, and are the evidence behind the
README's two contribution claims:

```bash
python experiments/bench_accum.py --steps 20        # layerwise x accumulation
python experiments/bench_optimizers.py --steps 6    # N stock optimizers, one wrapper
```

`bench_accum.py` holds the effective batch fixed at 32 and runs every
(micro_batch, grad_accum) split twice — layerwise off, then on — reporting both
loss trajectories and both peak memories. The claim it defends is a *conjunction*:
identical optimisation AND lower peak memory, at accumulation settings the
reference implementation refuses to run. A memory saving that moved the loss
curve would refute it, so read the `max |dloss|` column before the `saved`
column.

`bench_optimizers.py` runs the same model through the wrapper with AdamW, SGD,
RMSprop, Adagrad, Adamax, NAdam (plus Lion and bitsandbytes 8-bit AdamW where
available), recording live-tensor optimizer state per optimizer. Value-dependent
optimizers are expected to appear as a clean `refused`, not as a crash.

## Knobs that matter

All of these are accepted by `train.py` and `bench_memory.py`, and forwarded
through `sweep_lr.py`.

| Flag | Default | Notes |
|---|---|---|
| `--preset` | auto | `auto` / `16gb` / `24gb` / `40gb` / `80gb`; sets seq_len, batch, checkpointing. `auto` reads the card's total VRAM and picks the largest tier that fits; without CUDA it leaves the built-in defaults alone. Any of those four values you pass explicitly wins over the preset |
| `--galore-rank` | 128 | the paper's fine-tuning default (r=4–8 also works for pure FT) |
| `--galore-update-proj-gap` | 200 | SVD cadence; halve it if loss plateaus after subspace switches |
| `--galore-scale` | 0.25 | the paper's fine-tuning α |
| `--galore-layerwise` | true | per-layer grad freeing; exact with accumulation |
| `--galore-project-embeddings` | false | flips the last ~1 GiB of optimizer state; try both |
| `--dtype` | bf16 | autocast compute dtype; fp32 masters throughout. `fp16` is refused: it needs loss scaling, and a `GradScaler` cannot unscale gradients that layerwise projection already freed during backward. On a pre-Ampere card (no bf16) the run stops and tells you to use `fp32` for **all three** methods |

## Honesty checklist before publishing numbers

- [ ] `results/REPORT.md` shows **Fairness check passed** (no mixed settings)
      and reports no missing phase.
- [ ] Peak memory rows are marked `exact: yes` (CUDA); MPS/CPU approximations
      must not be presented as VRAM.
- [ ] Tokens/s column included — GaLore pays an SVD cost every
      `update_proj_gap` steps and per-layer hooks add overhead; hiding speed
      would misrepresent the trade.
- [ ] Each method reported at its *own* best LR from the sweep.
- [ ] OOM rows (if any) reported as OOM, not silently dropped.

# Enhanced
