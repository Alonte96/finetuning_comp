# RUNBOOK — producing the measured table on a CUDA GPU

Everything below runs on a rented GPU box (RunPod / Lambda / vast.ai / Colab
terminal). Nothing here needs an HF token: TinyLlama and alpaca-cleaned are
ungated. Total budget for the full protocol: **roughly 6–10 GPU-hours on a
24 GB card**; the memory table alone is ~30 minutes.

## 0. Setup (once per box)

```bash
git clone <this-repo> && cd github_project
pip install uv
uv venv .venv && source .venv/bin/activate
uv pip install -e '.[experiments,dev]'          # pulls the CUDA torch wheel
python -m pytest -q                             # ~1 min, must be all green
python -m gradproj.memory                       # predicted table, sanity check
```

If `python -c "import torch; print(torch.cuda.is_available())"` is not `True`,
stop and fix the torch install first.

## 1. The memory table (~30 min)

```bash
python experiments/bench_memory.py              # auto-detects GPU tier
# or pin the preset if auto-detect picks wrong: --preset 24gb
```

Runs all six configs for 30 optimizer steps each and records exact CUDA peaks.
**If full FT OOMs on a small card, that is a recorded result, not a failure** —
the run is written with `status: oom` and appears in the table as such.

## 2. The LR sweep (~2-3 h on 24 GB)

```bash
python experiments/sweep_lr.py --steps 200
```

Nine short runs (3 methods × 3 LRs), winners written to
`results/sweep_winners.jsonl` and printed. Expected neighbourhoods: full ~2e-5,
LoRA ~3e-4, GaLore ~3e-5. If a winner lands on a grid edge, extend the grid one
step in that direction and re-run that method.

## 3. Full runs at the winning LRs (~4-6 h on 24 GB)

```bash
python experiments/train.py --method full   --lr <winner> --run-name final_full
python experiments/train.py --method lora   --lr <winner> --run-name final_lora
python experiments/train.py --method galore --lr <winner> --run-name final_galore
```

1000 optimizer steps × effective batch 32 each, eval every 100 steps on the
same held-out 500 examples. The three runs MUST use identical `--seq-len /
--micro-batch / --grad-accum` (defaults already are identical; the report
enforces this and will refuse mixed settings).

## 4. The report

```bash
python experiments/report.py
cat results/REPORT.md
```

Produces the measured table (peak GiB, predicted GiB, eval loss/ppl, tokens/s,
status), training/eval curves, and the measured-vs-predicted memory bar chart.
Copy the table into README.md's "Measured results" section.

## Knobs that matter

| Flag | Default | Notes |
|---|---|---|
| `--preset` | auto | `16gb` / `24gb` / `40gb` / `80gb`; sets seq_len, batch, checkpointing |
| `--galore-rank` | 128 | the paper's fine-tuning default (r=4–8 also works for pure FT) |
| `--galore-update-proj-gap` | 200 | SVD cadence; halve it if loss plateaus after subspace switches |
| `--galore-scale` | 0.25 | the paper's fine-tuning α |
| `--galore-layerwise` | true | per-layer grad freeing; exact with accumulation |
| `--galore-project-embeddings` | false | flips the last ~1 GiB of optimizer state; try both |
| `--dtype` | bf16 | autocast compute dtype; fp32 masters throughout |

## Honesty checklist before publishing numbers

- [ ] `results/REPORT.md` shows **Fairness check passed** (no mixed settings).
- [ ] Peak memory rows are marked `exact: yes` (CUDA); MPS/CPU approximations
      must not be presented as VRAM.
- [ ] Tokens/s column included — GaLore pays an SVD cost every
      `update_proj_gap` steps and per-layer hooks add overhead; hiding speed
      would misrepresent the trade.
- [ ] Each method reported at its *own* best LR from the sweep.
- [ ] OOM rows (if any) reported as OOM, not silently dropped.
