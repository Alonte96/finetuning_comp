# Session notes — 2026-07-29 (the measured run)

The session that took this repo from "correctness verified locally, numbers
pending" to a measured result. Companion to
`2026-07-28-session-notes.md`, which recorded the build.

## What the numbers said

The project's headline claim did **not** survive measurement, and the write-up
in README.md now says so. Two separate reversals:

# improvement
1. **Memory.** The analytic prediction had GaLore's best config undercutting
   LoRA r=128 (5.28 vs 5.60 GiB). Measured: **7.82 vs 6.50 GiB** — GaLore loses.
   The prediction models static memory only, and the measured-vs-predicted gap
   is +2.54 GiB for GaLore against +0.90 for LoRA, because projection creates
   transient allocations that parameter shapes cannot reveal.
2. **Quality.** At 1000 steps, each method at its own swept LR: full 1.1251,
   LoRA 1.1277, GaLore 1.1336. **LoRA dominates GaLore on both axes at once** —
   better loss *and* 29% less peak memory. There is no trade left to argue for.

What survived, and is now the project's actual contribution, is the pair of
implementation claims — both measured at model scale this session:

* one wrapper drives **seven stock optimizers** unmodified (state 1539.7 MiB for
  two-moment, 769.9 for one-moment, 391.8 for 8-bit — exactly the ratios the
  mechanism predicts);
* layerwise projection **under gradient accumulation** frees a near-constant
  3.33 GiB at every batch split, a combination the reference forbids.

## Pre-flight paid for itself

An adversarial review of the pipeline before spending GPU money found 28
confirmed defects; six were fixed, each pinned by a regression test that was
verified non-vacuous by reverting the fix and watching the test fail. Two would
have silently corrupted the published table:

# * **Layerwise hooks were never detached.** The hook lives in C++ autograd state
  that Python's cyclic GC cannot traverse, so every GaLore run pinned ~5.9 GiB
  for the life of the process. In `bench_memory`'s single process that meant
  config 6's peak included config 5's model. The measured table now shows
  7.82 GiB *below* the previous row's 9.16 — proof the isolation works.
* **The fairness check was guaranteed to fail.** It compared every run against
  the first record on all fields including `max_steps`, while the RUNBOOK
  appends 30-, 200- and 1000-step runs to one file. Following the documented
  procedure always printed "Fairness check FAILED".

## Three bugs the *experiments themselves* caught

Worth recording, because each was found by running the thing rather than
reasoning about it:

1. **A false 220× win.** 8-bit AdamW first reported 7.0 MiB against AdamW's
   1539.7. `memory_breakdown()` skipped non-floating-point tensors, and
   bitsandbytes keeps its moments in `uint8` — the accounting was blind to the
   very state it was measuring. Corrected to 391.8 MiB (3.93×, which is what
   two moments at one byte instead of four should give).
2. **A phase-labelling error of mine.** `bench_accum` deliberately sweeps
   micro-batch and grad-accum, but I filed it under `phase=mem`, which requires
   them constant — so the fairness checker failed the whole report. The checker
   was right; the label was wrong. `accum` is now its own phase with those two
   fields exempt.
3. **An over-permissive fix, caught by an existing test.** Making seed exempt so
   replications would pass also let a *single* method be re-rolled under a new
   seed — the oldest way to manufacture a favourable number. Replaced with:
   seed may vary only as a *complete* replication covering every method.

## Method decisions worth remembering

* **Grid-edge extension changed the answer.** Both baselines' sweep winners sat
  on the bottom edge of their grids. Extending downward improved full FT
  (1.1549 → 1.1454 at 5e-6) and LoRA (1.1496 → 1.1484 at 3e-5) while GaLore's
  winner was unchanged and interior. That **flipped the 200-step ranking, moving
  GaLore from first to last.** Skipping it would have left both baselines
  under-tuned in GaLore's favour.
* **Known limitation:** every extension moved the winner down again, so the
  200-step sweep horizon systematically favours low LRs. All methods get the
  identical procedure, so the comparison is fair, but these are not proven to be
  each method's globally best settings.
* **Parallelisation without compromising fairness.** Three *identical* RTX 3090s,
  one method per GPU. Same GPU model matters: eval loss is machine-independent
  but peak memory and tokens/s are not. Fingerprints were verified to match
  across machines before trusting the results. Throughput for the headline
  comparison comes from the sequential memory bench, not from runs sharing a
  box.
* Each pod wrote its own `results/runs.jsonl`; concurrent appends from three
  machines to one file would interleave records. Merged afterwards.

## Environment notes

* RunPod community cloud. RTX 4090 and A5000 had no capacity; 3× RTX 3090 at
  $0.22/hr each. Total spend for the whole benchmark: under $3.
* The `runpod/pytorch:...-cu1290` image fails on a GeForce host with driver
  550.67 (`error 804: forward compatibility was attempted on non supported HW`).
  Fix: `mv /usr/local/cuda-12.9/compat{,.disabled} && ldconfig`. The bundled
# improvement
  forward-compat libraries only work on datacentre cards.

# Enhanced

# Enhanced

# Enhanced
