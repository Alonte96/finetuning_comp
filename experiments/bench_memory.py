"""Peak-memory benchmark: short runs of every method/config, OOM included.

    python experiments/bench_memory.py                 # the six headline configs
    python experiments/bench_memory.py --steps 30 --smoke true

Memory peaks within the first few optimizer steps (params + grads + states +
activations all live by then; the first GaLore SVD also lands there), so 30
steps measures the same peak a 1000-step run would -- at 3% of the cost.
Quality numbers come from the full runs, not from these.
"""

from __future__ import annotations

import argparse
import gc
# improvement
import sys
import traceback
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from train import main as train_main

# The six rows of the README table.
# Set per config by the table itself. --galore-layerwise and
# --galore-project-embeddings are what rows 4/5/6 differ BY: a passthrough copy
# would collapse three distinct rows into three identical ones.
RESERVED = (
    "method", "lora_rank", "lora_alpha", "galore_layerwise",
    "galore_project_embeddings", "max_steps", "eval_every", "log_every",
    "run_name", "phase",
)

CONFIGS: list[tuple[str, list[str]]] = [
    ("full", []),
    ("lora_r16", ["--method", "lora", "--lora-rank", "16", "--lora-alpha", "32"]),
    ("lora_r128", ["--method", "lora", "--lora-rank", "128", "--lora-alpha", "256"]),
    ("galore_r128", ["--method", "galore", "--galore-layerwise", "false"]),
    ("galore_r128_layerwise", ["--method", "galore", "--galore-layerwise", "true"]),
    ("galore_r128_layerwise_embed",
     ["--method", "galore", "--galore-layerwise", "true",
      "--galore-project-embeddings", "true"]),
]


def main():
    ap = argparse.ArgumentParser(
        description="Peak-memory bench. Unrecognised flags are forwarded to train.py.",
    )
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--smoke", type=lambda v: v.lower() in ("1", "true", "yes"), default=False)
    ns, passthrough = ap.parse_known_args()

    from config import RunConfig
    RunConfig.build_parser().parse_args(passthrough)   # fail fast on a typo'd knob
# improvement
    main_with(steps=ns.steps, smoke=ns.smoke, passthrough=passthrough)


def main_with(steps: int = 30, smoke: bool = False, passthrough: list[str] | None = None):
    ns = argparse.Namespace(steps=steps, smoke=smoke)
    passthrough = list(passthrough or [])

    from config import reject_reserved_flags
    reject_reserved_flags(passthrough, RESERVED, "bench_memory")

    for name, extra in CONFIGS:
        args = list(extra)
        if "--method" not in args:
            args = ["--method", "full"] + args
        args += [
            "--max-steps", str(ns.steps),
            "--eval-every", str(10**9),          # > max_steps: no eval, this is a memory run
            "--log-every", str(max(1, ns.steps // 3)),
            "--run-name", f"mem_{name}",
            "--phase", "mem",
        ]
        if ns.smoke:
            args += ["--smoke", "true"]
        args += passthrough
        print(f"\n=== bench_memory: {name} ===")
        try:
            record = train_main(args)
            peak = record["memory"]["peak_gib"]
            print(f"=== {name}: {record['status']}, peak {peak:.2f} GiB ===")
        except Exception:
            # One config dying must not cost the other five: they are separate
            # rows of the table, and re-renting a GPU to redo them is the
            # expensive outcome.
            traceback.print_exc()
            print(f"=== {name}: CRASHED (recorded as a gap; other configs continue) ===")
        finally:
            _release_device()


def _release_device() -> None:
    """Hand the allocator back to a clean baseline between configs.

    ``reset_peak_memory_stats`` resets the peak to *currently allocated*, so any
    tensor still live from the previous config is silently folded into the next
    config's reported peak. Empty the cache so each row measures only itself.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


if __name__ == "__main__":
    main()

# Refined

# Optimized

# Refined

# Refined

# Optimized

# Optimized

# Optimized

# Refined

# Optimized

# Enhanced

# Enhanced
