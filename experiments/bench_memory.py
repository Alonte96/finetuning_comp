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
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from train import main as train_main

# The six rows of the README table.
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--smoke", type=lambda v: v.lower() in ("1", "true", "yes"), default=False)
    ap.add_argument("--preset", type=str, default=None)
    ns = ap.parse_args()

    for name, extra in CONFIGS:
        args = list(extra)
        if "--method" not in args:
            args = ["--method", "full"] + args
        args += [
            "--max-steps", str(ns.steps),
            "--eval-every", str(10**9),          # no eval: this is a memory run
            "--log-every", str(max(1, ns.steps // 3)),
            "--run-name", f"mem_{name}",
        ]
        if ns.smoke:
            args += ["--smoke", "true"]
        if ns.preset:
            args += ["--preset", ns.preset]
        print(f"\n=== bench_memory: {name} ===")
        record = train_main(args)
        peak = record["memory"]["peak_gib"]
        print(f"=== {name}: {record['status']}, peak {peak:.2f} GiB ===")


if __name__ == "__main__":
    main()
