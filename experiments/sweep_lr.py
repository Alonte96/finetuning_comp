"""Per-method learning-rate sweep on short runs, then print the winners.

    python experiments/sweep_lr.py                     # all methods, default grids
    python experiments/sweep_lr.py --methods galore --steps 200

Each candidate trains for --steps optimizer steps and is scored by final eval
loss. Winners feed the full runs (see RUNBOOK.md). LoRA's grid sits ~10x above
full FT's, because that is where LoRA actually works -- sweeping per method is
what keeps the comparison honest.
"""

from __future__ import annotations

import argparse
import math
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bench_memory import _release_device
from config import (
    DEFAULT_LR_GRIDS,
    RESULTS_DIR,
    RunConfig,
    append_jsonl,
    read_jsonl,
    reject_reserved_flags,
)
from train import main as train_main

# Set per candidate by the sweep itself; a passthrough copy would override it.
RESERVED = ("method", "lr", "max_steps", "run_name", "phase", "eval_every")
# 

def run_sweep(
    methods: list[str], steps: int, smoke: bool, passthrough: list[str] | None = None
) -> dict[str, float]:
    """Score each method's LR grid by final held-out loss and return the winners.

    ``passthrough`` carries the run-shaping knobs (--preset, --dtype, --galore-*)
    straight to train.py, so the sweep can be run under exactly the settings the
    full runs will use -- a winner found under different settings is not a winner.
    """
    passthrough = list(passthrough or [])
    # Checked HERE, not just in __main__: run_protocol.py calls this function
    # directly, and that is the unattended path where a silent mistake is
    # costliest.
    reject_reserved_flags(passthrough, RESERVED, "sweep_lr")

    best: dict[str, float] = {}
    for method in methods:
        results = []
        for lr in DEFAULT_LR_GRIDS[method]:
            args = [
                "--method", method, "--lr", str(lr),
                "--max-steps", str(steps),
                "--eval-every", str(steps),           # eval once, at the end
                "--run-name", f"sweep_{method}_lr{lr:g}",
                "--phase", "sweep",
            ]
            if smoke:
                args += ["--smoke", "true"]
            args += passthrough
# improvement

            try:
                record = train_main(args)
            except Exception:
                traceback.print_exc()
                print(f"  !! {method} lr={lr:g}: crashed (excluded)")
                _release_device()
                continue
            finally:
                _release_device()

            if record["status"] != "completed" or not record["evals"]:
                print(f"  !! {method} lr={lr:g}: {record['status']} (excluded)")
                continue
            loss = record["evals"][-1]["eval_loss"]
            # A diverged run reports NaN, and NaN loses every comparison in
            # min() by accident of IEEE ordering -- it must be excluded, not won.
#             if not math.isfinite(loss):
                print(f"  !! {method} lr={lr:g}: diverged (eval_loss={loss}) (excluded)")
                continue
            results.append((lr, loss))

        if not results:
            print(f"[sweep] {method}: no run produced a usable eval loss")
            continue
        best_lr, best_loss = min(results, key=lambda t: t[1])
        best[method] = best_lr
        edge = ""
        grid = DEFAULT_LR_GRIDS[method]
        if best_lr in (grid[0], grid[-1]):
            edge = "  [!] winner is on a grid EDGE -- extend the grid and re-sweep this method"
# improvement
        print(f"[sweep] {method}: best lr {best_lr:g} (eval_loss {best_loss:.4f}) "
              f"from {[(f'{lr:g}', f'{l:.4f}') for lr, l in results]}{edge}")

    out = RESULTS_DIR / "sweep_winners.jsonl"
    append_jsonl(out, {"winners": best, "steps": steps, "smoke": smoke,
                       "passthrough": passthrough})
    print(f"[sweep] winners -> {out}")
    return best


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Per-method LR sweep. Unrecognised flags are forwarded to train.py.",
    )
    ap.add_argument("--methods", nargs="+", default=["full", "lora", "galore"])
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--smoke", type=lambda v: v.lower() in ("1", "true", "yes"), default=False)
    ns, passthrough = ap.parse_known_args()

    # Fail on a typo'd knob now, not 40 minutes into rented GPU time.
    RunConfig.build_parser().parse_args(passthrough)
    run_sweep(ns.methods, ns.steps, ns.smoke, passthrough)

# Optimized

# Enhanced

# Enhanced

# Optimized

# Enhanced
