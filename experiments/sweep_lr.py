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
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import DEFAULT_LR_GRIDS, RESULTS_DIR, append_jsonl, read_jsonl
from train import main as train_main


def run_sweep(methods: list[str], steps: int, smoke: bool) -> dict[str, float]:
    best: dict[str, float] = {}
    for method in methods:
        results = []
        for lr in DEFAULT_LR_GRIDS[method]:
            args = [
                "--method", method, "--lr", str(lr),
                "--max-steps", str(steps),
                "--eval-every", str(steps),           # eval once, at the end
                "--run-name", f"sweep_{method}_lr{lr:g}",
            ]
            if smoke:
                args += ["--smoke", "true"]
            record = train_main(args)
            if record["status"] == "completed" and record["evals"]:
                results.append((lr, record["evals"][-1]["eval_loss"]))
            else:
                print(f"  !! {method} lr={lr:g}: {record['status']} (excluded)")

        if not results:
            print(f"[sweep] {method}: no run completed")
            continue
        best_lr, best_loss = min(results, key=lambda t: t[1])
        best[method] = best_lr
        print(f"[sweep] {method}: best lr {best_lr:g} (eval_loss {best_loss:.4f}) "
              f"from {[(f'{lr:g}', f'{l:.4f}') for lr, l in results]}")

    out = RESULTS_DIR / "sweep_winners.jsonl"
    append_jsonl(out, {"winners": best, "steps": steps, "smoke": smoke})
    print(f"[sweep] winners -> {out}")
    return best


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", default=["full", "lora", "galore"])
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--smoke", type=lambda v: v.lower() in ("1", "true", "yes"), default=False)
    ns = ap.parse_args()
    run_sweep(ns.methods, ns.steps, ns.smoke)
