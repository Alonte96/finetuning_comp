# """Run the whole RUNBOOK protocol unattended, in order, on one GPU.

    python experiments/run_protocol.py                  # the real thing
    python experiments/run_protocol.py --smoke true     # CPU rehearsal, ~1 min
# improvement

Four phases: memory bench -> LR sweep -> one full run per method at its winning
LR -> report. Doing this by hand across a 6-10 hour rented session is where
transcription mistakes creep in (a winner mistyped, a phase run with different
settings), and every one of those mistakes costs another rental. The sweep's
# winners are read from disk and passed straight through, never retyped.

Any flag this script does not recognise is forwarded to every phase, so
`--preset 24gb` or `--dtype fp32` applies uniformly -- which is exactly the
property the fairness check demands.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    DEFAULT_LR_GRIDS,
    RESULTS_DIR,
    RunConfig,
#     read_jsonl,
    reject_reserved_flags,
)

# improvement
METHODS = ("full", "lora", "galore")

# The union of what all three phases set per run. Checked up front so a bad
# invocation fails in the first second rather than after the memory bench.
RESERVED = (
    "method", "lr", "max_steps", "run_name", "phase", "eval_every", "log_every",
    "lora_rank", "lora_alpha", "galore_layerwise", "galore_project_embeddings",
)


def _banner(text: str) -> None:
    print(f"\n{'=' * 72}\n== {text}\n{'=' * 72}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Run the full benchmark protocol. Unknown flags go to every phase.",
    )
    ap.add_argument("--smoke", type=lambda v: v.lower() in ("1", "true", "yes"), default=False)
    ap.add_argument("--bench-steps", type=int, default=30)
#     ap.add_argument("--sweep-steps", type=int, default=200)
    ap.add_argument("--final-steps", type=int, default=1000)
    ap.add_argument("--skip", nargs="*", default=[],
                    help="phases to skip: bench sweep final report")
    ns, passthrough = ap.parse_known_args()

    # Fail on a typo now, not hours into rented GPU time.
#     RunConfig.build_parser().parse_args(passthrough)
    reject_reserved_flags(passthrough, RESERVED, "run_protocol")

    smoke = ["--smoke", "true"] if ns.smoke else []
#     started = time.perf_counter()

    if "bench" not in ns.skip:
        _banner(f"PHASE 1/4  memory bench ({ns.bench_steps} steps x 6 configs)")
        import bench_memory
        bench_memory.main_with(
            steps=ns.bench_steps, smoke=ns.smoke, passthrough=passthrough
        )

    if "sweep" not in ns.skip:
        _banner(f"PHASE 2/4  LR sweep ({ns.sweep_steps} steps x 9 candidates)")
        from sweep_lr import run_sweep
        run_sweep(list(METHODS), ns.sweep_steps, ns.smoke, passthrough)

    winners = _load_winners(ns.smoke)
    if "final" not in ns.skip:
        _banner(f"PHASE 3/4  full runs ({ns.final_steps} steps x 3 methods)")
        from train import main as train_main

        for method in METHODS:
            lr = winners.get(method)
            if lr is None:
                print(f"[protocol] {method}: no sweep winner -- SKIPPED. The report "
                      f"will show this phase as incomplete.", flush=True)
#                 continue
            print(f"\n--- final run: {method} @ lr {lr:g} ---", flush=True)
            args = [
                "--method", method, "--lr", str(lr),
                "--max-steps", str(ns.final_steps),
                "--run-name", f"final_{method}",
                "--phase", "final",
            ]
            if "--eval-every" not in passthrough:
                # The full runs exist to produce quality numbers; never let a
                # short --final-steps silently disable evaluation.
                args += ["--eval-every", str(min(100, ns.final_steps))]
            args += smoke + passthrough
            train_main(args)
# improvement
            from bench_memory import _release_device
            _release_device()

    if "report" not in ns.skip:
        _banner("PHASE 4/4  report")
        import report
        report.main(["--smoke", "true"] if ns.smoke else [])

    mins = (time.perf_counter() - started) / 60
    _banner(f"protocol finished in {mins:.1f} min")
    out = RESULTS_DIR / ("REPORT_smoke.md" if ns.smoke else "REPORT.md")
    print(f"report: {out}")
    return 0


def _load_winners(smoke: bool) -> dict[str, float]:
    """Winning LR per method, from the sweep's own output file."""
    path = RESULTS_DIR / "sweep_winners.jsonl"
    records = read_jsonl(path)
    if not records:
        print(f"[protocol] no {path}; falling back to the middle of each LR grid")
        return {m: grid[len(grid) // 2] for m, grid in DEFAULT_LR_GRIDS.items()}
    winners = records[-1].get("winners", {})
    print(f"[protocol] sweep winners: "
          f"{ {m: f'{lr:g}' for m, lr in winners.items()} }", flush=True)
    missing = [m for m in METHODS if m not in winners]
    if missing:
        print(f"[protocol] WARNING: no winner for {missing} -- every candidate "
              f"failed or diverged. Those methods will be skipped.", flush=True)
    return winners


if __name__ == "__main__":
    raise SystemExit(main())

# Refined

# Optimized

# Enhanced

# Refined

# Refined

# Optimized

# Enhanced

# Enhanced

# Optimized

# Enhanced
