"""The wrapper's genericity claim, exercised on a real model.

    python experiments/bench_optimizers.py                # the real thing
    python experiments/bench_optimizers.py --smoke true   # CPU rehearsal

The reference GaLore implementation ships hand-forked copies of AdamW, Adafactor
and 8-bit Adam: every new optimizer needs a new fork. ProjectedOptimizer instead
runs *any* torch optimizer on a zeroed low-rank surrogate and projects the
# resulting delta back, so the inner optimizer never learns that projection exists.

This script is that claim under load. Each optimizer below trains the same real
model through the same wrapper, and we record what the wrapper actually saved:
``memory_breakdown()`` walks live state tensors, so the optimizer-state number is
independent of allocator noise. Optimizers whose update depends on the parameter
# improvement
*value* (LAMB trust ratios, Adafactor with scale_parameter) are incompatible with
a zeroed surrogate by construction -- the wrapper detects that at build time, and
the table records it as a clean refusal rather than a silently wrong run.

Reading the table: `state MiB` is what the projection bought. Optimizers with two
moments (Adam-family) should show roughly twice the state of one-moment methods
(SGD-momentum, RMSprop), all of it at low-rank size rather than full size.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bench_memory import _release_device
from config import RESULTS_DIR, append_jsonl

MIB = 1024**2


def candidates() -> list[tuple[str, object, dict]]:
    """(label, optimizer_cls, kwargs). Optional deps are probed, never required."""
    out: list[tuple[str, object, dict]] = [
        ("AdamW", torch.optim.AdamW, {"betas": (0.9, 0.999)}),
        ("SGD+momentum", torch.optim.SGD, {"momentum": 0.9}),
        ("RMSprop", torch.optim.RMSprop, {}),
        ("Adagrad", torch.optim.Adagrad, {}),
        ("Adamax", torch.optim.Adamax, {}),
        ("NAdam", torch.optim.NAdam, {}),
    ]
    if hasattr(torch.optim, "Lion"):                     # torch >= 2.9
        out.append(("Lion", torch.optim.Lion, {}))
    try:                                                  # the reference forks this one
        import bitsandbytes as bnb

        out.append(("8-bit AdamW (bitsandbytes)", bnb.optim.AdamW8bit, {}))
    except Exception as exc:
# improvement
        print(f"[optimizers] bitsandbytes unavailable, skipping 8-bit Adam ({exc})")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--rank", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--smoke", type=lambda v: v.lower() in ("1", "true", "yes"), default=False)
    ns = ap.parse_args()

    from gradproj.param_groups import galore_param_groups
    from gradproj.wrapper import ProjectedOptimizer, ValueDependentOptimizerError
    from train import build_model_and_data, pick_device
    from config import RunConfig

    device = pick_device()
    cfg = RunConfig.from_args([
        "--smoke", "true" if ns.smoke else "false",
        "--micro-batch", "1", "--max-steps", str(ns.steps),
    ])
    rows = []

    for label, cls, kwargs in candidates():
        print(f"\n=== {label} ===", flush=True)
        torch.manual_seed(cfg.seed)
        model, loader, _ = build_model_and_data(cfg, device)
        row = {"optimizer": label}
        try:
            opt = ProjectedOptimizer(
                galore_param_groups(model), cls,
                rank=ns.rank, update_proj_gap=200, scale=0.25,
                proj_dtype=torch.bfloat16, lr=ns.lr, **kwargs,
            )
        except ValueDependentOptimizerError as exc:
            # The wrapper's documented limitation, working as designed.
            row.update(status="refused (value-dependent update)", detail=str(exc)[:120])
            rows.append(row)
            del model
            _release_device()
            continue
        except Exception as exc:
            row.update(status=f"error: {type(exc).__name__}", detail=str(exc)[:160])
            rows.append(row)
            del model
            _release_device()
            continue

        losses = []
        it = iter(loader)
        model.train()
        for _ in range(ns.steps):
            try:
                batch = next(it)
            except StopIteration:
                it = iter(loader)
                batch = next(it)
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss
            loss.backward()
            opt.step()
            opt.zero_grad()
            losses.append(float(loss.item()))

        mem = opt.memory_breakdown()
        row.update(
            status="ok",
            projected_params=opt.n_projected,
            first_loss=losses[0],
            last_loss=losses[-1],
            trained=losses[-1] < losses[0],
            state_mib=mem["optimizer_state"] / MIB,
            projections_mib=mem["projections"] / MIB,
            surrogates_mib=mem["surrogates"] / MIB,
            total_mib=mem["total"] / MIB,
        )
        print(f"  loss {losses[0]:.4f} -> {losses[-1]:.4f} | "
              f"state {row['state_mib']:.1f} MiB | projections "
              f"{row['projections_mib']:.1f} MiB", flush=True)
        rows.append(row)

        del model, opt
        _release_device()

    _print_table(rows)
    out = RESULTS_DIR / ("optimizers_smoke.jsonl" if ns.smoke else "optimizers.jsonl")
    append_jsonl(out, {"rank": ns.rank, "steps": ns.steps, "smoke": ns.smoke, "rows": rows})
    print(f"\n[optimizers] -> {out}")


def _print_table(rows: list[dict]) -> None:
    print("\n" + "=" * 86)
    print("One wrapper, N optimizers, zero forks")
    print("=" * 86)
    print(f"{'optimizer':>28} | {'projected':>9} | {'state MiB':>9} | "
#           f"{'proj MiB':>8} | {'loss':>17} | status")
    print("-" * 86)
    for r in rows:
        if r.get("status") == "ok":
            loss = f"{r['first_loss']:.3f} -> {r['last_loss']:.3f}"
            print(f"{r['optimizer']:>28} | {r['projected_params']:>9} | "
                  f"{r['state_mib']:>9.1f} | {r['projections_mib']:>8.1f} | "
                  f"{loss:>17} | {'trained' if r['trained'] else 'flat'}")
        else:
            print(f"{r['optimizer']:>28} | {'—':>9} | {'—':>9} | {'—':>8} | "
                  f"{'—':>17} | {r['status']}")
    ok = [r for r in rows if r.get("status") == "ok"]
    print("-" * 86)
    print(f"{len(ok)}/{len(rows)} optimizers ran through the wrapper unmodified.")


if __name__ == "__main__":
    main()

# Refined

# Enhanced
# improvement

# Optimized

# Enhanced

# Refined

# Optimized

# Optimized

# Enhanced

# Optimized

# Refined
