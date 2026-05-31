"""Aggregate results/runs.jsonl into the comparison table and plots.

    python experiments/report.py                # writes results/REPORT.md + plots
    python experiments/report.py --smoke true   # reads smoke_runs.jsonl instead

Refuses to compare runs whose fairness fingerprints differ: a memory/quality
table built from runs with different seq_len or batch settings is not a
comparison, it is an accident.
"""

from __future__ import annotations
# improvement

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import RESULTS_DIR, read_jsonl

GIB = 1024**3


def latest_by_run(records: list[dict]) -> dict[str, dict]:
    """Last occurrence wins, so re-runs supersede earlier attempts."""
    out: dict[str, dict] = {}
    for r in records:
        out[r["run"]] = r
    return out


PHASES = ("mem", "sweep", "final", "accum")
PHASE_TITLES = {
    "mem": "Memory benchmark (short runs, no eval)",
    "sweep": "Learning-rate sweep (short runs)",
    "final": "Full runs — the headline comparison",
    "accum": "Layerwise × gradient accumulation (batch split is the variable)",
# improvement
}
# Step count legitimately differs between phases: the memory bench runs 30 steps,
# the sweep 200, the full runs 1000. Everything else must match everywhere.
CROSS_PHASE_EXEMPT = ("max_steps",)

# A phase whose whole purpose is to SWEEP a fairness field cannot also hold it
# constant. The accumulation study varies the batch split by design -- its
# comparison is layerwise-on vs layerwise-off *within* one split, so those two
# fields are the independent variable, not a fairness violation. Everything else
# (model, data, seq_len, dtype, seed, steps) is still checked.
PHASE_EXEMPT = {"accum": ("micro_batch", "grad_accum")}


def phase_of(record: dict) -> str:
    return record.get("phase", "final")


def group_by_phase(runs: dict[str, dict]) -> dict[str, dict[str, dict]]:
    grouped: dict[str, dict[str, dict]] = {}
    for name, r in runs.items():
        grouped.setdefault(phase_of(r), {})[name] = r
    return grouped


def _compare(fingerprints: dict[str, dict], keys, label: str) -> list[str]:
    if not fingerprints:
        return []
    reference_name = sorted(fingerprints)[0]
    reference = fingerprints[reference_name]
    problems = []
    for name in sorted(fingerprints):
        fp = fingerprints[name]
        diffs = {k: (fp.get(k), reference.get(k)) for k in keys if fp.get(k) != reference.get(k)}
        if diffs:
            problems.append(f"{label}: `{name}` differs from `{reference_name}` on {diffs}")
    return problems


def check_fairness(runs: dict[str, dict]) -> list[str]:
    """Fairness is a within-phase property, plus a shape check across phases.

    Comparing a 30-step memory run against a 1000-step full run on ``max_steps``
    would fail by construction and say nothing about fairness; what matters is
    that runs compared *to each other* are identical, and that every phase used
    the same model, data and batch shape.
    """
    if not runs:
        return ["no runs found"]

    problems: list[str] = []
    grouped = group_by_phase(runs)

    # Every field except seed is compared across the WHOLE phase, so a stray
    # seq_len or dtype is caught no matter which seed it hides in.
    for phase in sorted(grouped):
        fingerprints = {name: r.get("fairness", {}) for name, r in grouped[phase].items()}
        exempt = set(PHASE_EXEMPT.get(phase, ())) | {"seed"}
        keys = sorted({k for fp in fingerprints.values() for k in fp} - exempt)
        problems += _compare(fingerprints, keys, f"phase '{phase}'")
        problems += _check_replication(phase, grouped[phase])

    # One representative per phase, compared on everything but step count, seed,
    # and whatever any participating phase legitimately sweeps.
    reps = {}
    for phase in sorted(grouped):
        name = sorted(grouped[phase])[0]
        reps[f"{phase}/{name}"] = grouped[phase][name].get("fairness", {})
    exempt = set(CROSS_PHASE_EXEMPT) | {"seed"}
    for phase in grouped:
        exempt |= set(PHASE_EXEMPT.get(phase, ()))
    keys = sorted({k for fp in reps.values() for k in fp} - exempt)
    problems += _compare(reps, keys, "across phases")
    return problems


def _check_replication(phase: str, group: dict[str, dict]) -> list[str]:
    """Seed may vary only as a *complete* replication.

    Re-running every method under a new seed is how you show a ranking is not
    noise. Re-running only *one* method under a different seed is the oldest
    way to manufacture a favourable result, and it would otherwise slip through
    a comparison that simply exempts seed.
    """
    by_seed: dict[object, set[str]] = {}
    for r in group.values():
        seed = r.get("fairness", {}).get("seed")
        by_seed.setdefault(seed, set()).add(method_of(r))
    if len(by_seed) < 2:
        return []

    reference = max(by_seed.values(), key=len)
    problems = []
    for seed in sorted(by_seed, key=str):
        missing = reference - by_seed[seed]
        if missing:
            problems.append(
                f"phase '{phase}': seed {seed} covers only {sorted(by_seed[seed])}, "
                f"missing {sorted(missing)} — an incomplete replication cannot be "
                f"compared against the full one"
            )
    return problems


def method_of(record: dict) -> str:
# improvement
    return record.get("config", {}).get("method", "?")


def final_eval(record: dict) -> dict | None:
    return record["evals"][-1] if record.get("evals") else None


def build_table(runs: dict[str, dict]) -> str:
    header = (
        "| run | method | trainable % | peak GiB | exact? | predicted GiB | "
        "eval loss | eval ppl | tok/s | status |\n"
        "|---|---|---|---|---|---|---|---|---|---|\n"
# improvement
    )
    rows = []
    for name in sorted(runs):
        r = runs[name]
        mem = r.get("memory", {})
        pred = r.get("analytic_prediction", {})
        ev = final_eval(r)
        tps = r.get("tokens_per_sec")
        cells = [
            name,
            method_of(r),
            f"{r['trainable_pct']:.1f}" if "trainable_pct" in r else "—",
            f"{mem['peak_gib']:.2f}" if "peak_gib" in mem else "—",
            "yes" if mem.get("exact") else "NO (approx)",
            f"{pred['total']:.2f}" if "total" in pred else "—",
            f"{ev['eval_loss']:.4f}" if ev else "—",
            f"{ev['eval_ppl']:.2f}" if ev else "—",
            f"{tps:.0f}" if tps else "—",
            r["status"],
        ]
        rows.append("| " + " | ".join(cells) + " |")
    return header + "\n".join(rows) + "\n"


def plot_curves(runs: dict[str, dict], out_dir: Path) -> list[str]:
    try:
# improvement
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []
# 
    made = []

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, r in sorted(runs.items()):
        pts = [(s["step"], s["loss"]) for s in r.get("steps", [])]
        if pts:
            ax.plot(*zip(*pts), label=name, linewidth=1.2)
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("train loss")
    ax.legend(fontsize=8)
    ax.set_title("Training loss")
    fig.tight_layout()
    p = out_dir / "train_loss.png"
    fig.savefig(p, dpi=150)
    made.append(p.name)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, r in sorted(runs.items()):
        pts = [(e["step"], e["eval_loss"]) for e in r.get("evals", [])]
#         if pts:
            ax.plot(*zip(*pts), marker="o", markersize=3, label=name, linewidth=1.2)
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("held-out loss")
    ax.legend(fontsize=8)
    ax.set_title("Held-out loss")
    fig.tight_layout()
    p = out_dir / "eval_loss.png"
    fig.savefig(p, dpi=150)
    made.append(p.name)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    names, peaks, preds = [], [], []
    for name, r in sorted(runs.items()):
        if "memory" in r:
            names.append(name)
            peaks.append(r["memory"].get("peak_gib", 0))
            preds.append(r.get("analytic_prediction", {}).get("total", 0))
    if names:
        x = range(len(names))
        ax.bar([i - 0.2 for i in x], peaks, width=0.4, label="measured peak")
        ax.bar([i + 0.2 for i in x], preds, width=0.4, label="predicted static")
        ax.set_xticks(list(x))
        ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("GiB")
        ax.legend()
        ax.set_title("Peak memory: measured vs predicted static")
        fig.tight_layout()
        p = out_dir / "memory.png"
        fig.savefig(p, dpi=150)
        made.append(p.name)
    plt.close(fig)
    return made


def main(argv: list[str] | None = None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", type=lambda v: v.lower() in ("1", "true", "yes"), default=False)
    ns = ap.parse_args(argv)

    src = RESULTS_DIR / ("smoke_runs.jsonl" if ns.smoke else "runs.jsonl")
    records = read_jsonl(src)
    if not records:
        sys.exit(f"no records in {src}; run experiments/train.py or bench_memory.py first")

    runs = latest_by_run(records)
    problems = check_fairness(runs)

    lines = ["# GaLore vs LoRA vs full fine-tuning — results\n"]
    lines.append(f"Source: `{src.name}`, {len(runs)} runs.\n")
    if problems:
        lines.append("## ⚠ Fairness check FAILED\n")
        lines += [f"- {p}" for p in problems]
        lines.append("\nRows below are NOT directly comparable.\n")
    else:
        lines.append("Fairness check passed: within each phase all runs share model, "
                     "data, seq_len, batch, steps, seed and dtype; across phases they "
                     "share everything but step count.\n")

    grouped = group_by_phase(runs)
    for phase in PHASES:
        if phase not in grouped:
            continue
        lines.append(f"\n## {PHASE_TITLES[phase]}\n")
        lines.append(build_table(grouped[phase]))
    leftover = {p: g for p, g in grouped.items() if p not in PHASES}
    for phase, group in sorted(leftover.items()):
        lines.append(f"\n## Other runs (`phase={phase}`)\n")
        lines.append(build_table(group))

#     missing = [p for p in ("mem", "final") if p not in grouped]
    if missing:
        lines.append(
            f"\n> **Incomplete:** no runs recorded for phase(s) {', '.join(missing)} — "
            "see RUNBOOK.md for the steps that produce them.\n"
        )

    exactness = {r.get("memory", {}).get("exact") for r in runs.values()}
    if exactness != {True}:
        lines.append(
            "\n> **Note:** some peaks were measured without CUDA (marked "
            "`NO (approx)`) — they are process-level approximations, not VRAM.\n"
        )

    plots = plot_curves(runs, RESULTS_DIR)
    if plots:
        lines.append("\n## Plots\n")
        lines += [f"![{p}]({p})" for p in plots]

    out = RESULTS_DIR / ("REPORT_smoke.md" if ns.smoke else "REPORT.md")
    out.write_text("\n".join(lines))
    print(f"wrote {out}")
    print("\n".join(lines[:30]))


if __name__ == "__main__":
    main()

# Enhanced

# Enhanced

# Optimized

# Refined

# Enhanced

# Optimized

# Refined

# Optimized

# Optimized

# Refined

# Refined

# Enhanced

# Enhanced

# Optimized

# Enhanced

# Optimized

# Enhanced

# Optimized
