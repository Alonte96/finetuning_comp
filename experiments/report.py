"""Aggregate results/runs.jsonl into the comparison table and plots.

    python experiments/report.py                # writes results/REPORT.md + plots
    python experiments/report.py --smoke true   # reads smoke_runs.jsonl instead

Refuses to compare runs whose fairness fingerprints differ: a memory/quality
table built from runs with different seq_len or batch settings is not a
comparison, it is an accident.
"""

from __future__ import annotations

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


def check_fairness(runs: dict[str, dict]) -> list[str]:
    problems = []
    fingerprints = {name: r.get("fairness", {}) for name, r in runs.items()}
    if not fingerprints:
        return ["no runs found"]
    reference_name = next(iter(fingerprints))
    reference = fingerprints[reference_name]
    for name, fp in fingerprints.items():
        diffs = {k: (fp.get(k), reference.get(k)) for k in reference if fp.get(k) != reference.get(k)}
        if diffs:
            problems.append(f"{name} differs from {reference_name} on {diffs}")
    return problems


def method_of(record: dict) -> str:
    return record.get("config", {}).get("method", "?")


def final_eval(record: dict) -> dict | None:
    return record["evals"][-1] if record.get("evals") else None


def build_table(runs: dict[str, dict]) -> str:
    header = (
        "| run | method | trainable % | peak GiB | exact? | predicted GiB | "
        "eval loss | eval ppl | tok/s | status |\n"
        "|---|---|---|---|---|---|---|---|---|---|\n"
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
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []

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
        if pts:
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", type=lambda v: v.lower() in ("1", "true", "yes"), default=False)
    ns = ap.parse_args()

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
        lines.append("Fairness check passed: identical model, data, seq_len, "
                     "batch, steps, seed and dtype across all runs.\n")

    lines.append(build_table(runs))

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
