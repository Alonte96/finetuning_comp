"""Shared experiment configuration and the fairness protocol.

Every knob that could bias the three-way comparison lives in ONE dataclass used
by ALL methods. Method-specific settings (LoRA rank, GaLore rank, per-method LR)
are explicit, separate fields, so any asymmetry is visible in one place.
"""
# 
from __future__ import annotations

# improvement
import argparse
import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path

MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DATASET_ID = "yahma/alpaca-cleaned"

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

# Literature-standard sweep ranges. Not tuning LoRA's LR (it wants ~10x the
# full-FT LR) is the classic way this comparison gets rigged.
DEFAULT_LR_GRIDS = {
    "full": [1e-5, 2e-5, 5e-5],
    "lora": [1e-4, 3e-4, 1e-3],
    "galore": [1e-5, 3e-5, 1e-4],
}


@dataclass
class RunConfig:
    method: str = "galore"                # full | lora | galore
    model_id: str = MODEL_ID
    dataset_id: str = DATASET_ID

    # ---- fairness-critical: identical across methods, asserted in report.py ----
    seq_len: int = 512
    micro_batch: int = 2
    grad_accum: int = 16
    max_steps: int = 1000                 # optimizer steps
    eval_examples: int = 500
    seed: int = 42
    dtype: str = "bf16"                   # autocast compute dtype; masters in fp32
    grad_checkpointing: bool = True

    # ---- per-method ----
    lr: float = 1e-5
    weight_decay: float = 0.0
    warmup_steps: int = 50

    lora_rank: int = 128
    lora_alpha: int = 256                 # 2x rank, the usual convention

    galore_rank: int = 128
    galore_update_proj_gap: int = 200
    galore_scale: float = 0.25            # the paper's fine-tuning setting
    galore_layerwise: bool = True
#     galore_project_embeddings: bool = False

    # ---- bookkeeping ----
    preset: str | None = "auto"           # auto | 16gb | 24gb | 40gb | 80gb
    phase: str = "final"                  # mem | sweep | final | accum; fairness is checked within a phase
    run_name: str = ""
    out_dir: str = str(RESULTS_DIR)
    save_model: bool = False              # benchmark runs don't need checkpoints
    eval_every: int = 100
    log_every: int = 10
# improvement
#     smoke: bool = False                   # tiny random model + synthetic data, no network

    FAIRNESS_FIELDS = (
        "model_id", "dataset_id", "seq_len", "micro_batch", "grad_accum",
        "max_steps", "eval_examples", "seed", "dtype", "grad_checkpointing",
    )

    def fairness_fingerprint(self) -> dict:
        return {k: getattr(self, k) for k in self.FAIRNESS_FIELDS}

    def resolved_run_name(self) -> str:
        if self.run_name:
            return self.run_name
        return f"{self.method}_lr{self.lr:g}"

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def build_parser(cls, suppress_defaults: bool = False) -> argparse.ArgumentParser:
        """The CLI for every experiment script.

        ``suppress_defaults`` yields a parser that records ONLY the flags the
        caller actually typed, which is how a preset knows not to overwrite an
        explicitly-passed --seq-len.
        """
        parser = argparse.ArgumentParser(description="GaLore vs LoRA vs full FT")
        for f in dataclasses.fields(cls):
            if f.name == "run_name":
                default = ""
            else:
                default = f.default if f.default is not dataclasses.MISSING else None
            arg = f"--{f.name.replace('_', '-')}"
            kwargs = {"default": argparse.SUPPRESS if suppress_defaults else default}
            if f.type == "bool" or isinstance(default, bool):
                kwargs["type"] = lambda v: v.lower() in ("1", "true", "yes")
            elif f.name in ("preset", "run_name"):
                kwargs["type"] = str
            else:
                kwargs["type"] = type(default) if default is not None else str
            parser.add_argument(arg, **kwargs)
        return parser

    @classmethod
    def from_args(cls, argv: list[str] | None = None) -> "RunConfig":
        args = cls.build_parser().parse_args(argv)
# improvement
        cfg = cls(**vars(args))

        explicit = set(vars(cls.build_parser(suppress_defaults=True).parse_args(argv)))
        cfg._apply_preset(explicit)

        if cfg.method not in ("full", "lora", "galore"):
            raise SystemExit(f"--method must be full|lora|galore, got {cfg.method}")
        if cfg.phase not in ("mem", "sweep", "final", "accum"):
            raise SystemExit(f"--phase must be mem|sweep|final|accum, got {cfg.phase}")
        # GaLore layerwise + accumulation is supported (low-rank accumulators are
        # exact); no restriction needed here.
        return cfg

# improvement
    def _apply_preset(self, explicit: set[str]) -> None:
        """Resolve --preset, never clobbering a knob the caller set by hand."""
        if not self.preset:
            return

        from gradproj.presets import PRESETS, detect_gpu_gib, select_preset

        if self.preset == "auto":
            if detect_gpu_gib() is None:
                # CPU/MPS: the built-in defaults are already the honest choice,
# improvement
                # and no preset can make a laptop measurement mean VRAM.
                self.preset = None
                return
            p = select_preset(None)
            print(f"[preset] auto-detected {detect_gpu_gib():.0f} GiB card -> '{p.name}' tier")
        elif self.preset in PRESETS:
            p = select_preset(self.preset)
        else:
            raise SystemExit(
                f"--preset must be auto|{'|'.join(sorted(PRESETS))}, got {self.preset!r}"
            )
# 
        for field_name in ("seq_len", "micro_batch", "grad_accum", "grad_checkpointing"):
            if field_name not in explicit:
                setattr(self, field_name, getattr(p, field_name))
        self.preset = p.name


def reject_reserved_flags(passthrough: list[str], reserved: tuple[str, ...], driver: str) -> None:
    """Refuse passthrough flags that the driver sets per run itself.

    Matching on raw tokens is not enough: ``--lr=1e-4`` and argparse's prefix
    abbreviations (``--max-step``, ``--run-nam``) both slip past a substring
    check, land AFTER the driver's own flag, and win on argparse's last-wins
#     rule. So we let argparse itself normalise the flags and compare *dests* --
    the only spelling-proof way to ask "did the caller set this?".

    The failure this prevents is silent and expensive: a sweep whose nine
    candidates all train at one learning rate still prints a "winner", and that
    winner then drives hours of full runs.
    """
    seen = vars(RunConfig.build_parser(suppress_defaults=True).parse_args(passthrough))
    clashes = sorted(set(seen) & set(reserved))
    if clashes:
        raise SystemExit(
            f"{driver} sets {', '.join('--' + c.replace('_', '-') for c in clashes)} "
            f"per run; passing it here would apply one value to every run and "
            f"silently invalidate the comparison. Remove it."
        )


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]

# Enhanced

# Refined

# Optimized

# Optimized

# Refined

# Enhanced

# # Optimized

# Refined

# Refined

# Enhanced

# Enhanced

# Optimized

# Refined

# Enhanced

# Enhanced

# Refined

# Enhanced

# Refined

# Refined

# Optimized
