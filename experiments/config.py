"""Shared experiment configuration and the fairness protocol.

Every knob that could bias the three-way comparison lives in ONE dataclass used
by ALL methods. Method-specific settings (LoRA rank, GaLore rank, per-method LR)
are explicit, separate fields, so any asymmetry is visible in one place.
"""

from __future__ import annotations

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
    galore_project_embeddings: bool = False

    # ---- bookkeeping ----
    preset: str | None = None
    run_name: str = ""
    out_dir: str = str(RESULTS_DIR)
    save_model: bool = False              # benchmark runs don't need checkpoints
    eval_every: int = 100
    log_every: int = 10
    smoke: bool = False                   # tiny random model + synthetic data, no network

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
    def from_args(cls, argv: list[str] | None = None) -> "RunConfig":
        parser = argparse.ArgumentParser(description="GaLore vs LoRA vs full FT")
        for f in dataclasses.fields(cls):
            if f.name == "run_name":
                default = ""
            else:
                default = f.default if f.default is not dataclasses.MISSING else None
            arg = f"--{f.name.replace('_', '-')}"
            if f.type == "bool" or isinstance(default, bool):
                parser.add_argument(arg, type=lambda v: v.lower() in ("1", "true", "yes"),
                                    default=default)
            elif f.name == "preset":
                parser.add_argument(arg, type=str, default=None)
            else:
                parser.add_argument(arg, type=type(default) if default is not None else str,
                                    default=default)
        args = parser.parse_args(argv)
        cfg = cls(**vars(args))

        if cfg.preset:
            from gradproj.presets import select_preset
            p = select_preset(cfg.preset)
            cfg.seq_len, cfg.micro_batch = p.seq_len, p.micro_batch
            cfg.grad_accum, cfg.grad_checkpointing = p.grad_accum, p.grad_checkpointing

        if cfg.method not in ("full", "lora", "galore"):
            raise SystemExit(f"--method must be full|lora|galore, got {cfg.method}")
        # GaLore layerwise + accumulation is supported (low-rank accumulators are
        # exact); no restriction needed here.
        return cfg


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]
