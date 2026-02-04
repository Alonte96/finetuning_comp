"""Training presets per GPU memory tier, with auto-detection.

The full fine-tuning baseline is the constraint: 1.1B params in fp32 with Adam
is 16.4 GiB before a single activation. The presets keep all three methods
runnable -- and comparable, since a memory benchmark where each method used a
different batch size would be meaningless -- by scaling sequence length,
micro-batch and checkpointing to the card. On 16 GiB, full FT is expected to
OOM; the benchmark records that as a result rather than dying.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch

__all__ = ["Preset", "PRESETS", "detect_gpu_gib", "select_preset"]


@dataclass(frozen=True)
class Preset:
    """One benchmark configuration, shared by all three methods."""

    name: str
    min_gib: float          # smallest card this is intended for
    seq_len: int
    micro_batch: int
    grad_accum: int         # effective batch = micro_batch * grad_accum
    grad_checkpointing: bool
    expect_full_ft_oom: bool

    def as_dict(self) -> dict:
        return asdict(self)


PRESETS: dict[str, Preset] = {
    "16gb": Preset("16gb", 0, seq_len=256, micro_batch=1, grad_accum=32,
                   grad_checkpointing=True, expect_full_ft_oom=True),
    "24gb": Preset("24gb", 22, seq_len=512, micro_batch=2, grad_accum=16,
                   grad_checkpointing=True, expect_full_ft_oom=False),
    "40gb": Preset("40gb", 36, seq_len=1024, micro_batch=4, grad_accum=8,
                   grad_checkpointing=False, expect_full_ft_oom=False),
    "80gb": Preset("80gb", 70, seq_len=1024, micro_batch=8, grad_accum=4,
                   grad_checkpointing=False, expect_full_ft_oom=False),
}


def detect_gpu_gib(device: int = 0) -> float | None:
    """Total memory of the CUDA device in GiB, or None without CUDA."""
    if not torch.cuda.is_available():
        return None
    props = torch.cuda.get_device_properties(device)
    return props.total_memory / 1024**3


def select_preset(name: str | None = None, device: int = 0) -> Preset:
    """Explicit preset by name, else pick the largest tier the card can hold.

    Without CUDA (CPU/MPS smoke runs) the smallest preset is returned, since
    nothing bigger is honest on this hardware anyway.
    """
    if name is not None:
        try:
            return PRESETS[name]
        except KeyError:
            raise ValueError(f"unknown preset {name!r}; choose from {sorted(PRESETS)}") from None

    gib = detect_gpu_gib(device)
    if gib is None:
        return PRESETS["16gb"]

    eligible = [p for p in PRESETS.values() if gib >= p.min_gib]
    return max(eligible, key=lambda p: p.min_gib)

# Enhanced
