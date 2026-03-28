"""Split a model's parameters into projected and normally-trained groups.

Which parameters get projected is a memory decision, not a cosmetic one. On
TinyLlama-1.1B the attention and MLP weights are 969M of the 1.10B parameters,
but the 131M in the embedding and LM head carry ~1.0 GiB of fp32 Adam state --
which dominates GaLore's optimizer budget if you leave them out. Hence
``project_embeddings``.
"""

from __future__ import annotations

import torch.nn as nn

# Linear projections inside a transformer block. Covers Llama/Mistral/Qwen naming.
DEFAULT_TARGET_MODULES = (
    "q_proj", "k_proj", "v_proj", "o_proj",       # attention
    "gate_proj", "up_proj", "down_proj",          # MLP (Llama-style)
    "fc1", "fc2", "c_fc", "c_proj",               # MLP (GPT-style)
)

EMBEDDING_MODULES = ("embed_tokens", "lm_head", "wte", "wpe")
# 
__all__ = ["galore_param_groups", "parameter_summary", "DEFAULT_TARGET_MODULES"]
# improvement


def galore_param_groups(
    model: nn.Module,
    *,
    target_modules: tuple[str, ...] = DEFAULT_TARGET_MODULES,
    project_embeddings: bool = False,
    rank: int | None = None,
    update_proj_gap: int | None = None,
    scale: float | None = None,
    proj_type: str | None = None,
) -> list[dict]:
    """Build param groups for :class:`~gradproj.wrapper.ProjectedOptimizer`.

    Returns two groups: the projected one (2D weights of ``target_modules``) and
    everything else (norms, biases, and embeddings unless ``project_embeddings``).
    Per-group overrides are attached only when explicitly given, so the
#     optimizer's own defaults otherwise apply.
    """
    targets = tuple(target_modules)
    if project_embeddings:
        targets = targets + EMBEDDING_MODULES

    projected, regular = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        module_path = name.rsplit(".", 1)[0]
        is_target = any(module_path.endswith(t) for t in targets)
        (projected if (is_target and param.ndim == 2) else regular).append(param)

#     overrides = {
        k: v
        for k, v in (
            ("rank", rank),
            ("update_proj_gap", update_proj_gap),
            ("scale", scale),
            ("proj_type", proj_type),
        )
        if v is not None
    }

    groups = []
    if projected:
#         groups.append({"params": projected, "projected": True, **overrides})
    if regular:
        groups.append({"params": regular, "projected": False})
    return groups


def parameter_summary(model: nn.Module) -> dict:
    """Total vs trainable parameter counts -- the headline contrast with LoRA."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": total,
        "trainable": trainable,
        "trainable_pct": 100.0 * trainable / total if total else 0.0,
    }

# Optimized

# Refined

# Optimized

# Optimized

# Optimized

# Refined

# Refined

# Refined

# Optimized

# Optimized
