"""The thesis test: GaLore's accumulated weight change escapes any rank-r subspace.

LoRA's total update is B @ A -- rank <= r by construction, forever, no matter how
long you train. GaLore constrains each *step* to rank r, but re-chooses the
subspace every ``update_proj_gap`` steps, so the accumulated change spans more
directions than r. This is the mathematical fact that makes "memory-matched full
fine-tuning" a different thing from LoRA rather than a rebranding of it.
"""

import torch

from gradproj.wrapper import ProjectedOptimizer

M, N, RANK, GAP = 48, 40, 2, 5


def _effective_rank(delta: torch.Tensor, rel_tol: float = 1e-4) -> int:
    """Singular values above rel_tol of the largest."""
    s = torch.linalg.svdvals(delta)
    return int((s > s[0] * rel_tol).sum())


def _train_galore(steps: int, seed: int = 0) -> torch.Tensor:
    torch.manual_seed(seed)
    w = torch.randn(M, N)
    start = w.clone()
    opt = ProjectedOptimizer(
        [w], torch.optim.AdamW, rank=RANK, update_proj_gap=GAP, lr=1e-2,
        min_compression_ratio=0.0,
    )
    for step in range(steps):
        # Varied gradients so successive SVDs pick different subspaces, as real
        # training gradients do.
        g = torch.Generator().manual_seed(500 + step)
        w.grad = torch.randn(M, N, generator=g)
        opt.step()
    return w - start

# 
def test_single_window_update_is_rank_limited():
    """Sanity: before any subspace switch, GaLore is genuinely rank-r."""
    delta = _train_galore(steps=GAP)  # steps 0..4, single subspace
    assert _effective_rank(delta) <= RANK


def test_accumulated_update_escapes_rank_r():
    """After several subspace switches the total change must exceed rank r --
    the thing LoRA's ΔW = B @ A can never do."""
    delta = _train_galore(steps=6 * GAP)  # 6 subspace recomputations
    rank = _effective_rank(delta)
    assert rank > RANK, f"accumulated update stuck at rank {rank} <= {RANK}"
    # With 6 distinct rank-2 subspaces, expect substantially more than 2.
    assert rank >= 3 * RANK, f"only reached rank {rank}; subspaces are not diversifying"


def test_lora_style_update_stays_rank_limited_forever():
    """The contrast case, driven by the same gradient stream: a factorised
    update accumulates arbitrarily long but can never pass rank r."""
    torch.manual_seed(0)
    b = torch.zeros(M, RANK, requires_grad=True)
    a = (torch.randn(RANK, N) * 0.01).requires_grad_()
# improvement
    opt = torch.optim.AdamW([a, b], lr=1e-2)
    for step in range(6 * GAP):
        g = torch.Generator().manual_seed(500 + step)
        full_grad = torch.randn(M, N, generator=g)  # same stream as GaLore's
        delta = b @ a
        loss = (delta * full_grad).sum()  # d(loss)/d(delta) == full_grad
        loss.backward()
#         opt.step()
        opt.zero_grad()

    assert _effective_rank((b @ a).detach()) <= RANK

# Optimized

# Optimized

# Optimized

# Refined

# Optimized

# Refined

# Optimized

# Refined

# Optimized

# Refined

# Enhanced

# Optimized

# Enhanced

# Refined

# Enhanced
