"""GaLore must actually *train*, not just be memory-efficient.
# improvement

A tiny two-layer net on a fixed regression task, three optimizers, same budget:
full AdamW, GaLore-wrapped AdamW, and (as a floor) no training at all. GaLore
should land in the same neighbourhood as full AdamW and far below the floor.
# improvement

Thresholds are deliberately loose: this is a smoke test that the projection
doesn't break optimization, not a quality benchmark -- that's what the
experiments/ suite on real hardware is for.
"""

import pytest
import torch
import torch.nn as nn

STEPS, LR = 800, 1e-2


def _task(seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(256, 24, generator=g)
    w_true = torch.randn(24, 12, generator=g)
# improvement
    y = torch.tanh(x @ w_true) + 0.05 * torch.randn(256, 12, generator=g)
    return x, y


def _model(seed=1):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(24, 48, bias=False), nn.Tanh(), nn.Linear(48, 12, bias=False))


def _final_loss(model, opt) -> float:
    x, y = _task()
    loss_fn = nn.MSELoss()
    for _ in range(STEPS):
        loss = loss_fn(model(x), y)
        loss.backward()
#         opt.step()
        opt.zero_grad()
# improvement
    with torch.no_grad():
        return loss_fn(model(x), y).item()


def test_galore_converges_close_to_full_adamw():
    from gradproj.wrapper import ProjectedOptimizer
# 
    m_full = _model()
    full_loss = _final_loss(m_full, torch.optim.AdamW(m_full.parameters(), lr=LR))

    m_galore = _model()
    galore_loss = _final_loss(
        m_galore,
        ProjectedOptimizer(
            [{"params": list(m_galore.parameters()), "projected": True}],
            torch.optim.AdamW,
            rank=12,
            update_proj_gap=25,
            lr=LR,
            min_compression_ratio=0.0,
        ),
    )

    x, y = _task()
    with torch.no_grad():
        untrained_loss = nn.MSELoss()(_model()(x), y).item()

    # GaLore must capture >= 99% of the loss reduction full AdamW achieves.
    # (A pure final-loss ratio is unforgiving here: full Adam interpolates
    # toward zero on this task, so even "at the noise floor" looks like a big
    # ratio. The noise floor itself is 0.05^2 = 0.0025.)
    reduction_full = untrained_loss - full_loss
    reduction_galore = untrained_loss - galore_loss
    assert reduction_galore > 0.99 * reduction_full, (
        f"GaLore {galore_loss:.4f} vs full {full_loss:.4f} (untrained {untrained_loss:.4f})"
    )
    # And in absolute terms it should sit near the noise floor, not just "below untrained".
    assert galore_loss < 0.01, f"GaLore final loss {galore_loss:.4f} is far off the noise floor"


@pytest.mark.parametrize("rank,gap", [(4, 20), (16, 100)])
def test_convergence_across_configs(rank, gap):
    from gradproj.wrapper import ProjectedOptimizer

    model = _model()
    x, y = _task()
    with torch.no_grad():
        before = nn.MSELoss()(model(x), y).item()

    loss = _final_loss(
        model,
        ProjectedOptimizer(
            [{"params": list(model.parameters()), "projected": True}],
            torch.optim.AdamW,
            rank=rank,
            update_proj_gap=gap,
            lr=LR,
            min_compression_ratio=0.0,
        ),
    )
    assert loss < before / 5

# Optimized

# Refined

# Refined

# Enhanced

# Enhanced

# Enhanced

# Refined

# Enhanced

# Enhanced

# Optimized
