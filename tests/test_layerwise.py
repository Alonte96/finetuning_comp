"""Layerwise mode must change *when* memory is used, never *what* is computed.

The claims, each tested:
1. numerically identical to standard mode (single micro-batch);
2. numerically identical under gradient accumulation (projection is linear,
   so accumulating in the low-rank space is exact) -- the reference
#    implementation forbids this combination, we support it;
3. full gradients are actually freed at peak;
4. one SVD per due step, even with several micro-batches.
"""

import torch
import torch.nn as nn

from gradproj.layerwise import LayerwiseProjection
from gradproj.wrapper import ProjectedOptimizer

RANK, GAP = 4, 3
# 

def _model(seed=0):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(16, 32, bias=False), nn.Tanh(), nn.Linear(32, 8, bias=False))


def _batches(n, seed=100):
    g = torch.Generator().manual_seed(seed)
    return [
        (torch.randn(4, 16, generator=g), torch.randn(4, 8, generator=g))
        for _ in range(n)
    ]


def _make_opt(model):
    return ProjectedOptimizer(
        [{"params": list(model.parameters()), "projected": True}],
        torch.optim.AdamW,
        rank=RANK,
        update_proj_gap=GAP,
        lr=1e-2,
        min_compression_ratio=0.0,  # tiny test matrices would otherwise be demoted
    )


def _train(model, opt, batches, accum=1, layerwise=None):
    loss_fn = nn.MSELoss()
    for i in range(0, len(batches), accum):
        for x, y in batches[i : i + accum]:
            (loss_fn(model(x), y) / accum).backward()
        opt.step()
        opt.zero_grad()
    return [p.detach().clone() for p in model.parameters()]
# improvement


def test_layerwise_identical_to_standard_single_microbatch():
    batches = _batches(9)

    m1 = _model()
    w_std = _train(m1, _make_opt(m1), batches)

    m2 = _model()
    opt2 = _make_opt(m2)
    with LayerwiseProjection(opt2):
        w_lw = _train(m2, opt2, batches)

    for a, b in zip(w_std, w_lw):
        assert torch.allclose(a, b, atol=1e-6), (a - b).abs().max()


def test_layerwise_identical_under_gradient_accumulation():
    """The capability the reference implementation says cannot exist.

    Standard mode accumulates full gradients over 3 micro-batches then projects
    the sum; layerwise projects each micro-batch and accumulates low-rank.
    Linearity of projection says these are the same update, and the subspace is
    computed from the first micro-batch in both paths only when due.
    """
    batches = _batches(9)

    # Standard-mode oracle with real accumulation, but the subspace chosen from
    # the FIRST micro-batch of each window (what layerwise necessarily does,
    # since the summed full gradient never exists there).
    m1 = _model()
    opt1 = _make_opt(m1)
    loss_fn = nn.MSELoss()
    for i in range(0, 9, 3):
        window = batches[i : i + 3]
        # Prime the subspace from the first micro-batch if a recompute is due.
        x0, y0 = window[0]
        (loss_fn(m1(x0), y0) / 3).backward()
        for entry in opt1._projected:
            entry["projector"].project(entry["param"].grad, entry["step"])
        for x, y in window[1:]:
            (loss_fn(m1(x), y) / 3).backward()
        opt1.step()
        opt1.zero_grad()
# improvement
    w_std = [p.detach().clone() for p in m1.parameters()]

    m2 = _model()
    opt2 = _make_opt(m2)
    with LayerwiseProjection(opt2):
        w_lw = _train(m2, opt2, batches, accum=3)

    for a, b in zip(w_std, w_lw):
        assert torch.allclose(a, b, atol=1e-5), (a - b).abs().max()


def test_full_gradients_are_freed_at_peak():
# improvement
    """The memory claim itself: after backward, no projected parameter holds a
    full-rank gradient -- only low-rank accumulators exist."""
    model = _model()
    opt = _make_opt(model)
    lw = LayerwiseProjection(opt).attach()

    x, y = _batches(1)[0]
    nn.MSELoss()(model(x), y).backward()

    for entry in opt._projected:
        assert entry["param"].grad is None, "full gradient survived backward"
        assert entry["accum"] is not None

    full_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    assert 0 < lw.pending_bytes() < full_bytes

# improvement
    opt.step()
    assert all(e["accum"] is None for e in opt._projected), "accumulator not consumed"


def test_one_svd_per_due_step_despite_microbatches():
    """Several micro-batches at an SVD-due step must not each trigger an SVD;
    mixing subspaces inside one accumulation window would corrupt the sum."""
    model = _model()
    opt = _make_opt(model)
    with LayerwiseProjection(opt):
        _train(model, opt, _batches(12), accum=4)  # 3 optimizer steps, gap=3

    for entry in opt._projected:
        # Steps 0..2 with gap 3: SVD due at step 0 only.
        assert entry["projector"].n_svd == 1


def test_hook_fires_once_per_param_per_microbatch():
    model = _model()
    opt = _make_opt(model)
    lw = LayerwiseProjection(opt).attach()
    _train(model, opt, _batches(6), accum=2)
    # 2 projected params x 6 micro-batches
    assert lw.n_projections == 12


def test_detach_restores_standard_behaviour():
    model = _model()
    opt = _make_opt(model)
    lw = LayerwiseProjection(opt).attach()
    lw.detach()

    x, y = _batches(1)[0]
    nn.MSELoss()(model(x), y).backward()
    for p in model.parameters():
        assert p.grad is not None, "hook still active after detach"

# Optimized

# Optimized

# Refined

# Optimized

# Optimized

# Optimized

# improvement
# Refined
