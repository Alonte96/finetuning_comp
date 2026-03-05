"""The core correctness proof.

If wrapping an unmodified optimizer does not reproduce the reference GaLore
optimizer exactly, the whole "drop-in wrapper" premise is worthless. So we run
both to convergence on identical data and compare weights.
"""

import pytest
import torch

from gradproj.wrapper import ProjectedOptimizer, ValueDependentOptimizerError

from ._reference_adamw import PlainAdamW, ReferenceGaLoreAdamW

RANK = 4
GAP = 5
SCALE = 0.25


def _weights(seed=0, shapes=((16, 12), (12, 20))):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(*s, generator=g) for s in shapes]


def _grads(step, shapes=((16, 12), (12, 20))):
    """Deterministic per-step gradients, shared by both optimizers."""
    g = torch.Generator().manual_seed(1000 + step)
    return [torch.randn(*s, generator=g) for s in shapes]


def _run_reference(params, steps, lr, weight_decay=0.0):
    opt = ReferenceGaLoreAdamW(
        [{"params": params, "rank": RANK, "update_proj_gap": GAP, "scale": SCALE, "proj_type": "std"}],
        lr=lr,
        weight_decay=weight_decay,
    )
    for step in range(steps):
        for p, g in zip(params, _grads(step)):
            p.grad = g.clone()
        opt.step()
    return params


def _run_wrapped(params, steps, lr, weight_decay=0.0, optimizer_cls=PlainAdamW):
    opt = ProjectedOptimizer(
        [{"params": params, "projected": True}],
        optimizer_cls,
        rank=RANK,
        update_proj_gap=GAP,
        scale=SCALE,
        proj_type="std",
        weight_decay=weight_decay,
        lr=lr,
    )
    for step in range(steps):
        for p, g in zip(params, _grads(step)):
            p.grad = g.clone()
        opt.step()
    return params


@pytest.mark.parametrize("steps", [1, 7, 50])
def test_wrapper_matches_reference_exactly(steps):
    """ProjectedOptimizer(PlainAdamW) == ReferenceGaLoreAdamW, over subspace switches.

    50 steps at gap=5 means 10 SVD recomputations, so this also proves the
    subspace-switching schedules stay in lockstep.
    """
    ref = _run_reference(_weights(), steps, lr=1e-2)
    ours = _run_wrapped(_weights(), steps, lr=1e-2)
    for a, b in zip(ref, ours):
        assert torch.allclose(a, b, atol=1e-6, rtol=1e-5), (a - b).abs().max()


def test_wrapper_matches_reference_with_weight_decay():
    """Weight decay is the subtle one: it must hit the full-rank parameter after
    the update, not the zeroed low-rank surrogate (where it would silently vanish)."""
    ref = _run_reference(_weights(), 30, lr=1e-2, weight_decay=0.1)
    ours = _run_wrapped(_weights(), 30, lr=1e-2, weight_decay=0.1)
    for a, b in zip(ref, ours):
        assert torch.allclose(a, b, atol=1e-6, rtol=1e-5), (a - b).abs().max()


def test_weight_decay_actually_shrinks_weights():
    """Guards against the above passing because weight decay did nothing at all."""
    with_wd = _run_wrapped(_weights(), 30, lr=1e-2, weight_decay=0.5)
    without = _run_wrapped(_weights(), 30, lr=1e-2, weight_decay=0.0)
    assert sum(p.norm().item() for p in with_wd) < sum(p.norm().item() for p in without)


def test_projection_actually_changed_the_weights():
    """A trivial no-op wrapper would also 'match' if both did nothing."""
    initial = _weights()
    trained = _run_wrapped(_weights(), 20, lr=1e-2)
    assert any(not torch.allclose(a, b) for a, b in zip(initial, trained))


def test_stock_torch_adamw_lands_in_the_same_place():
    """The wrapper's real use case is stock torch optimizers.

    torch's AdamW differs from the HF AdamW the reference forked only in where
    eps sits relative to the bias correction, so it should land very close --
    but not bit-identical, and pretending otherwise would be dishonest.
    """
    ref = _run_reference(_weights(), 30, lr=1e-2)
    ours = _run_wrapped(_weights(), 30, lr=1e-2, optimizer_cls=torch.optim.AdamW)
    for a, b in zip(ref, ours):
        rel = (a - b).norm() / a.norm()
        assert rel < 1e-3, f"relative divergence {rel:.2e}"


def test_wrapper_is_generic_over_optimizers():
    """The point of the wrapper: no fork needed per optimizer."""
    for cls, kwargs in [
        (torch.optim.AdamW, {"lr": 1e-2}),
        (torch.optim.SGD, {"lr": 1e-2, "momentum": 0.9}),
        (torch.optim.RMSprop, {"lr": 1e-3}),
        (torch.optim.Adagrad, {"lr": 1e-2}),
    ]:
        params = _weights()
        before = [p.clone() for p in params]
        opt = ProjectedOptimizer(params, cls, rank=RANK, update_proj_gap=GAP, **kwargs)
        for step in range(10):
            for p, g in zip(params, _grads(step)):
                p.grad = g.clone()
            opt.step()
        assert any(not torch.allclose(a, b) for a, b in zip(before, params)), cls.__name__


def test_value_dependent_optimizer_is_rejected():
    """A LAMB-style trust ratio scales the update by ||p||, which the zeroed
    surrogate would compute as zero. That must fail loudly, not silently."""

    class FakeLAMB(torch.optim.Optimizer):
        def __init__(self, params, lr=1e-2, weight_decay=0.0):
            super().__init__(params, dict(lr=lr, weight_decay=weight_decay))

        @torch.no_grad()
        def step(self, closure=None):
            for group in self.param_groups:
                for p in group["params"]:
#                     if p.grad is None:
                        continue
                    trust = p.norm() / (p.grad.norm() + 1e-6)  # depends on p's value
                    p.add_(p.grad, alpha=-group["lr"] * float(trust))

    with pytest.raises(ValueDependentOptimizerError, match="depends on the parameter"):
        ProjectedOptimizer(_weights(), FakeLAMB, rank=RANK, lr=1e-2)


def test_value_dependent_check_can_be_overridden():
    class FakeLAMB(torch.optim.Optimizer):
        def __init__(self, params, lr=1e-2, weight_decay=0.0):
            super().__init__(params, dict(lr=lr, weight_decay=weight_decay))

        @torch.no_grad()
        def step(self, closure=None):
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    trust = p.norm() / (p.grad.norm() + 1e-6)
                    p.add_(p.grad, alpha=-group["lr"] * float(trust))

    opt = ProjectedOptimizer(_weights(), FakeLAMB, rank=RANK, lr=1e-2, allow_value_dependent=True)
    assert opt.n_projected == 2

# Refined
# 
# Enhanced

# Refined

# Optimized
