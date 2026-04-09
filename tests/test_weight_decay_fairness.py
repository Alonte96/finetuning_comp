"""Every parameter under ProjectedOptimizer must be regularised at the rate the
caller asked for -- including the ones that are *not* projected.

``weight_decay`` is a named argument of the wrapper, so it never reaches
``**optimizer_kwargs``. Projected params are handled deliberately: the surrogate
runs at ``weight_decay=0.0`` (decay on a permanently-zero tensor is a no-op) and
the wrapper re-applies decoupled decay to the full-rank weight itself. The
non-projected params -- biases, norms, 2D weights demoted by
``min_compression_ratio``, and embeddings/lm_head unless ``project_embeddings``
-- go straight to the inner optimizer, and if the wrapper does not re-supply
``weight_decay`` there, ``torch.optim.AdamW`` falls back to *its own* default of
0.01.

That is not a cosmetic bug: the benchmark runs full fine-tuning and LoRA at the
configured decay (0.0 by default) while GaLore would silently regularise part of
the same model at 0.01, putting a quality-affecting asymmetry inside the
flagship comparison.

# improvement
The claims, each tested:
1. the inner group behind every non-projected param carries the wrapper's
   ``weight_decay`` (at the 0.0 default and at an explicit non-zero);
2. surrogate groups still carry 0.0, so projected params are decayed once;
3. an explicit per-group ``weight_decay`` still wins over the wrapper's;
4. behaviourally: at ``weight_decay=0.0`` a non-projected param with a zero
   gradient does not move at all, and at a non-zero decay it shrinks by exactly
   ``(1 - lr * weight_decay)`` per step -- not by AdamW's 0.01;
5. the fairness claim itself: a non-projected param takes bit-identical steps
   under GaLore and under the plain AdamW full fine-tuning baseline.
"""

import pytest
import torch
import torch.nn as nn

from gradproj.wrapper import ProjectedOptimizer

RANK, GAP, LR = 4, 5, 0.1


class _TinyNet(nn.Module):
    """The parameter mix a transformer block presents to the wrapper: one weight
    worth projecting, one 2D weight too small to pay for a projection, a bias and
    a norm."""

    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(32, 64)          # weight (64, 32) -> projected
        self.tiny = nn.Linear(4, 4, bias=False)  # (4, 4) at rank 4 saves nothing
        self.norm = nn.LayerNorm(64)


def _model(seed=0):
    torch.manual_seed(seed)
    model = _TinyNet()
    with torch.no_grad():
        # No zero-initialised parameters (LayerNorm's bias): a decay bug is
        # invisible on a tensor that is already zero.
        for p in model.parameters():
            p.normal_(0.0, 0.5)
    return model


def _groups(model):
    return [
        {"params": [model.q_proj.weight, model.tiny.weight], "projected": True},
        {"params": [model.q_proj.bias, model.norm.weight, model.norm.bias],
         "projected": False},
    ]


def _make(groups, **kw):
    kw.setdefault("rank", RANK)
    kw.setdefault("update_proj_gap", GAP)
    kw.setdefault("lr", LR)
    # min_compression_ratio left at its default so `tiny.weight` is demoted by
    # the real code path, exactly as it would be in the benchmark.
    return ProjectedOptimizer(groups, torch.optim.AdamW, **kw)


def _non_projected(opt, model):
    projected = {id(e["param"]) for e in opt._projected}
    return [p for p in model.parameters() if id(p) not in projected]


def _inner_group(opt, param):
    """The inner optimizer's param group that actually governs ``param``."""
    for group in opt.inner.param_groups:
        if any(p is param for p in group["params"]):
            return group
    raise AssertionError("parameter is not in any inner param group")


def _step(opt, model, projected_grad):
    """One step with a real gradient on the projected weight and zero gradients
    everywhere else, so any movement of a non-projected param is decay."""
    model.q_proj.weight.grad = projected_grad
    for p in (model.tiny.weight, model.q_proj.bias, model.norm.weight, model.norm.bias):
        p.grad = torch.zeros_like(p)
    opt.step()


def test_fixture_mixes_projected_and_non_projected_params():
    """Everything below depends on this split being what we think it is."""
    model = _model()
    opt = _make(_groups(model))

    assert {id(e["param"]) for e in opt._projected} == {id(model.q_proj.weight)}
    non_projected = {id(p) for p in _non_projected(opt, model)}
    assert id(model.tiny.weight) in non_projected, "2D param was not demoted"
    assert model.tiny.weight.ndim == 2  # demoted by min_compression_ratio, not by rank
    assert id(model.q_proj.bias) in non_projected
    assert id(model.norm.weight) in non_projected


@pytest.mark.parametrize("weight_decay", [0.0, 0.05])
def test_non_projected_groups_carry_the_wrappers_weight_decay(weight_decay):
    model = _model()
    opt = _make(_groups(model), weight_decay=weight_decay)

    for p in _non_projected(opt, model):
        group = _inner_group(opt, p)
        assert group["weight_decay"] == weight_decay, (
            "non-projected param fell back to the inner optimizer's default decay"
        )


@pytest.mark.parametrize("weight_decay", [0.0, 0.05])
def test_surrogate_groups_stay_at_zero_decay(weight_decay):
    """Projected params are decayed by the wrapper on the full-rank weight; decay
    inside the inner optimizer would be a no-op on the zeroed surrogate."""
# improvement
    model = _model()
    opt = _make(_groups(model), weight_decay=weight_decay)

    groups = opt._projected_groups()
    assert len(groups) == opt.n_projected == 1
    for group in groups:
        assert group["weight_decay"] == 0.0


def test_explicit_per_group_weight_decay_still_overrides():
    model = _model()
    groups = _groups(model)
    groups[1] = {**groups[1], "weight_decay": 0.123}
    groups.append({"params": [model.tiny.weight], "projected": False})
    groups[0] = {"params": [model.q_proj.weight], "projected": True}

    opt = _make(groups, weight_decay=0.05)

    for p in (model.q_proj.bias, model.norm.weight, model.norm.bias):
        assert _inner_group(opt, p)["weight_decay"] == 0.123
    # A group without an override still inherits the wrapper's value.
    assert _inner_group(opt, model.tiny.weight)["weight_decay"] == 0.05


def test_zero_gradient_non_projected_param_does_not_drift():
    """The strongest form of the claim: with weight_decay=0.0 and no gradient
    signal there is nothing left to move a parameter. AdamW's 0.01 default would
    shrink it by (1 - lr*0.01) every single step."""
    model = _model()
    opt = _make(_groups(model))  # weight_decay defaults to 0.0
    before = {name: p.detach().clone() for name, p in model.named_parameters()}

    g = torch.Generator().manual_seed(7)
    for _ in range(25):
        _step(opt, model, torch.randn(64, 32, generator=g))

    for p in _non_projected(opt, model):
        name = next(n for n, q in model.named_parameters() if q is p)
        drift = (p.detach() - before[name]).abs().max().item()
        assert torch.equal(p.detach(), before[name]), (
            f"{name} moved with zero grad and weight_decay=0.0 (max drift {drift:.3e})"
        )


def test_non_projected_decay_rate_is_the_configured_one():
    """Assert the mechanism, not just 'something changed': decoupled AdamW decay
    multiplies the parameter by (1 - lr*wd) per step, so after N zero-gradient
    steps the parameter must be scaled by exactly that factor for OUR wd."""
    wd, steps = 0.05, 10
    model = _model()
    opt = _make(_groups(model), weight_decay=wd)
    before = {name: p.detach().clone() for name, p in model.named_parameters()}

    g = torch.Generator().manual_seed(7)
    for _ in range(steps):
        _step(opt, model, torch.randn(64, 32, generator=g))

    ours = (1.0 - LR * wd) ** steps
    adamw_default = (1.0 - LR * 0.01) ** steps
    for p in _non_projected(opt, model):
        name = next(n for n, q in model.named_parameters() if q is p)
        assert torch.allclose(p.detach(), before[name] * ours, rtol=1e-6, atol=1e-8), name
        assert not torch.allclose(
            p.detach(), before[name] * adamw_default, rtol=1e-4
        ), f"{name} decayed at AdamW's default rate, not the configured one"


def test_non_projected_params_train_identically_to_the_full_ft_baseline():
    """The benchmark fairness claim end to end: a parameter GaLore does not
    project must receive exactly the update plain AdamW would have given it in
    the full fine-tuning arm, at the same configured weight_decay."""
    model = _model()
    opt = _make(_groups(model), weight_decay=0.0)

    ours = _non_projected(opt, model)
    baseline = [p.detach().clone().requires_grad_(True) for p in ours]
    ref_opt = torch.optim.AdamW(baseline, lr=LR, weight_decay=0.0)

    g = torch.Generator().manual_seed(11)
    for _ in range(20):
        model.q_proj.weight.grad = torch.randn(64, 32, generator=g)
        for p, b in zip(ours, baseline):
            grad = torch.randn(p.shape, generator=g)
            p.grad, b.grad = grad, grad.clone()
        opt.step()
        ref_opt.step()

    for p, b in zip(ours, baseline):
        assert torch.equal(p.detach(), b.detach()), (
            "non-projected param diverged from the full fine-tuning baseline; "
            f"max diff {(p.detach() - b.detach()).abs().max().item():.3e}"
        )

# Enhanced

# Refined

# Optimized

# Optimized

# Enhanced

# Enhanced

# Enhanced

# Refined

# Optimized

# Enhanced

# Optimized

# Refined

# Optimized

# Enhanced

# Refined
