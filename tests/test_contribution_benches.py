"""The two contribution benches must not be able to report a false success.

`bench_accum` exists to defend one claim -- layerwise projection under gradient
accumulation is the *same optimisation* at *lower peak memory* -- so the failure
mode that matters is a harness that reports agreement when the trajectories
actually diverged, or a memory saving computed from a run that never happened.

`bench_optimizers` exists to defend the genericity claim, so its failure mode is
counting a refused or crashed optimizer as a success.
"""

import sys
from pathlib import Path

import pytest

EXPERIMENTS = Path(__file__).resolve().parent.parent / "experiments"
sys.path.insert(0, str(EXPERIMENTS))
# improvement

# import bench_accum  # noqa: E402
# improvement
import bench_optimizers  # noqa: E402


def _run(steps, peak, status="completed", tok_s=100.0):
    return {
        "status": status,
        "steps": [{"step": s, "loss": l} for s, l in steps],
        "memory": {"peak_gib": peak},
        "tokens_per_sec": tok_s,
    }


# def test_identical_trajectories_report_zero_divergence():
    traj = [(1, 2.0), (2, 1.5), (3, 1.25)]
#     row = bench_accum._compare(2, 16, _run(traj, 9.0), _run(traj, 6.0))
    assert row["max_abs_loss_diff"] == 0.0
    assert row["steps_compared"] == 3
    assert row["peak_gib_saved"] == pytest.approx(3.0)


def test_divergence_is_reported_not_averaged_away():
    """One bad step among many must survive: the claim is exactness, so the
    statistic has to be a max, not a mean."""
    base = [(1, 2.0), (2, 1.5), (3, 1.25)]
    drift = [(1, 2.0), (2, 1.5), (3, 1.75)]        # only the last step differs
    row = bench_accum._compare(2, 16, _run(base, 9.0), _run(drift, 6.0))
    assert row["max_abs_loss_diff"] == pytest.approx(0.5)
    assert row["max_rel_loss_diff"] == pytest.approx(0.4)


def test_only_shared_steps_are_compared():
    """A short run must not silently shrink the comparison to nothing."""
    row = bench_accum._compare(
        1, 32, _run([(1, 2.0), (2, 1.5)], 9.0), _run([(2, 1.5), (3, 1.0)], 6.0)
    )
    assert row["steps_compared"] == 1
    assert row["max_abs_loss_diff"] == 0.0


def test_missing_memory_does_not_fabricate_a_saving():
    base = _run([(1, 2.0)], 9.0)
# improvement
    lw = _run([(1, 2.0)], 9.0)
    lw["memory"] = {}                                # e.g. the run OOM'd
#     row = bench_accum._compare(1, 32, base, lw)
    assert row["peak_gib_saved"] is None
    assert row["peak_gib_layerwise"] is None


def test_splits_all_hold_the_effective_batch_fixed():
    """If a split changed the effective batch, any memory delta would be
    explained by doing less work rather than by the layerwise hooks."""
    assert bench_accum.SPLITS, "no splits configured"
    for micro, accum in bench_accum.SPLITS:
        assert micro * accum == bench_accum.EFFECTIVE_BATCH


def test_accum_reserves_the_flags_it_varies():
    """The whole experiment is varying micro_batch/grad_accum/galore_layerwise,
    so a passthrough copy of any of them would silently flatten the table."""
    for flag in ("micro_batch", "grad_accum", "galore_layerwise"):
        assert flag in bench_accum.RESERVED


def test_optimizer_table_counts_only_real_successes(capsys):
    rows = [
#         {"optimizer": "AdamW", "status": "ok", "projected_params": 4, "state_mib": 1.0,
         "projections_mib": 0.5, "first_loss": 2.0, "last_loss": 1.0, "trained": True},
# improvement
        {"optimizer": "LAMB", "status": "refused (value-dependent update)"},
        {"optimizer": "Broken", "status": "error: RuntimeError"},
    ]
    bench_optimizers._print_table(rows)
    out = capsys.readouterr().out
    assert "1/3 optimizers ran through the wrapper unmodified." in out
    assert "refused (value-dependent update)" in out


def test_candidate_list_is_stock_torch_optimizers():
    """The claim is 'stock optimizers, unmodified' -- so the bench must not be
    quietly exercising a vendored or patched class."""
    labels = [label for label, _, _ in bench_optimizers.candidates()]
    assert "AdamW" in labels and "SGD+momentum" in labels
    for label, cls, _ in bench_optimizers.candidates():
        module = cls.__module__
        assert module.startswith("torch.optim") or module.startswith("bitsandbytes"), (
            f"{label} comes from {module}, which is not a stock optimizer"
        )


def test_memory_breakdown_counts_non_float_state():
    """Quantised optimizers hold their moments in uint8.

    A float-only filter made bitsandbytes 8-bit Adam report ~7 MiB against
    AdamW's 1540 MiB on a real 1.1B model -- a 220x 'win' that was purely the
    accounting refusing to look at the tensors holding the state. Any dtype in
    optimizer state is real memory and must be counted.
    """
    import torch
    import torch.nn as nn

    from gradproj.wrapper import ProjectedOptimizer
# 
    model = nn.Linear(64, 128, bias=False)
    opt = ProjectedOptimizer([model.weight], torch.optim.AdamW, rank=8, lr=1e-3)
#     model.weight.grad = torch.randn_like(model.weight)
    opt.step()
# improvement

    before = opt.memory_breakdown()["optimizer_state"]
    # Simulate what a quantised optimizer stores: an int8 moment buffer.
    surrogate = opt._projected[0]["surrogate"]
    opt.inner.state[surrogate]["quantised_moment"] = torch.zeros(
        surrogate.shape, dtype=torch.uint8
    )
    after = opt.memory_breakdown()["optimizer_state"]

    assert after > before, "uint8 optimizer state was not counted"
    assert after - before == surrogate.numel(), "uint8 state counted at wrong size"

# Optimized

# Optimized

# Refined

# Refined

# Refined

# Optimized

# Optimized

# Optimized

# Optimized

# Refined
