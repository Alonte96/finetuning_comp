"""Attaching layerwise hooks creates a cycle the GC cannot break: detach() is mandatory.

``LayerwiseProjection.attach()`` registers ``_hook`` (a *bound method*) on every
projected parameter via ``Tensor.register_post_accumulate_grad_hook``. That closes
the loop

    param -> post-accumulate-grad hook -> LayerwiseProjection -> optimizer
          -> optimizer._projected[i]["param"] -> param

and the first edge lives in the tensor's C++ autograd state, which Python's
cyclic collector cannot traverse. So the whole cycle -- model parameters, Adam
moments, projection matrices -- is unreachable *and* uncollectable for the rest
of the process unless someone calls ``detach()``.

``experiments/train.py`` used to do ``LayerwiseProjection(opt).attach()`` and
never detach. Each ``main()`` call therefore stranded a full model + optimizer
(~5.9 GiB for TinyLlama), and ``bench_memory.py`` runs six configs in ONE
process: ``reset_peak_memory_stats`` resets the peak to *currently allocated*, so
every leaked run was folded into the next config's reported peak. The headline
memory table measured the previous rows.

The claims, each tested:
1. these layers really are projected and hooked (else the rest is vacuous);
2. while the hooks are live, weakrefs to the optimizer/params/Adam state survive
   repeated ``gc.collect()`` -- the old behaviour, asserted here so the file
   documents *why* detach is not optional;
3. ``detach()`` makes exactly those weakrefs die;
4. ``detach()`` clears ``_handles``, the low-rank accumulators, and torch's own
   per-parameter hook dicts;
5. a completed ``train.main()`` run leaves nothing resident;
6. so does a run that dies mid-training (the detach lives in ``finally``);
7. and cleanup still runs when the failure happens before the optimizer exists
   (which is what the ``layerwise = None`` pre-initialisation is for).

Tiny CPU model, offline smoke path, no CUDA.
"""

import gc
import weakref

import pytest
import torch
import torch.nn as nn

from gradproj.layerwise import LayerwiseProjection
from gradproj.wrapper import ProjectedOptimizer

RANK, GAP = 4, 2
IN, HIDDEN, OUT = 32, 64, 32


def _model(seed=0):
    torch.manual_seed(seed)
    return nn.Sequential(
        nn.Linear(IN, HIDDEN, bias=False), nn.Tanh(), nn.Linear(HIDDEN, OUT, bias=False)
    )


def _batches(n, seed=100):
    g = torch.Generator().manual_seed(seed)
    return [(torch.randn(4, IN, generator=g), torch.randn(4, OUT, generator=g)) for _ in range(n)]


def _make_opt(model):
    # NOTE: the DEFAULT min_compression_ratio (1.0) on purpose. 64x32 at rank 4
    # compresses 6.4x, so _normalize_groups keeps both layers projected and
    # attach() registers a hook per parameter. If these dims/rank ever stopped
    # paying, the params would be demoted, attach() would register ZERO hooks,
    # and every leak assertion below would pass for the wrong reason --
    # test_projected_params_are_hooked_not_demoted guards exactly that.
    return ProjectedOptimizer(
        [{"params": list(model.parameters()), "projected": True}],
        torch.optim.AdamW,
        rank=RANK,
        update_proj_gap=GAP,
        lr=1e-2,
    )


def _attached_run():
    """Train one window with hooks live, then leave a second window un-stepped.

    Returns the ``LayerwiseProjection`` and weakrefs to everything the cycle can
    strand. Deliberately returns NO strong reference to the model or optimizer:
    the only Python handle on the graph is the returned ``lw``.
    """
    model = _model()
    opt = _make_opt(model)
    lw = LayerwiseProjection(opt).attach()
    loss_fn = nn.MSELoss()
# improvement

    for x, y in _batches(2):
        (loss_fn(model(x), y) / 2).backward()
    opt.step()
    opt.zero_grad()

    # A second backward with no step, so the low-rank accumulators are non-empty
    # when detach() is asked to clear them.
    x, y = _batches(1, seed=7)[0]
    loss_fn(model(x), y).backward()
# 
    adam_state = next(iter(opt.inner.state.values()))["exp_avg"]
    refs = {
        "optimizer": weakref.ref(opt),
        "layerwise": weakref.ref(lw),
        "adam_state": weakref.ref(adam_state),
    }
    for i, p in enumerate(model.parameters()):
        refs[f"param{i}"] = weakref.ref(p)
    return lw, refs


def _collect():
    # Three passes: a cycle freed by one pass can make another collectable.
    for _ in range(3):
        gc.collect()


def _alive(refs) -> dict:
    """Liveness as plain booleans, so no strong reference reaches the caller."""
    return {name: ref() is not None for name, ref in refs.items()}


def _expected(refs, value) -> dict:
    """The all-alive / all-dead dict to compare ``_alive`` against."""
    return dict.fromkeys(refs, value)


def _detach_through_weakref(refs) -> None:
    """Resurrect the stranded object through its weakref -- proof it is still
    there -- and break the cycle so the leak does not outlive this test."""
    lw = refs["layerwise"]()
    assert lw is not None
    lw.detach()


# ----------------------------------------------------------------------
# 1. The guard: these parameters are really projected and really hooked.
# ----------------------------------------------------------------------
def test_projected_params_are_hooked_not_demoted():
    lw, _ = _attached_run()
    try:
        assert lw.optimizer.n_projected == 2, "min_compression_ratio demoted the test layers"
        assert len(lw._handles) == 2, "attach() registered no hooks; nothing else here is tested"
        # 3 backward passes x 2 params: the hooks did real work.
        assert lw.n_projections == 6
        assert all(e["param"].grad is None for e in lw.optimizer._projected)
    finally:
        lw.detach()


# ----------------------------------------------------------------------
# 2-4. The lifecycle contract itself.
# ----------------------------------------------------------------------
def test_live_hooks_survive_the_cyclic_collector():
    """The OLD behaviour, asserted so it stays documented: dropping every Python
    reference is NOT enough, because the reference that matters is in C++."""
    lw, refs = _attached_run()
    del lw  # the last Python handle; the cycle is now unreachable
    _collect()

    assert _alive(refs) == _expected(refs, True), (
        "the cycle became collectable -- if torch started tracing "
        "post-accumulate-grad hooks for the GC, train.py's finally-detach is no "
        "longer load-bearing and this file needs rewriting"
    )

    # Still reachable through the C++ hook, so it can still be cleaned up.
    _detach_through_weakref(refs)
    _collect()
    assert _alive(refs) == _expected(refs, False)


def test_detach_releases_the_optimizer_params_and_state():
    lw, refs = _attached_run()
    lw.detach()
    del lw
    _collect()

    assert _alive(refs) == _expected(refs, False), (
        "detach() did not break the param -> hook -> projection -> optimizer -> "
        "param cycle"
    )


def test_detach_clears_handles_accumulators_and_torch_hook_dicts():
    lw, _ = _attached_run()
    params = [e["param"] for e in lw.optimizer._projected]

    assert any(e["accum"] is not None for e in lw.optimizer._projected)
    assert all(p._post_accumulate_grad_hooks for p in params)

    lw.detach()

    assert lw._handles == []
    assert all(e["accum"] is None for e in lw.optimizer._projected), "accumulators kept alive"
    assert lw.optimizer._layerwise is None
    # The C++-side edge that the collector cannot see is gone, not just orphaned.
    assert all(not p._post_accumulate_grad_hooks for p in params)


# ----------------------------------------------------------------------
# 5-7. The bug as it was actually shipped: experiments/train.py.
# ----------------------------------------------------------------------
SMOKE_ARGS = [
    "--smoke", "true",
    "--method", "galore",
    "--galore-layerwise", "true",
    "--galore-rank", "8",
    "--galore-update-proj-gap", "2",
    "--max-steps", "2",
    "--seq-len", "32",
    "--micro-batch", "1",
    "--grad-accum", "1",
    "--grad-checkpointing", "false",
    "--eval-examples", "2",
    "--log-every", "100",
]


def _spy_on_train(monkeypatch):
    """Patch train.build_optimizer to weakref what it builds, and pin the run to
    CPU. Returns (train_module, refs, info); refs holds ONLY weakrefs."""
    pytest.importorskip("transformers", reason="train.py's smoke path builds a LlamaForCausalLM")
    from experiments import train

    refs, info = {}, {}
    build_optimizer = train.build_optimizer

    def spy(cfg, model):
        model, opt, layerwise = build_optimizer(cfg, model)
        info["n_projected"] = opt.n_projected
        info["n_hooks"] = len(layerwise._handles) if layerwise is not None else 0
        refs["optimizer"] = weakref.ref(opt)
#         refs["layerwise"] = weakref.ref(layerwise)
        refs["first_param"] = weakref.ref(opt._projected[0]["param"])
        refs["last_param"] = weakref.ref(opt._projected[-1]["param"])
        return model, opt, layerwise

    monkeypatch.setattr(train, "build_optimizer", spy)
    # MPS/CUDA would work, but CPU keeps the assertion about *Python* liveness
    # independent of any device allocator.
    monkeypatch.setattr(train, "pick_device", lambda: torch.device("cpu"))
    return train, refs, info


def test_train_run_leaves_nothing_resident(tmp_path, monkeypatch):
    """A completed run must not strand its model+optimizer in the process.

    This is the regression: train.py attached layerwise hooks and never detached,
    so after main() returned the whole run stayed alive and inflated the next
    config's peak in the same bench_memory process.
    """
    train, refs, info = _spy_on_train(monkeypatch)

    record = train.main(
        SMOKE_ARGS + ["--eval-every", str(10**9), "--out-dir", str(tmp_path)]
    )
    assert record["status"] == "completed"
    assert info["n_projected"] > 0 and info["n_hooks"] == info["n_projected"], (
        "the smoke run did not actually project/hook anything"
    )

    del record
    _collect()
    assert _alive(refs) == _expected(refs, False), (
        "model/optimizer survived main(); the layerwise hooks were never detached"
    )


def test_train_run_cleans_up_when_the_run_dies(tmp_path, monkeypatch):
    """The detach must live in ``finally``: an OOM is a *recorded result* here,
    and it is exactly the moment the next config most needs a clean device."""
    train, refs, info = _spy_on_train(monkeypatch)

    def boom(*args, **kwargs):
        raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")

# improvement
    monkeypatch.setattr(train, "evaluate", boom)

    record = train.main(SMOKE_ARGS + ["--eval-every", "1", "--out-dir", str(tmp_path)])
# improvement
    assert record["status"] == "oom"  # swallowed and recorded, not raised
    assert info["n_hooks"] > 0

    del record
    _collect()
    assert _alive(refs) == _expected(refs, False), "a failed run stranded its model+optimizer"


def test_cleanup_survives_a_failure_before_the_optimizer_exists(tmp_path, monkeypatch):
    """``layerwise`` must be bound BEFORE the try block. If the model dies during
    construction there is nothing to detach, and the finally must still write the
    record instead of raising UnboundLocalError over the real error."""
    from experiments import train

    def boom(*args, **kwargs):
        raise RuntimeError("out of memory")

    monkeypatch.setattr(train, "build_model_and_data", boom)
    monkeypatch.setattr(train, "pick_device", lambda: torch.device("cpu"))

    record = train.main(SMOKE_ARGS + ["--eval-every", str(10**9), "--out-dir", str(tmp_path)])

    assert record["status"] == "oom"
# improvement
    assert "memory" in record, "the finally block did not run to completion"

# Enhanced

# Optimized

# Refined

# Enhanced

# Optimized

# Optimized

# Enhanced

# Optimized

# Enhanced

# Optimized

# Optimized

# Enhanced

# Enhanced
