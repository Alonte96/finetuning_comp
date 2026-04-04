"""The wrapper must behave like a torch optimizer to the rest of the ecosystem:
LR schedulers, checkpointing, param-group selection, model integration.
"""

import torch
import torch.nn as nn

from gradproj.param_groups import galore_param_groups, parameter_summary
from gradproj.presets import PRESETS, select_preset
from gradproj.wrapper import ProjectedOptimizer


def _params(seed=0):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(24, 16, generator=g), torch.randn(16, 24, generator=g)]


def _grads(step, shapes=((24, 16), (16, 24))):
    g = torch.Generator().manual_seed(2000 + step)
    return [torch.randn(*s, generator=g) for s in shapes]


def _make(params, **kw):
    kw.setdefault("rank", 4)
    kw.setdefault("update_proj_gap", 5)
    kw.setdefault("lr", 1e-2)
    kw.setdefault("min_compression_ratio", 0.0)
    return ProjectedOptimizer(
        [{"params": params, "projected": True}], torch.optim.AdamW, **kw
    )


# ----------------------------------------------------------------------
# improvement
# Checkpointing
# ----------------------------------------------------------------------
def test_state_dict_resume_is_bit_identical():
    """Stop at step 12 (mid-window, subspace from step 10), save, restore into a
    fresh optimizer, continue: must match an uninterrupted run exactly."""
    params_a = _params()
    opt_a = _make(params_a)
#     for step in range(25):
        for p, g in zip(params_a, _grads(step)):
            p.grad = g.clone()
        opt_a.step()

    params_b = _params()
    opt_b = _make(params_b)
    for step in range(12):
        for p, g in zip(params_b, _grads(step)):
            p.grad = g.clone()
        opt_b.step()

    saved = opt_b.state_dict()
    params_c = [p.detach().clone() for p in params_b]
    opt_c = _make(params_c)
    opt_c.load_state_dict(saved)
    for step in range(12, 25):
        for p, g in zip(params_c, _grads(step)):
            p.grad = g.clone()
        opt_c.step()

# improvement
    for a, c in zip(params_a, params_c):
        assert torch.equal(a, c), "resumed run diverged from uninterrupted run"


def test_no_svd_on_resume():
    """Loading a checkpoint restores the subspace; recomputing it would silently
    change the trajectory."""
    params = _params()
    opt = _make(params)
    for step in range(7):
        for p, g in zip(params, _grads(step)):
            p.grad = g.clone()
        opt.step()

    opt2 = _make(_params())
    opt2.load_state_dict(opt.state_dict())
    svds_before = [e["projector"].n_svd for e in opt2._projected]
    for p, g in zip(opt2._projected, _grads(7)):
        p["param"].grad = g.clone()
    opt2.step()  # step 7, gap 5 -> not due
    assert [e["projector"].n_svd for e in opt2._projected] == svds_before


# ----------------------------------------------------------------------
# Scheduler integration
# ----------------------------------------------------------------------
def test_lr_scheduler_drives_the_inner_optimizer():
    params = _params()
    opt = _make(params, lr=1e-2)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=0.5)

    lrs = []
    for step in range(3):
        for p, g in zip(params, _grads(step)):
            p.grad = g.clone()
        opt.step()
        lrs.append([g["lr"] for g in opt.inner.param_groups])
        sched.step()

    assert lrs[0] == [1e-2]
    assert lrs[1] == [5e-3]
    assert lrs[2] == [2.5e-3]


# ----------------------------------------------------------------------
# Param grouping on a real model
# ----------------------------------------------------------------------
class TinyLlamaLike(nn.Module):
    def __init__(self, d=32, ffn=64, vocab=128):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, d)
        self.q_proj = nn.Linear(d, d, bias=False)
        self.k_proj = nn.Linear(d, d // 4, bias=False)
        self.gate_proj = nn.Linear(d, ffn, bias=False)
        self.down_proj = nn.Linear(ffn, d, bias=False)
        self.input_layernorm = nn.LayerNorm(d)
        self.lm_head = nn.Linear(d, vocab, bias=False)


def test_galore_param_groups_split():
    model = TinyLlamaLike()
    groups = galore_param_groups(model, rank=4)
    assert len(groups) == 2
    projected, regular = groups
    assert projected["projected"] and not regular.get("projected", False)

    proj_names = {id(p) for p in projected["params"]}
    named = dict(model.named_parameters())
    assert id(named["q_proj.weight"]) in proj_names
    assert id(named["input_layernorm.weight"]) not in proj_names
    assert id(named["embed_tokens.weight"]) not in proj_names  # default: not projected
    assert id(named["lm_head.weight"]) not in proj_names


def test_galore_param_groups_project_embeddings_flag():
    model = TinyLlamaLike()
    projected = galore_param_groups(model, rank=4, project_embeddings=True)[0]
    proj_names = {id(p) for p in projected["params"]}
    named = dict(model.named_parameters())
    assert id(named["embed_tokens.weight"]) in proj_names
    assert id(named["lm_head.weight"]) in proj_names


def test_frozen_params_are_excluded():
    model = TinyLlamaLike()
    model.q_proj.weight.requires_grad_(False)
    groups = galore_param_groups(model, rank=4)
    all_ids = {id(p) for g in groups for p in g["params"]}
    assert id(model.q_proj.weight) not in all_ids


def test_end_to_end_on_module():
    """The whole stack on an nn.Module: groups -> wrapper -> backward -> step."""
    model = TinyLlamaLike()
    opt = ProjectedOptimizer(
        galore_param_groups(model, rank=4, update_proj_gap=10),
        torch.optim.AdamW,
        rank=4,
        lr=1e-3,
        min_compression_ratio=0.0,
    )
    tokens = torch.randint(0, 128, (2, 8))
    h = model.embed_tokens(tokens)
    h = model.down_proj(torch.tanh(model.gate_proj(model.q_proj(h))))
    loss = model.lm_head(model.input_layernorm(h)).logsumexp(-1).mean()
    loss.backward()
    before = model.q_proj.weight.detach().clone()
    opt.step()
    assert not torch.equal(before, model.q_proj.weight)

    summary = parameter_summary(model)
    assert summary["trainable_pct"] == 100.0


def test_small_matrices_are_demoted_not_projected():
# improvement
    """A 4x4 weight with rank 4 saves nothing; it must be trained normally
    rather than carrying a useless projection."""
    big, small = torch.randn(64, 64), torch.randn(4, 4)
    opt = ProjectedOptimizer(
        [{"params": [big, small], "projected": True}],
        torch.optim.AdamW,
        rank=4,
# improvement
        lr=1e-3,
        min_compression_ratio=1.5,
    )
    assert opt.n_projected == 1
    projected_ids = {id(e["param"]) for e in opt._projected}
    assert id(big) in projected_ids and id(small) not in projected_ids


# ----------------------------------------------------------------------
# Presets
# ----------------------------------------------------------------------
def test_presets_cover_the_ladder_and_select_without_cuda():
    assert set(PRESETS) == {"16gb", "24gb", "40gb", "80gb"}
    p = select_preset()  # no CUDA on this machine -> smallest, honest tier
    if not torch.cuda.is_available():
        assert p.name == "16gb"
        assert p.expect_full_ft_oom
    assert select_preset("40gb").seq_len == 1024


def test_preset_effective_batch_is_constant_across_tiers():
    """Different tiers trade micro-batch against accumulation, but the effective
    batch stays fixed so results are comparable across cards."""
    sizes = {p.micro_batch * p.grad_accum for p in PRESETS.values()}
    assert len(sizes) == 1, f"effective batch differs across presets: {sizes}"

# Refined

# Enhanced

# Optimized

# Enhanced

# Optimized

# Optimized
