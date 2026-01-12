"""Memory accounting must be right, or the whole benchmark is decoration.

Two things are checked: that the analytic prediction matches what the optimizer
actually allocates, and that the TinyLlama spec matches its published config.
"""

import pytest
import torch

from gradproj.memory import (
    GIB, ParamSpec, MemoryProbe, predict, spec_from_model, tinyllama_1b_spec,
)
from gradproj.wrapper import ProjectedOptimizer


class Tiny(torch.nn.Module):
    """Stand-in with Llama-style module names so target matching applies."""

    def __init__(self, d=32, h=64):
        super().__init__()
        self.q_proj = torch.nn.Linear(d, d, bias=False)
        self.gate_proj = torch.nn.Linear(d, h, bias=False)
        self.down_proj = torch.nn.Linear(h, d, bias=False)
        self.norm = torch.nn.LayerNorm(d)


def test_tinyllama_spec_matches_published_config():
    """1.1B, and the exact composition the memory table depends on."""
    spec = tinyllama_1b_spec()
    total = sum(s.numel for s in spec)
    assert total == 1_100_048_384
    assert 1.09e9 < total < 1.11e9

    embed = sum(s.numel for s in spec if "embed_tokens" in s.name or "lm_head" in s.name)
    assert embed == 2 * 32000 * 2048  # untied embeddings

    # Grouped-query attention: 4 KV heads x 64 head_dim = 256 rows, not 2048.
    kv = next(s for s in spec if s.name.endswith("k_proj.weight"))
    assert kv.shape == (256, 2048)


def test_predicted_optimizer_state_matches_reality():
    """The claim under test: projected Adam state is the size we say it is."""
    model = Tiny()
    params = [p for p in model.parameters() if p.ndim == 2]
    opt = ProjectedOptimizer(params, torch.optim.AdamW, rank=8, lr=1e-3)

    for p in params:
        p.grad = torch.randn_like(p)
    opt.step()

    spec = [ParamSpec(n, tuple(p.shape)) for n, p in model.named_parameters() if p.ndim == 2]
    est = predict(spec, "galore", rank=8, param_dtype=torch.float32, proj_dtype=torch.float32,
                  target_modules=("q_proj", "gate_proj", "down_proj"))
    measured = opt.memory_breakdown()

    assert measured["optimizer_state"] == est.optimizer_state
    assert measured["projections"] + measured["surrogates"] == est.projections


def test_galore_state_is_much_smaller_than_full():
    spec = tinyllama_1b_spec()
    full = predict(spec, "full")
    galore = predict(spec, "galore", rank=128)
    assert galore.optimizer_state < full.optimizer_state / 4
    assert galore.trainable_params == full.trainable_params  # every weight still trains


def test_projecting_embeddings_cuts_the_dominant_remaining_state():
    """Embedding + LM head are only 12% of params but dominate GaLore's optimizer
    state once the attention/MLP weights are projected."""
    spec = tinyllama_1b_spec()
    without = predict(spec, "galore", rank=128, project_embeddings=False)
    with_embed = predict(spec, "galore", rank=128, project_embeddings=True)
    assert with_embed.optimizer_state < without.optimizer_state / 2


def test_layerwise_collapses_gradient_memory():
    spec = tinyllama_1b_spec()
    standard = predict(spec, "galore", rank=128)
    layerwise = predict(spec, "galore", rank=128, layerwise=True)
    assert layerwise.grads < standard.grads / 10
    assert layerwise.total < standard.total


def test_lora_trains_a_small_fraction_galore_trains_everything():
    spec = tinyllama_1b_spec()
    lora = predict(spec, "lora", lora_rank=16)
    galore = predict(spec, "galore", rank=128)
    assert lora.trainable_pct < 2.0
    assert galore.trainable_pct == 100.0


def test_full_finetune_matches_hand_arithmetic():
    """1.1B params, fp32 weights + fp32 grads + two fp32 Adam moments = 4x params."""
    spec = tinyllama_1b_spec()
    est = predict(spec, "full")
    n = 1_100_048_384
    assert est.params == n * 4
    assert est.grads == n * 4
    assert est.optimizer_state == n * 4 * 2
    assert est.total / GIB == pytest.approx(16.39, abs=0.02)


def test_lora_adapter_size_matches_hand_arithmetic():
    """r * (m + n) per targeted weight."""
    spec = [ParamSpec("layers.0.q_proj.weight", (2048, 2048))]
    est = predict(spec, "lora", lora_rank=16, target_modules=("q_proj",))
    assert est.trainable_params == 16 * (2048 + 2048)


def test_spec_from_model_roundtrips():
    model = Tiny()
    spec = spec_from_model(model)
    assert sum(s.numel for s in spec) == sum(p.numel() for p in model.parameters())


def test_memory_probe_is_honest_about_precision():
    """On non-CUDA devices the probe must not claim to be measuring VRAM."""
    probe = MemoryProbe("cpu")
    assert probe.exact is False
    assert "RSS" in probe.backend

    with probe:
        blob = torch.zeros(4_000_000)  # ~16 MB
        del blob
    assert probe.peak_bytes() >= 0
    assert probe.report()["exact"] is False


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_memory_probe_is_exact_on_cuda():
    probe = MemoryProbe("cuda")
    assert probe.exact
    with probe:
        blob = torch.zeros(4_000_000, device="cuda")  # 16 MB
        del blob
    assert probe.peak_bytes() >= 16_000_000

# Enhanced
