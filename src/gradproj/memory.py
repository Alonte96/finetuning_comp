"""Memory accounting: analytic prediction, and measurement of the real thing.

Two independent views, because each covers the other's weakness:

* :func:`predict` computes what the numbers *should* be from parameter shapes
  alone. It runs on any laptop with no GPU and isolates the claim being made
  (optimizer state size) from activation memory and allocator noise.
* :class:`MemoryProbe` measures what actually happened on the device.

The benchmark reports both side by side. Predictions that don't match
measurement mean one of them is wrong, and that is worth knowing.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

import torch

from gradproj.param_groups import DEFAULT_TARGET_MODULES, EMBEDDING_MODULES
from gradproj.projector import GaLoreProjector

GIB = 1024**3

__all__ = [
    "ParamSpec", "MemoryEstimate", "predict", "spec_from_model",
    "llama_spec", "tinyllama_1b_spec", "MemoryProbe",
]

# 
def _dtype_bytes(dtype: torch.dtype) -> int:
    return torch.empty((), dtype=dtype).element_size()


@dataclass(frozen=True)
class ParamSpec:
    """A parameter's identity and shape -- everything memory accounting needs."""

    name: str
    shape: tuple[int, ...]

    @property
    def numel(self) -> int:
        n = 1
        for d in self.shape:
            n *= d
        return n

    def is_target(self, target_modules: tuple[str, ...]) -> bool:
        module_path = self.name.rsplit(".", 1)[0]
        return any(module_path.endswith(t) for t in target_modules)


@dataclass
class MemoryEstimate:
    """Static (non-activation) training memory, in bytes."""

    method: str
    params: int = 0
    grads: int = 0
    optimizer_state: int = 0
    projections: int = 0
    total_params: int = 0
    trainable_params: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def total(self) -> int:
        return self.params + self.grads + self.optimizer_state + self.projections

    @property
    def trainable_pct(self) -> float:
        return 100.0 * self.trainable_params / self.total_params if self.total_params else 0.0

    def as_gib(self) -> dict[str, float]:
        return {
            "params": self.params / GIB,
            "grads": self.grads / GIB,
            "optimizer_state": self.optimizer_state / GIB,
            "projections": self.projections / GIB,
            "total": self.total / GIB,
        }


def predict(
    spec: list[ParamSpec],
    method: str,
    *,
    rank: int = 128,
    lora_rank: int = 128,
    param_dtype: torch.dtype = torch.float32,
    grad_dtype: torch.dtype | None = None,
    state_dtype: torch.dtype = torch.float32,
    proj_dtype: torch.dtype = torch.bfloat16,
    proj_type: str = "std",
    layerwise: bool = False,
    project_embeddings: bool = False,
    target_modules: tuple[str, ...] = DEFAULT_TARGET_MODULES,
    n_optimizer_states: int = 2,
) -> MemoryEstimate:
    """Predict static training memory for ``method`` in ``{full, lora, galore}``.

    Args:
        n_optimizer_states: 2 for Adam-family (m and v), 1 for SGD-momentum, 0 for SGD.
        layerwise: GaLore only. With per-layer updates each gradient is consumed and
            freed as it is produced, so peak gradient memory is the *largest single*
            parameter rather than the sum.
    """
    p_bytes = _dtype_bytes(param_dtype)
    g_bytes = _dtype_bytes(grad_dtype or param_dtype)
    s_bytes = _dtype_bytes(state_dtype)
    pr_bytes = _dtype_bytes(proj_dtype)

    total_params = sum(s.numel for s in spec)
    est = MemoryEstimate(method=method, total_params=total_params)
    est.params = total_params * p_bytes

    if method == "full":
        est.trainable_params = total_params
        est.grads = total_params * g_bytes
        est.optimizer_state = total_params * s_bytes * n_optimizer_states

    elif method == "lora":
        # Base weights stay frozen: no gradients, no optimizer state.
        # A LoRA on a (m, n) weight adds B (m, r) and A (r, n).
        adapter = sum(
            lora_rank * (s.shape[0] + s.shape[1])
            for s in spec
            if len(s.shape) == 2 and s.is_target(target_modules)
        )
        est.trainable_params = adapter
        est.params += adapter * p_bytes
        est.grads = adapter * g_bytes
        est.optimizer_state = adapter * s_bytes * n_optimizer_states
        est.notes.append(f"LoRA r={lora_rank} on {len(target_modules)} module types")

    elif method == "galore":
        targets = tuple(target_modules) + (EMBEDDING_MODULES if project_embeddings else ())
        projector = GaLoreProjector(rank=rank, proj_type=proj_type)
        est.trainable_params = total_params

        low_elems = 0
        proj_elems = 0
        full_state_elems = 0
        for s in spec:
            if len(s.shape) == 2 and s.is_target(targets):
                shape = (s.shape[0], s.shape[1])
                low = projector.low_rank_shape(shape)
                low_elems += low[0] * low[1]
                proj_elems += projector.ortho_numel(shape)
            else:
                full_state_elems += s.numel

        est.optimizer_state = (low_elems + full_state_elems) * s_bytes * n_optimizer_states
        est.projections = proj_elems * pr_bytes
        # The zeroed surrogate parameters are one extra low-rank copy.
        est.projections += low_elems * p_bytes

        if layerwise:
            est.grads = max(s.numel for s in spec) * g_bytes
            est.notes.append("layerwise: peak grad = largest single parameter")
        else:
            est.grads = total_params * g_bytes

        unprojected = full_state_elems * s_bytes * n_optimizer_states
        est.notes.append(
            f"rank={rank}; {unprojected / GIB:.2f} GiB of optimizer state is on "
            f"unprojected params ({full_state_elems / 1e6:.1f}M)"
        )
    else:
        raise ValueError(f"unknown method {method!r}; expected full, lora or galore")

    return est


# ----------------------------------------------------------------------
# Model specs
# ----------------------------------------------------------------------
def spec_from_model(model) -> list[ParamSpec]:
    return [ParamSpec(name, tuple(p.shape)) for name, p in model.named_parameters()]


def llama_spec(
    *,
    hidden_size: int,
    intermediate_size: int,
    num_hidden_layers: int,
    num_attention_heads: int,
    num_key_value_heads: int,
    vocab_size: int,
    tie_word_embeddings: bool = False,
) -> list[ParamSpec]:
    """Parameter shapes of a Llama-family model, without downloading weights.

    Lets the memory table be computed offline, and lets the predictions be
    checked against a real model's ``named_parameters()``.
    """
    head_dim = hidden_size // num_attention_heads
    kv_dim = num_key_value_heads * head_dim
    spec = [ParamSpec("model.embed_tokens.weight", (vocab_size, hidden_size))]

    for i in range(num_hidden_layers):
        p = f"model.layers.{i}"
        spec += [
            ParamSpec(f"{p}.self_attn.q_proj.weight", (hidden_size, hidden_size)),
            ParamSpec(f"{p}.self_attn.k_proj.weight", (kv_dim, hidden_size)),
            ParamSpec(f"{p}.self_attn.v_proj.weight", (kv_dim, hidden_size)),
            ParamSpec(f"{p}.self_attn.o_proj.weight", (hidden_size, hidden_size)),
            ParamSpec(f"{p}.mlp.gate_proj.weight", (intermediate_size, hidden_size)),
            ParamSpec(f"{p}.mlp.up_proj.weight", (intermediate_size, hidden_size)),
            ParamSpec(f"{p}.mlp.down_proj.weight", (hidden_size, intermediate_size)),
            ParamSpec(f"{p}.input_layernorm.weight", (hidden_size,)),
            ParamSpec(f"{p}.post_attention_layernorm.weight", (hidden_size,)),
        ]

    spec.append(ParamSpec("model.norm.weight", (hidden_size,)))
    if not tie_word_embeddings:
        spec.append(ParamSpec("lm_head.weight", (vocab_size, hidden_size)))
    return spec


def tinyllama_1b_spec() -> list[ParamSpec]:
    """TinyLlama-1.1B-Chat-v1.0, from its published config.json."""
    return llama_spec(
        hidden_size=2048,
        intermediate_size=5632,
        num_hidden_layers=22,
        num_attention_heads=32,
        num_key_value_heads=4,
        vocab_size=32000,
        tie_word_embeddings=False,
    )


# improvement
# ----------------------------------------------------------------------
# Measurement
# ----------------------------------------------------------------------
class MemoryProbe:
    """Peak device-memory measurement, honest about its own precision.

    On CUDA this is exact and resettable. On MPS and CPU it is a sampler, and
    ``exact`` is False -- the benchmark labels those numbers accordingly rather
    than passing them off as VRAM measurements.
    """

    def __init__(self, device: torch.device | str, sample_interval: float = 0.005):
        self.device = torch.device(device)
        self.sample_interval = sample_interval
        self._peak = 0
        self._baseline = 0
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    @property
    def exact(self) -> bool:
        return self.device.type == "cuda"

    @property
    def backend(self) -> str:
        return {"cuda": "torch.cuda.max_memory_allocated",
                "mps": "torch.mps.current_allocated_memory (sampled)",
                "cpu": "process RSS (sampled)"}.get(self.device.type, "unknown")

    def _current(self) -> int:
        if self.device.type == "cuda":
            return torch.cuda.memory_allocated(self.device)
        if self.device.type == "mps":
            return torch.mps.current_allocated_memory()
        import resource
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # macOS reports bytes, Linux reports kilobytes.
        import sys
        return rss if sys.platform == "darwin" else rss * 1024

    def reset(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
            return
        self._baseline = self._current()
        self._peak = 0

    def __enter__(self) -> MemoryProbe:
        self.reset()
        if not self.exact:
            self._stop.clear()
            self._thread = threading.Thread(target=self._sample_loop, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        if self._thread is not None:
            self._stop.set()
            self._thread.join(timeout=1.0)
            self._thread = None

    def _sample_loop(self) -> None:
        while not self._stop.is_set():
            self._peak = max(self._peak, self._current())
            time.sleep(self.sample_interval)

    def peak_bytes(self) -> int:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            return torch.cuda.max_memory_allocated(self.device)
        self._peak = max(self._peak, self._current())
        return max(0, self._peak - self._baseline)

    def reserved_bytes(self) -> int:
        """What the allocator holds -- i.e. roughly what nvidia-smi shows."""
        if self.device.type == "cuda":
            return torch.cuda.max_memory_reserved(self.device)
        if self.device.type == "mps":
            return torch.mps.driver_allocated_memory()
        return self.peak_bytes()

    def report(self) -> dict:
        return {
            "device": str(self.device),
            "backend": self.backend,
            "exact": self.exact,
            "peak_bytes": self.peak_bytes(),
            "peak_gib": self.peak_bytes() / GIB,
            "reserved_bytes": self.reserved_bytes(),
            "reserved_gib": self.reserved_bytes() / GIB,
        }


def _print_prediction_table() -> None:
    """The README's predicted-memory table: python -m gradproj.memory"""
    spec = tinyllama_1b_spec()
    rows = [
        ("Full fine-tune", "full", {}),
        ("LoRA r=16", "lora", {"lora_rank": 16}),
        ("LoRA r=128", "lora", {"lora_rank": 128}),
        ("GaLore r=128", "galore", {"rank": 128}),
        ("+ layerwise", "galore", {"rank": 128, "layerwise": True}),
        ("+ layerwise + proj. embeddings", "galore",
         {"rank": 128, "layerwise": True, "project_embeddings": True}),
    ]
    print(f"TinyLlama-1.1B ({sum(s.numel for s in spec) / 1e9:.3f}B params), "
          "fp32 masters, AdamW, static memory in GiB\n")
    print(f"{'method':<32}{'params':>8}{'grads':>8}{'opt':>8}{'proj':>8}{'TOTAL':>9}{'train%':>9}")
    for name, method, kw in rows:
        e = predict(spec, method, **kw)
        g = e.as_gib()
        print(f"{name:<32}{g['params']:>8.2f}{g['grads']:>8.2f}"
              f"{g['optimizer_state']:>8.2f}{g['projections']:>8.2f}"
              f"{g['total']:>9.2f}{e.trainable_pct:>8.1f}%")


if __name__ == "__main__":
    _print_prediction_table()

# Optimized

# Refined
