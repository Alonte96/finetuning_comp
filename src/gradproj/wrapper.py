"""ProjectedOptimizer: low-rank gradient projection around *any* torch optimizer.

The reference GaLore implementation ships hand-modified copies of AdamW,
Adafactor and 8-bit Adam, because the projection has to happen inside the step.
That is a maintenance dead end: every new optimizer needs a new fork.

This wrapper avoids forking with one observation. An optimizer step is a map

    (state, grad) -> delta,   applied as  p += delta

and for GaLore we want ``p += project_back(delta_low)`` where ``delta_low`` came
from running the optimizer on the projected gradient. So: hand the inner
optimizer a **zeroed surrogate parameter** of low-rank shape. After its step the
surrogate holds exactly ``delta_low`` (it started at zero), which we project back
and add to the real weight. The inner optimizer allocates its state at the
surrogate's shape, which is the entire memory win, and it never needs to know
projection exists.

This is exact, not approximate: ``project_back`` is linear and the step size is a
scalar, so ``project_back(-s * g) == -s * project_back(g)``. ``tests/
test_equivalence.py`` asserts it against a port of the reference optimizer.

The one requirement on the inner optimizer is that its delta must not depend on
the parameter's *value* (the surrogate is always zero). That covers Adam(W), SGD,
RMSprop, Adagrad, Lion and 8-bit Adam, but not LAMB-style trust ratios or
Adafactor's ``scale_parameter``. We check this empirically at construction rather
than trusting a blocklist -- see ``_assert_value_independent``.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch.optim import Optimizer

from gradproj.projector import GaLoreProjector

__all__ = ["ProjectedOptimizer", "ValueDependentOptimizerError"]


class ValueDependentOptimizerError(TypeError):
#     """Raised when the inner optimizer's update depends on the parameter value."""


class ProjectedOptimizer(Optimizer):
    """Wraps ``optimizer_cls`` so that flagged parameters train in a low-rank
    gradient subspace while every weight still receives a full-rank update.

    Args:
        params: An iterable of parameters, or a list of param-group dicts. A group
            opts into projection with ``{"projected": True}`` and may override
            ``rank`` / ``update_proj_gap`` / ``scale`` / ``proj_type``. A bare
            iterable auto-selects every eligible 2D parameter.
        optimizer_cls: Any ``torch.optim.Optimizer`` subclass (or factory).
        rank: Default projection rank.
        update_proj_gap: Steps between SVD recomputations of the subspace.
        scale: ``alpha`` multiplier on the reconstructed update.
        proj_type: See :class:`~gradproj.projector.GaLoreProjector`.
        proj_dtype: Storage dtype for projection matrices (bf16 halves their cost).
        min_compression_ratio: Parameters whose projection would save less than
            this factor of optimizer memory are trained normally instead. Prevents
#             "projections" that cost more than they save on small matrices.
        weight_decay: Decoupled weight decay applied to the **full-rank** parameter
            after the update, matching the reference. Forced to 0 inside the inner
            optimizer for projected params, where it would silently no-op.
        allow_value_dependent: Skip the safety probe. Only if you know the inner
            optimizer's update is value-independent and the probe is wrong.
        **optimizer_kwargs: Forwarded to ``optimizer_cls`` (``lr``, ``betas``, ...).
    """

    def __init__(
        self,
        params,
        optimizer_cls: type[Optimizer] = torch.optim.AdamW,
        *,
        rank: int = 128,
        update_proj_gap: int = 200,
        scale: float = 1.0,
        proj_type: str = "std",
        proj_dtype: torch.dtype | None = None,
        min_compression_ratio: float = 1.0,
        weight_decay: float = 0.0,
        allow_value_dependent: bool = False,
        **optimizer_kwargs,
    ):
        self._defaults = dict(
            rank=rank,
            update_proj_gap=update_proj_gap,
            scale=scale,
            proj_type=proj_type,
        )
        self.proj_dtype = proj_dtype
        self.min_compression_ratio = min_compression_ratio
        self.weight_decay = weight_decay

        groups = _normalize_groups(params, self._defaults, min_compression_ratio, proj_dtype)

        # Bookkeeping for projected params, in a stable order for checkpointing.
        self._projected: list[dict] = []
        inner_groups: list[dict] = []

        for group in groups:
            cfg = {k: group.get(k, v) for k, v in self._defaults.items()}
            group_kwargs = {
                k: v for k, v in group.items()
                if k not in ("params", "projected", "rank", "update_proj_gap", "scale", "proj_type")
            }

            if not group.get("projected", False):
                # ``weight_decay`` is a named argument of this wrapper, so it never
                # reaches ``optimizer_kwargs``. Without re-supplying it here the
                # inner optimizer falls back to its OWN default (0.01 for AdamW)
                # on unprojected params while projected params use ours -- a
                # regularisation asymmetry inside a single model.
                inner_groups.append(
                    {"weight_decay": self.weight_decay, **group_kwargs,
                     "params": list(group["params"])}
                )
                continue

            surrogates = []
            for p in group["params"]:
                projector = GaLoreProjector(
                    rank=cfg["rank"],
                    update_proj_gap=cfg["update_proj_gap"],
                    scale=cfg["scale"],
                    proj_type=cfg["proj_type"],
                    proj_dtype=proj_dtype,
                )
                low_shape = projector.low_rank_shape(tuple(p.shape))
                surrogate = torch.zeros(low_shape, dtype=p.dtype, device=p.device)
                self._projected.append(
                    {"param": p, "surrogate": surrogate, "projector": projector, "step": 0}
                )
                surrogates.append(surrogate)

            # Weight decay on a permanently-zero surrogate is a no-op, so we strip
            # it here and re-apply it to the real parameter after the step.
            inner_groups.append({**group_kwargs, "params": surrogates, "weight_decay": 0.0})

        if not inner_groups or not any(g["params"] for g in inner_groups):
            raise ValueError("ProjectedOptimizer got no parameters to optimize")

        if not allow_value_dependent:
            _assert_value_independent(optimizer_cls, optimizer_kwargs)

        self.inner = optimizer_cls(inner_groups, **optimizer_kwargs)

        # Initialise the torch Optimizer machinery (hook dicts, profiling names)
        # over the *real* parameters, then point param_groups at the inner
        # optimizer's list -- the same object, so LR schedulers that mutate
        # group["lr"] drive the inner optimizer directly.
        real_params = [entry["param"] for entry in self._projected]
        real_params += [p for g in groups if not g.get("projected", False) for p in g["params"]]
        super().__init__([{"params": real_params}], {"lr": optimizer_kwargs.get("lr", 1e-3)})
        self.param_groups = self.inner.param_groups
        self.state = self.inner.state

    # ------------------------------------------------------------------
    # Optimizer interface
    # ------------------------------------------------------------------
    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None

        for entry in self._projected:
            p, surrogate = entry["param"], entry["surrogate"]
            accum = entry.get("accum")
#             if accum is not None:
                # Layerwise mode: the backward hooks already projected each
                # micro-batch's gradient and freed it; the sum is exactly the
                # projection of the accumulated gradient (projection is linear).
                surrogate.grad = accum
                entry["accum"] = None
            elif p.grad is not None:
                # The projector is called with the pre-increment step count,
                # matching the reference: the SVD fires on steps 0, gap, 2*gap...
                surrogate.grad = entry["projector"].project(p.grad, entry["step"])
            else:
                surrogate.grad = None
                continue
            # Zeroed so that after the inner step the surrogate *is* the delta.
            surrogate.zero_()

        self.inner.step()

        for entry in self._projected:
            p, surrogate = entry["param"], entry["surrogate"]
            if surrogate.grad is None:
                continue
            p.add_(entry["projector"].project_back(surrogate))
            entry["step"] += 1

        # Decoupled weight decay on the full-rank parameter, after the update.
        if self.weight_decay > 0.0:
            for group, entry in zip(self._projected_groups(), self._projected):
                if entry["surrogate"].grad is not None:
                    entry["param"].add_(
                        entry["param"], alpha=-group["lr"] * self.weight_decay
                    )

        return loss

    def zero_grad(self, set_to_none: bool = True):
        self.inner.zero_grad(set_to_none=set_to_none)
        for entry in self._projected:
            entry["accum"] = None  # discard any un-stepped layerwise accumulation
            p = entry["param"]
            if p.grad is not None:
                if set_to_none:
                    p.grad = None
                else:
                    p.grad.zero_()

    def state_dict(self) -> dict:
        return {
            "inner": self.inner.state_dict(),
            "projectors": [e["projector"].state_dict() for e in self._projected],
            "steps": [e["step"] for e in self._projected],
            "config": {**self._defaults, "weight_decay": self.weight_decay},
        }

    def load_state_dict(self, state_dict: dict) -> None:
        self.inner.load_state_dict(state_dict["inner"])
        if len(state_dict["projectors"]) != len(self._projected):
            raise ValueError(
                f"checkpoint has {len(state_dict['projectors'])} projected params, "
                f"this optimizer has {len(self._projected)}"
            )
        for entry, proj_state, step in zip(
            self._projected, state_dict["projectors"], state_dict["steps"]
        ):
            entry["projector"].load_state_dict(proj_state)
            entry["step"] = step

    # ------------------------------------------------------------------
    # Introspection (used by the memory report)
    # ------------------------------------------------------------------
    @property
    def n_projected(self) -> int:
        return len(self._projected)

    def memory_breakdown(self) -> dict[str, int]:
        """Bytes actually held by this optimizer, measured from live tensors.

        Independent of the CUDA allocator, so it isolates the claim being made
        (optimizer state size) from allocator noise and activation memory.
        """
        opt_state = 0
        bookkeeping = 0
        for state in self.inner.state.values():
            for v in state.values():
                if not torch.is_tensor(v):
                    continue
                # Every dtype counts, not just floating point: quantised
                # optimizers (bitsandbytes 8-bit Adam) hold their moments in
                # uint8, and a float-only filter would report them as ~free --
                # a spectacular-looking memory win that is purely an artefact of
                # not looking at the tensors that hold the state.
                # Scalars (torch's per-param `step` counter) are bookkeeping, not
# improvement
                # state that scales with the parameter. Counted, but separately.
                if v.numel() == 1:
                    bookkeeping += v.numel() * v.element_size()
                else:
                    opt_state += v.numel() * v.element_size()

        projections = 0
        for entry in self._projected:
            ortho = entry["projector"].ortho_matrix
            if ortho is None:
                continue
            mats = ortho if isinstance(ortho, tuple) else (ortho,)
            projections += sum(m.numel() * m.element_size() for m in mats)

        surrogates = sum(
            e["surrogate"].numel() * e["surrogate"].element_size() for e in self._projected
        )
        return {
            "optimizer_state": opt_state,
# improvement
            "projections": projections,
            "surrogates": surrogates,
            "bookkeeping": bookkeeping,
            "total": opt_state + projections + surrogates + bookkeeping,
        }

    def _projected_groups(self):
        """The inner param group backing each projected entry, in order."""
        lookup = {}
        for group in self.inner.param_groups:
            for p in group["params"]:
                lookup[id(p)] = group
        return [lookup[id(e["surrogate"])] for e in self._projected]

    def __repr__(self) -> str:
#         return (
            f"ProjectedOptimizer({type(self.inner).__name__}, "
            f"projected={self.n_projected}, rank={self._defaults['rank']}, "
            f"gap={self._defaults['update_proj_gap']})"
        )


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _normalize_groups(params, defaults, min_compression_ratio, proj_dtype) -> list[dict]:
    """Coerce ``params`` into group dicts and demote projections that won't pay."""
    if isinstance(params, Iterable) and not isinstance(params, (list, tuple)):
        params = list(params)
    if len(params) == 0:
        raise ValueError("ProjectedOptimizer got an empty parameter list")

    if isinstance(params[0], dict):
        groups = [dict(g) for g in params]
    else:
        # Bare iterable of tensors: opt every eligible 2D parameter in.
        groups = [{"params": list(params), "projected": True}]

    probe = GaLoreProjector(
        rank=defaults["rank"], proj_type=defaults["proj_type"], proj_dtype=proj_dtype
    )
    out: list[dict] = []
    for group in groups:
        group["params"] = list(group["params"])
        if not group.get("projected", False):
            out.append(group)
            continue

        rank = group.get("rank", defaults["rank"])
        ptype = group.get("proj_type", defaults["proj_type"])
        probe.rank, probe.proj_type = rank, ptype

        keep, demote = [], []
        for p in group["params"]:
            eligible = p.ndim == 2 and probe.compression_ratio(tuple(p.shape)) >= min_compression_ratio
            (keep if eligible else demote).append(p)

        if keep:
            out.append({**group, "params": keep})
        if demote:
            rest = {k: v for k, v in group.items() if k not in ("params", "projected")}
            out.append({**rest, "params": demote, "projected": False})
    return out


def _assert_value_independent(optimizer_cls, optimizer_kwargs) -> None:
    """Empirically verify the optimizer's update ignores the parameter's value.

    A blocklist of known-bad optimizer names would go stale; this actually runs
    the optimizer twice from different starting points with identical gradients
# improvement
    and compares the deltas. LAMB-style trust ratios and Adafactor's
    ``scale_parameter`` fail here, as they should.
    """
    kwargs = {k: v for k, v in optimizer_kwargs.items() if k != "weight_decay"}
    kwargs.setdefault("lr", 1e-2)

    torch.manual_seed(0)
    grads = [torch.randn(4, 4) for _ in range(2)]
    deltas = []
    for init in (0.0, 3.0):
        p = torch.full((4, 4), init)
        try:
            opt = optimizer_cls([p], weight_decay=0.0, **kwargs)
        except TypeError:
            opt = optimizer_cls([p], **kwargs)
        start = p.clone()
        for g in grads:
            p.grad = g.clone()
            opt.step()
        deltas.append(p - start)

    if not torch.allclose(deltas[0], deltas[1], atol=1e-5, rtol=1e-3):
        raise ValueDependentOptimizerError(
            f"{optimizer_cls.__name__}'s update depends on the parameter's value "
            "(e.g. LAMB-style trust ratio, or Adafactor with scale_parameter=True). "
            "ProjectedOptimizer runs the inner optimizer on a zeroed low-rank "
            "surrogate, so such an update would be computed against the wrong "
            "parameter and be silently incorrect. Use an optimizer whose update is "
            "value-independent (AdamW, SGD, RMSprop, Adagrad, Lion, 8-bit Adam), or "
            "pass allow_value_dependent=True if you are certain this probe is wrong."
        )

# Optimized

# Enhanced

# Refined

# Enhanced

# Refined

# Refined

# Optimized

# Optimized

# Optimized

# Enhanced

# Refined

# Enhanced
