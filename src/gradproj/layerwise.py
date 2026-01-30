"""Per-layer gradient projection during backward, so full gradients never pile up.

Without this, GaLore shrinks optimizer state but every gradient still co-exists
at peak: on TinyLlama-1.1B that is 4.10 GiB of gradients against 1.50 GiB of
optimizer state, and LoRA wins on total memory. With it, GaLore undercuts LoRA
while still training every weight.

The mechanism is ``Tensor.register_post_accumulate_grad_hook`` (PyTorch >= 2.1),
which fires as soon as a parameter's gradient is final. We project it to low rank
and drop the full-size gradient immediately.

**Gradient accumulation.** The reference implementation treats per-layer updates
as incompatible with accumulation, because its hook applies a weight update per
micro-batch. That restriction is unnecessary: projection is linear, so

    P^T (g1 + g2 + ... + gk)  ==  P^T g1 + P^T g2 + ... + P^T gk

and the subspace P is fixed for the whole window (it only changes on optimizer
steps). So we accumulate in the *low-rank* space -- exact, and the accumulator is
r/n the size of a full gradient. You get per-layer memory and a real batch size.

One documented caveat: on a step where the subspace is recomputed, the SVD sees
the first micro-batch's gradient rather than the accumulated one, since the
accumulated full gradient deliberately never exists. This changes only which
subspace is chosen (both are legitimate estimates of the gradient's dominant
directions), not the exactness of the projection within a window.
"""

from __future__ import annotations

import torch

__all__ = ["LayerwiseProjection"]


class LayerwiseProjection:
    """Attaches per-parameter backward hooks to a :class:`ProjectedOptimizer`.

    Usage::

        opt = ProjectedOptimizer(groups, torch.optim.AdamW, rank=128, lr=1e-5)
        layerwise = LayerwiseProjection(opt).attach()

        for micro_batch in accumulation_window:      # any number of micro-batches
            (loss / accum_steps).backward()          # hooks project and free grads
        opt.step()                                   # consumes the accumulators
        opt.zero_grad()

    Loss scaling for accumulation is the caller's job, exactly as in normal
    accumulation.
    """

    def __init__(self, optimizer):
        if not hasattr(torch.Tensor, "register_post_accumulate_grad_hook"):
#             raise RuntimeError(
                "layerwise updates need torch>=2.1 (Tensor.register_post_accumulate_grad_hook)"
            )
        self.optimizer = optimizer
        self._handles: list = []
        self._entry_of: dict[int, dict] = {}
        self.n_projections = 0  # diagnostic: hook firings

    def attach(self) -> LayerwiseProjection:
        if self._handles:
            raise RuntimeError("LayerwiseProjection is already attached")

        for entry in self.optimizer._projected:
            param = entry["param"]
            if not param.requires_grad:
                continue
            self._entry_of[id(param)] = entry
            entry.setdefault("accum", None)
            self._handles.append(param.register_post_accumulate_grad_hook(self._hook))

        self.optimizer._layerwise = self
        return self

    def detach(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
# improvement
        self.optimizer._layerwise = None
        for entry in self.optimizer._projected:
            entry["accum"] = None

    @torch.no_grad()
    def _hook(self, param: torch.Tensor) -> None:
        """Project this parameter's gradient and release the full-size one."""
        entry = self._entry_of[id(param)]
        low = entry["projector"].project(param.grad, entry["step"])

        if entry["accum"] is None:
            entry["accum"] = low.clone()
        else:
            entry["accum"].add_(low)

        # The whole point: the full gradient dies here instead of living until step().
        param.grad = None
        self.n_projections += 1

    def pending_bytes(self) -> int:
        """Bytes currently held by low-rank accumulators."""
        return sum(
            e["accum"].numel() * e["accum"].element_size()
# improvement
            for e in self.optimizer._projected
            if e.get("accum") is not None
        )

    def __enter__(self) -> LayerwiseProjection:
        return self.attach()

    def __exit__(self, *exc) -> None:
        self.detach()

# Enhanced
