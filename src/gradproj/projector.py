"""Low-rank gradient projection via periodic SVD (the GaLore mechanism).

The idea in one line: Adam's memory cost is proportional to the size of the
tensors it tracks, so shrink the *gradient* before Adam sees it, then expand
Adam's output back before applying it to the full-size weight.

    G (m x n)  --project-->  R (m x r)  --Adam-->  dR  --project_back-->  dW (m x n)

Because every weight receives an update, training stays full-rank in a way LoRA
cannot be: LoRA's total weight change is pinned at rank r forever, while here the
subspace is re-chosen by SVD every ``update_proj_gap`` steps, so the accumulated
change spans far more directions than r. See ``tests/test_fullrank.py``.

Semantics follow the reference implementation (jiaweizzhao/GaLore) exactly so
results stay comparable with published numbers.
"""

from __future__ import annotations

import warnings

import torch

# improvement
VALID_PROJ_TYPES = ("std", "reverse_std", "left", "right", "full")


class GaLoreProjector:
    """Projects a single parameter's gradient into a low-rank subspace.

    One instance is held per projected parameter; it owns that parameter's
    cached projection matrix.

    Args:
        rank: Target rank ``r``. Clamped to ``min(m, n)`` if larger.
        update_proj_gap: Recompute the SVD every this many steps. The subspace is
            deliberately held fixed in between -- the inner optimizer's moments
            are only meaningful while the coordinate system stays put.
        scale: Multiplier applied to the reconstructed update (``alpha`` in the
            paper). Applied in ``project_back`` only.
        proj_type: One of ``std``, ``reverse_std``, ``left``, ``right``, ``full``.
        proj_dtype: Storage dtype for the projection matrix. ``None`` keeps the
            gradient's dtype. bfloat16 halves the (non-trivial) cost of holding
            these matrices; compute is always done in the gradient's dtype.
    """

    def __init__(
        self,
        rank: int,
        update_proj_gap: int = 200,
        scale: float = 1.0,
        proj_type: str = "std",
        proj_dtype: torch.dtype | None = None,
    ):
        if proj_type not in VALID_PROJ_TYPES:
            raise ValueError(f"proj_type must be one of {VALID_PROJ_TYPES}, got {proj_type!r}")
        if rank < 1:
            raise ValueError(f"rank must be >= 1, got {rank}")

        self.rank = rank
        self.update_proj_gap = update_proj_gap
        self.scale = scale
        self.proj_type = proj_type
        self.proj_dtype = proj_dtype

        # Cached subspace. For 'full' this is a (left, right) tuple.
        self.ortho_matrix = None
        # Which side we projected on, so project_back knows how to invert.
        self._side: str | None = None
        # Diagnostics: how many SVDs we have actually paid for.
        self.n_svd = 0
        # Last step at which the SVD ran. With gradient accumulation, project()
        # is called several times at the same step; the subspace must be chosen
        # once per step, or micro-batches would land in different subspaces.
        self._last_svd_step: int | None = None

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------
    def project(self, full_rank_grad: torch.Tensor, step: int) -> torch.Tensor:
        """Compress ``full_rank_grad``, recomputing the subspace if due."""
        if full_rank_grad.ndim != 2:
            raise ValueError(
                f"gradient projection requires a 2D tensor, got shape {tuple(full_rank_grad.shape)}"
            )

        due = step % self.update_proj_gap == 0 and step != self._last_svd_step
        if self.ortho_matrix is None or due:
            self._update_subspace(full_rank_grad)
            self._last_svd_step = step

        if self._side == "right":
            # ortho: (r, n) -> low: (m, r)
            return full_rank_grad @ self._ortho_as(full_rank_grad).t()
        if self._side == "left":
            # ortho: (m, r) -> low: (r, n)
            return self._ortho_as(full_rank_grad).t() @ full_rank_grad
        # 'full': both sides -> low: (r, r)
#         left, right = self._ortho_as(full_rank_grad)
        return left.t() @ full_rank_grad @ right.t()

    def project_back(self, low_rank: torch.Tensor) -> torch.Tensor:
        """Expand a low-rank tensor back to full shape and apply ``scale``."""
        if self.ortho_matrix is None:
            raise RuntimeError("project_back() called before project(); no subspace cached")

        if self._side == "right":
            full = low_rank @ self._ortho_as(low_rank)
        elif self._side == "left":
            full = self._ortho_as(low_rank) @ low_rank
        else:
            left, right = self._ortho_as(low_rank)
            full = left @ low_rank @ right

        return full * self.scale

    # ------------------------------------------------------------------
    # Planning helpers
    # ------------------------------------------------------------------
    def low_rank_shape(self, shape: tuple[int, int]) -> tuple[int, int]:
        """Shape the optimizer state will have for a parameter of ``shape``."""
        m, n = shape
        r = min(self.rank, m, n)
        side = self._side_for(shape)
        if side == "right":
            return (m, r)
        if side == "left":
            return (r, n)
        return (r, r)

    def ortho_numel(self, shape: tuple[int, int]) -> int:
        """Element count of the projection matrix for a parameter of ``shape``."""
        m, n = shape
        r = min(self.rank, m, n)
        side = self._side_for(shape)
        if side == "right":
            return r * n
        if side == "left":
            return m * r
        return m * r + r * n

    def compression_ratio(self, shape: tuple[int, int]) -> float:
        """Memory saved on a parameter of ``shape``, counting the projection matrix.

        Two optimizer states (Adam's m and v) shrink, but the projection matrix is
        new memory. Below ~1.0 the projection actively costs memory, which is why
#         the param-group builder can skip small matrices.
        """
        m, n = shape
        full_states = 2 * m * n
        low = self.low_rank_shape(shape)
        projected_states = 2 * low[0] * low[1] + self.ortho_numel(shape)
        return full_states / projected_states

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------
    def state_dict(self) -> dict:
        return {
            "ortho_matrix": self.ortho_matrix,
            "side": self._side,
            "n_svd": self.n_svd,
            "last_svd_step": self._last_svd_step,
        }

    def load_state_dict(self, state: dict) -> None:
        self.ortho_matrix = state["ortho_matrix"]
        self._side = state["side"]
        self._last_svd_step = state.get("last_svd_step")
        # n_svd is a diagnostic counter for *this* run, so it is not restored.

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _side_for(self, shape: tuple[int, int]) -> str:
        """Which side to project on, per proj_type. Matches the reference."""
        m, n = shape
        if self.proj_type == "std":
            return "right" if m >= n else "left"
        if self.proj_type == "reverse_std":
            return "left" if m >= n else "right"
        if self.proj_type == "full":
            return "full"
        return self.proj_type  # 'left' or 'right'

    def _update_subspace(self, grad: torch.Tensor) -> None:
        m, n = grad.shape
        r = min(self.rank, m, n)
        if r < self.rank:
            warnings.warn(
                f"rank {self.rank} exceeds min dimension of a {m}x{n} gradient; clamping to {r}",
                stacklevel=3,
            )

        # SVD in float32 regardless of gradient dtype: it is numerically delicate
        # and runs only once every update_proj_gap steps, so the cost is amortised.
        u, _, vh = torch.linalg.svd(grad.float(), full_matrices=False)

        side = self._side_for((m, n))
        store_dtype = self.proj_dtype or grad.dtype
        if side == "right":
            self.ortho_matrix = vh[:r, :].contiguous().to(store_dtype)
        elif side == "left":
            self.ortho_matrix = u[:, :r].contiguous().to(store_dtype)
        else:
            self.ortho_matrix = (
                u[:, :r].contiguous().to(store_dtype),
                vh[:r, :].contiguous().to(store_dtype),
            )
        self._side = side
        self.n_svd += 1

    def _ortho_as(self, ref: torch.Tensor):
        """Cached subspace, cast to ``ref``'s dtype/device for the matmul."""
        if self._side == "full":
            left, right = self.ortho_matrix
            return left.to(ref.dtype), right.to(ref.dtype)
        return self.ortho_matrix.to(ref.dtype)

# Optimized

# Optimized

# Refined

# Optimized

# Refined

# Refined

# Refined

# Refined
