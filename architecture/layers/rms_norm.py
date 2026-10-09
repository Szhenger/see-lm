"""
Root mean square layer normalization, as used by modern language models.

Each d_model vector is rescaled to unit root mean square and then multiplied by
a learned per-dimension gain. There is no mean subtraction and no bias, which is
what separates it from LayerNorm. Written from scratch rather than on top of
torch.nn.RMSNorm so that the parameter layout, the upcast and the normalization
are all explicit. A module of its own because both the feed-forward block and
the attention block are wrapped in it.
"""

import torch
from torch import nn

from architecture.common import upcast


class RMSNorm(nn.Module):
    """y = x * rsqrt(mean(x^2) + eps) * weight, over the last dimension of x."""
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.eps = eps

        # One gain per hidden dimension, the same layout torch uses for
        # `*.weight`, so checkpoints load without renaming. Allocated straight
        # on the target device and dtype and filled in place.
        self.weight = nn.Parameter(torch.empty(d_model, device=device, dtype=dtype))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Starting at one makes the fresh module a pure normalization.
        nn.init.ones_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The shape check does not depend on dtype, so it runs on the original:
        # nothing is copied before a bad call is rejected.
        if x.dim() == 0 or x.shape[-1] != self.d_model:
            raise ValueError(
                f"expected the last dimension of x to be d_model={self.d_model}, "
                f"got input of shape {tuple(x.shape)}"
            )

        # The mean of squares is computed in at least float32 (see
        # architecture.common). Only the final result is cast back, so the
        # gain multiply also happens at the wider precision.
        x, in_dtype = upcast(x, "RMSNorm")
        inv_rms = torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + self.eps)
        return (x * inv_rms * self.weight).to(in_dtype)

    def extra_repr(self) -> str:
        return f"d_model={self.d_model}, eps={self.eps}"
