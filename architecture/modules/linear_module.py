"""
The linear layer y = Wx, note with no bias, as used by modern language models.

Written from scratch rather than on top of torch.nn.Linear so that the parameter
layout, the initialization and the forward pass are all explicit.
"""

import math
import torch
from torch import nn


class Linear(nn.Module):
    """Implements the linear transformation y = Wx (and no bias)."""
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # W itself, not W^T: the same layout torch uses for `*.weight`, so
        # checkpoints load without renaming or transposing. Allocated straight on
        # the target device and dtype and filled in place, so it is never copied.
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # A zero-sized layer has nothing to fill, and its fan sum may be zero.
        if self.weight.numel() == 0:
            return
        # trunc_normal_'s bounds are absolute values, not multiples of std.
        std = math.sqrt(2.0 / (self.in_features + self.out_features))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3.0 * std, b=3.0 * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # `.mT` is a view; matmul passes the transpose flag to the GEMM, so this
        # is the same kernel path torch.nn.functional.linear takes, with no copy.
        return x @ self.weight.mT

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}"
