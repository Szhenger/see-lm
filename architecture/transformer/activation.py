"""
The SiLU activation x * sigmoid(x), as used inside the SwiGLU feed-forward
network. Written as a module of its own so that it can be swapped, counted or
inspected independently of the block that uses it.
"""

import torch
from torch import nn


def silu(x: torch.Tensor) -> torch.Tensor:
    # torch.sigmoid is numerically stable at both tails, unlike a hand-rolled
    # 1 / (1 + exp(-x)), which overflows exp for large negative x.
    return x * torch.sigmoid(x)


class SiLU(nn.Module):
    """Elementwise x * sigmoid(x). Stateless, so it works on any shape."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return silu(x)
