"""
The position-wise feed-forward network, a SwiGLU block, and the SiLU
activation it is built on, as used by modern language models.

SwiGLU gates an up-projection with a SiLU-activated projection and then
projects back down:
    FFN(x) = W2 (SiLU(W1 x) * W3 x)
with no biases. Built on this repo's Linear rather than torch.nn.Linear so that
the parameter layout and the gating are all explicit. The inner width d_ff defaults
to 8/3 * d_model rounded up to a multiple of 64, so that the three matrices
together cost about the same as a classic 4 * d_model two-matrix FFN while
keeping the GEMM shapes friendly to the hardware.
"""

import math

import torch
from torch import nn

from architecture.layers.basic_modules import Linear


def silu(x: torch.Tensor) -> torch.Tensor:
    # torch.sigmoid is numerically stable at both tails, unlike a hand-rolled
    # 1 / (1 + exp(-x)), which overflows exp for large negative x.
    return x * torch.sigmoid(x)


class SiLU(nn.Module):
    """The SiLU activation x * sigmoid(x), as used inside the SwiGLU feed-forward
    network. Elementwise and stateless, so it works on any shape. A module of its
    own so that it can be swapped, counted or inspected independently of the
    block that uses it.
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return silu(x)


class SwiGLU(nn.Module):
    """FFN(x) = w2(SiLU(w1(x)) * w3(x)); w1 is the gate, w3 the up-projection."""
    def __init__(
        self,
        d_model: int,
        d_ff: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if d_ff is None:
            d_ff = 64 * math.ceil(8 * d_model / 3 / 64)
        self.d_model = d_model
        self.d_ff = d_ff

        # Named w1, w2, w3 so the state dict keys are `w1.weight` and so on,
        # the layout checkpoints use for this block.
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)
        # A submodule rather than a bare call to silu, so the activation shows
        # up in named_modules() and can be swapped, counted or hooked. It has
        # no parameters, so the state dict keys are unchanged.
        self.act = SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(self.act(self.w1(x)) * self.w3(x))

    def extra_repr(self) -> str:
        return f"d_model={self.d_model}, d_ff={self.d_ff}"
