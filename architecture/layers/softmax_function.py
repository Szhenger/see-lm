"""
The softmax, as used by attention to turn scores into a probability distribution.

softmax(x)_i = exp(x_i) / sum_j exp(x_j), taken along one dimension of x. The
output has the same shape as the input, and along the chosen dimension it is a
normalized probability distribution. Written from scratch rather than on top of
torch.softmax so that the numerics are explicit.

exp overflows for inputs above about 88 in float32, and the sum of exponentials
can overflow long before any single term does. Since softmax is invariant to
shifting every x_i by the same constant, the maximum along the dimension is
subtracted first: every exponent is then at most zero, exp is at most one, and
the largest term is exactly one, so neither overflow nor an all-zero denominator
is possible for finite input.
"""

import torch
from torch import nn

from architecture.common import upcast


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Softmax over dimension `dim` of x, with the max-subtraction trick."""
    # The exponentials and the sum are computed in at least float32 (see
    # architecture.common); only the final result is cast back.
    x, in_dtype = upcast(x, "softmax")

    # An empty dimension has nothing to normalize; amax would raise on it,
    # where torch.softmax returns an (empty) copy of the input. A copy, not
    # the input itself, so that writing into the result cannot touch the
    # caller's tensor. x.size(dim) rather than x.shape[dim] so that an
    # out-of-range dim gets torch's dimension error, not a tuple IndexError.
    # A 0-d tensor has no dimension to size, so it skips the guard: amax
    # accepts dim 0 or -1 on a scalar, and the result is 1, as from
    # torch.softmax.
    if x.dim() > 0 and x.size(dim) == 0:
        return x.to(in_dtype, copy=True)

    # amax rather than max: it returns just the values, with no indices, and
    # keepdim keeps the result broadcastable against x. Entries that are -inf
    # (a mask) become exp(-inf) = 0 and drop out of the sum; a slice that is
    # entirely -inf has no distribution to give and comes back NaN, as it
    # does from torch.softmax.
    shifted = x - x.amax(dim=dim, keepdim=True)
    exp = shifted.exp()
    return (exp / exp.sum(dim=dim, keepdim=True)).to(in_dtype)


class Softmax(nn.Module):
    """Softmax over a fixed dimension. Stateless, so it works on any shape."""
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return softmax(x, self.dim)

    def extra_repr(self) -> str:
        return f"dim={self.dim}"
