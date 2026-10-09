"""
The basic parameterized modules: the linear layer and the token embedding.

Both are written from scratch rather than on top of torch.nn.Linear and
torch.nn.Embedding (and without torch.nn.functional.embedding) so that the
parameter layout, the initialization and the forward pass are all explicit.
"""

import math

import torch
from torch import nn

from architecture.common import lookup


def trunc_normal_init_(weight: torch.Tensor, fan_in: int, fan_out: int) -> None:
    """Fill `weight` in place as a linear layer with the given fans is initialized.

    N(0, 2 / (fan_in + fan_out)) truncated at three standard deviations.
    """
    # A zero-sized layer has nothing to fill, and its fan sum may be zero.
    if weight.numel() == 0:
        return
    # trunc_normal_'s bounds are absolute values, not multiples of std.
    std = math.sqrt(2.0 / (fan_in + fan_out))
    nn.init.trunc_normal_(weight, mean=0.0, std=std, a=-3.0 * std, b=3.0 * std)


class Linear(nn.Module):
    """Implements the linear transformation y = Wx (and no bias), as used by modern language models."""
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        init_fans: tuple[int, int] | None = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # The fans the initialization uses; a layer that fuses several
        # projections passes the fans of one so the fusion does not narrow it.
        self.init_fans = init_fans if init_fans is not None else (in_features, out_features)

        # W itself, not W^T: the same layout torch uses for `*.weight`, so
        # checkpoints load without renaming or transposing. Allocated straight on
        # the target device and dtype and filled in place, so it is never copied.
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        trunc_normal_init_(self.weight, *self.init_fans)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # `.mT` is a view; matmul passes the transpose flag to the GEMM, so this
        # is the same kernel path torch.nn.functional.linear takes, with no copy.
        return x @ self.weight.mT

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}"


class Embedding(nn.Module):
    """The token embedding: a learned lookup table from token id to a d_model vector.

    weight has shape (num_embeddings, d_model); forward returns row token_id of
    weight for every id, as a (*token_ids.shape, d_model) tensor. Negative ids
    and bool masks are rejected rather than wrapping around or selecting rows.
    """
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        # Row i is the vector for token id i, d_model last: the same layout torch
        # uses for `*.weight`, so checkpoints load without renaming or transposing.
        # Allocated straight on the target device and dtype and filled in place.
        self.weight = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # An empty table has nothing to fill.
        if self.weight.numel() == 0:
            return
        # Embeddings use N(0, 1) truncated at [-3, 3]; unlike Linear, the std does
        # not depend on the fan sizes.
        nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # See architecture.common.lookup for why this is not `self.weight[token_ids]`.
        return lookup(self.weight, token_ids, "Embedding")

    def extra_repr(self) -> str:
        return f"num_embeddings={self.num_embeddings}, embedding_dim={self.embedding_dim}"
