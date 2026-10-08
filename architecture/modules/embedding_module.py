"""
The token embedding: a learned lookup table from token id to a d_model vector.

Written from scratch rather than on top of torch.nn.Embedding (and without
torch.nn.functional.embedding) so that the parameter layout, the initialization
and the lookup are all explicit.
"""

import torch
from torch import nn


class Embedding(nn.Module):
    """weight has shape (num_embeddings, d_model); forward is weight[token_ids]."""
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
        # index_select rather than `self.weight[token_ids]`: plain indexing would
        # let a negative id (say a -1 padding sentinel) silently wrap around to
        # the end of the table, and would treat a bool tensor as a row mask.
        # index_select raises on both, like torch.nn.functional.embedding.
        rows = self.weight.index_select(0, token_ids.reshape(-1))
        return rows.view(*token_ids.shape, self.embedding_dim)

    def extra_repr(self) -> str:
        return f"num_embeddings={self.num_embeddings}, embedding_dim={self.embedding_dim}"
