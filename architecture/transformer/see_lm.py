"""
The Transformer language model:

    h = TokenEmbedding(ids)
    h = Block_L(... Block_1(h))
    logits = LMHead(RMSNorm(h))

Input has shape (..., seq_len) of token ids and output (..., seq_len, vocab_size)
of unnormalized next-token logits. Every block is pre-norm (see transformer_block.py)
and shares one RoPE table sized to context_length.
"""

import torch
from torch import nn

from architecture.layers.basic_modules import Embedding, Linear
from architecture.layers.rms_norm import RMSNorm
from architecture.layers.rope_matrix import RotaryPositionalEmbedding
from architecture.transformer.transformer_block import TransformerBlock


class TransformerLM(nn.Module):
    """A decoder-only Transformer: token embedding, num_layers pre-norm blocks, final RMSNorm, output head."""
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {vocab_size}")
        if context_length <= 0:
            raise ValueError(f"context_length must be positive, got {context_length}")
        if num_layers < 0:
            raise ValueError(f"num_layers must be non-negative, got {num_layers}")
        if d_ff <= 0:
            raise ValueError(f"d_ff must be positive, got {d_ff}")
        # Checked here as well as in the attention layers: the head dimension is
        # needed for the RoPE table before any layer exists (and there may be none).
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        if d_model % num_heads != 0:
            raise ValueError(f"d_model={d_model} is not divisible by num_heads={num_heads}")
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope_theta = rope_theta

        # One RoPE table for every layer; it has no parameters and its buffer is
        # not persisted, so sharing it adds nothing to the state dict.
        self.rope = RotaryPositionalEmbedding(rope_theta, d_model // num_heads, context_length, device=device)

        # Named to match the checkpoint layout: token_embeddings, layers.N.*, ln_final, lm_head.
        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.layers = nn.ModuleList(
            TransformerBlock(d_model, num_heads, d_ff, rope=self.rope, eps=eps, device=device, dtype=dtype)
            for _ in range(num_layers)
        )
        self.ln_final = RMSNorm(d_model, eps=eps, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)  # untied from token_embeddings

    def forward(self, token_ids: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        if token_ids.dim() < 1:
            raise ValueError(f"expected token_ids of shape (..., seq_len), got {tuple(token_ids.shape)}")
        seq_len = token_ids.shape[-1]
        if seq_len > self.context_length:
            raise ValueError(f"seq_len={seq_len} exceeds context_length={self.context_length}")

        # Positions 0..seq_len-1, generated once rather than in every attention layer.
        if token_positions is None:
            token_positions = torch.arange(seq_len, device=token_ids.device)

        # (..., seq_len) -> (..., seq_len, d_model)
        x = self.token_embeddings(token_ids)
        for layer in self.layers:
            x = layer(x, token_positions)
        # (..., seq_len, d_model) -> (..., seq_len, vocab_size), raw logits
        return self.lm_head(self.ln_final(x))

    def extra_repr(self) -> str:
        return (
            f"vocab_size={self.vocab_size}, context_length={self.context_length}, "
            f"d_model={self.d_model}, num_layers={self.num_layers}, num_heads={self.num_heads}, "
            f"d_ff={self.d_ff}, rope_theta={self.rope_theta}"
        )
