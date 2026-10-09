"""
The pre-norm Transformer block:

    y = x + MultiHeadSelfAttention(RMSNorm(x))
    z = y + SwiGLU(RMSNorm(y))

Input and output have shape (..., seq_len, d_model). Normalization is applied
before each sublayer and the residual path carries the unnormalized stream.
"""

import torch
from torch import nn

from architecture.layers.multihead_attention import MultiHeadSelfAttention
from architecture.layers.positionwise_feedforward import SwiGLU
from architecture.layers.rms_norm import RMSNorm
from architecture.layers.rope_matrix import RotaryPositionalEmbedding


class TransformerBlock(nn.Module):
    """One pre-norm block: causal multi-head self-attention then a SwiGLU feed-forward, each with a residual."""
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        rope: RotaryPositionalEmbedding | None = None,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff

        # Named to match the checkpoint layout: ln1 -> attn, ln2 -> ffn.
        self.ln1 = RMSNorm(d_model, eps=eps, device=device, dtype=dtype)
        self.attn = MultiHeadSelfAttention(d_model, num_heads, rope=rope, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model, eps=eps, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), token_positions)
        return x + self.ffn(self.ln2(x))

    def extra_repr(self) -> str:
        return f"d_model={self.d_model}, num_heads={self.num_heads}, d_ff={self.d_ff}"
