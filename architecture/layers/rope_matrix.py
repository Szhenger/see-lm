"""
Rotary positional embedding (RoPE), as used by modern language models.

RoPE encodes position by rotating each query and key vector: the d_k
dimensions are taken as d_k / 2 pairs, and pair k at position i is rotated by
the angle i * theta^(-2k / d_k). The dot product of a rotated query and a
rotated key then depends only on their relative offset, which is what makes
the scheme work inside attention. The full d_k x d_k rotation matrix is block
diagonal with 2 x 2 blocks, so the rotation is applied pairwise instead of
being materialized.

The cos and sin values depend only on the position and the pair index, never on
the input, the layer or the batch, so they are computed once for every
position up to max_seq_len and kept as a non-persistent buffer. One instance can
be shared by all attention layers, and the module has no learnable parameters.
"""

import torch
from torch import nn

from architecture.common import lookup, upcast


class RotaryPositionalEmbedding(nn.Module):
    """Rotates each adjacent pair of x's last dimension by a position-dependent angle."""
    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        if theta <= 0:
            raise ValueError(f"theta must be positive, got {theta}")
        if d_k <= 0 or d_k % 2 != 0:
            raise ValueError(f"d_k must be a positive even integer, got {d_k}")
        if max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be positive, got {max_seq_len}")
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        # Fixed tables, so a buffer rather than a parameter. Not persisted: it
        # is cheap to rebuild and carrying it in checkpoints would only tie
        # the checkpoint to one max_seq_len.
        self.register_buffer("cos_sin", self._build_table(device), persistent=False)

    def _build_table(self, device: torch.device | None) -> torch.Tensor:
        # One angular frequency per pair of dimensions, theta^(-2k / d_k) for
        # k in [0, d_k / 2). The angles are formed in float64: position times
        # frequency grows large at long contexts and float32 would lose the
        # low bits before cos and sin are ever taken. The angles are formed on
        # the CPU because some accelerators (MPS) have no float64; only the
        # float32 table moves to the target device. So, unlike the rest of
        # architecture.common's policy, a float64 input is rotated by a cos and
        # sin of float32 precision (the arithmetic itself runs in float64).
        # The output is cast back to the input dtype.
        inv_freq = self.theta ** (-torch.arange(0, self.d_k, 2, dtype=torch.float64) / self.d_k)
        positions = torch.arange(self.max_seq_len, dtype=torch.float64)
        angles = torch.outer(positions, inv_freq)  # (max_seq_len, d_k / 2)

        # cos and sin are stacked into one (max_seq_len, 2, d_k / 2) table so
        # that forward gathers both with a single lookup; RoPE runs on Q and K
        # in every attention layer.
        return torch.stack((angles.cos(), angles.sin()), dim=1).to(device=device, dtype=torch.float32)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        return self.rotate(x, *self.tables(token_positions, x.dim()))

    def tables(self, token_positions: torch.Tensor, ndim: int) -> tuple[torch.Tensor, torch.Tensor]:
        """The cos and sin rows for token_positions, shaped to broadcast against an ndim-dimensional input.

        Gathered once and passed to `rotate` for each input that shares the
        positions (the query and key of one attention layer).
        """
        if token_positions.dim() < 1 or token_positions.dim() > ndim - 1:
            raise ValueError(
                f"token_positions must have shape (..., seq_len) with at most "
                f"{ndim - 1} dimensions, got {tuple(token_positions.shape)}"
            )

        # .to(dtype), .half() and .bfloat16() cast floating point buffers along
        # with the parameters, which would quantize the table. It is rebuilt in
        # float32 on first use after such a cast, once per instance however
        # many layers share it. A device move keeps the dtype and costs nothing.
        # Built outside inference mode so the table is an ordinary tensor even
        # when the first use after a cast is an inference pass.
        if self.cos_sin.dtype != torch.float32:
            with torch.inference_mode(False):
                self.cos_sin = self._build_table(self.cos_sin.device)

        # Gather the table along the sequence dimension (see
        # architecture.common.lookup for why this is not `self.cos_sin[token_positions]`)
        # into shape (*batch, 1..., seq_len, 2, d_k / 2), where *batch are
        # token_positions' leading dimensions. Those must line up with the
        # input's leading dimensions from the left, not the right, so any
        # dimensions the input has beyond them (heads, for instance) get a
        # singleton inserted before seq_len rather than being matched against batch.
        n_missing = ndim - 1 - token_positions.dim()
        shape = (*token_positions.shape[:-1], *([1] * n_missing), token_positions.shape[-1], 2, self.d_k // 2)
        return lookup(self.cos_sin, token_positions, "RotaryPositionalEmbedding").view(shape).unbind(-2)

    def rotate(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """Rotate each adjacent pair of x's last dimension by the angles in cos and sin (from `tables`)."""
        # The shape checks do not depend on dtype, so they run on the original:
        # nothing is copied before a bad call is rejected.
        if x.dim() < 2 or x.shape[-1] != self.d_k:
            raise ValueError(
                f"expected x of shape (..., seq_len, d_k={self.d_k}), got {tuple(x.shape)}"
            )
        if cos.shape[-2] != x.shape[-2]:
            raise ValueError(
                f"token_positions has sequence length {cos.shape[-2]} "
                f"but x has sequence length {x.shape[-2]}"
            )

        # The rotation is computed in at least float32 (see architecture.common);
        # only the final result is cast back to the input dtype.
        x, in_dtype = upcast(x, "RotaryPositionalEmbedding")

        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        out1 = x1 * cos - x2 * sin
        out2 = x1 * sin + x2 * cos

        # Interleave the rotated pairs back into the original dimension order.
        out = torch.stack((out1, out2), dim=-1).flatten(-2)
        return out.to(in_dtype)

    def extra_repr(self) -> str:
        return f"theta={self.theta}, d_k={self.d_k}, max_seq_len={self.max_seq_len}"
