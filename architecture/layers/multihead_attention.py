"""
Scaled dot-product attention and causal multi-head self-attention.

Queries and keys have shape (..., seq_len, d_k) and values (..., seq_len, d_v),
where ... is any number of batch-like dimensions. The attention output has
shape (..., seq_len, d_v). The multi-head module takes (..., seq_len, d_model)
and returns the same shape, with d_k = d_v = d_model / num_heads per head.
"""

import math

import torch
from torch import nn

from architecture.common import upcast_all
from architecture.layers.basic_modules import Linear
from architecture.layers.rope_matrix import RotaryPositionalEmbedding
from architecture.layers.softmax_function import softmax

# The checkpoint layout of the QKV projections, in the order they are fused.
_QKV_KEYS = ("q_proj.weight", "k_proj.weight", "v_proj.weight")


def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """softmax(Q K^T / sqrt(d_k)) V, with an optional boolean mask (True = attend)."""
    # The shape checks do not depend on dtype, so they run on the originals:
    # nothing is copied before a bad call is rejected.
    if Q.dim() < 2 or K.dim() < 2 or V.dim() < 2:
        raise ValueError(
            f"expected Q, K and V of shape (..., seq_len, d), got "
            f"{tuple(Q.shape)}, {tuple(K.shape)} and {tuple(V.shape)}"
        )
    if Q.shape[-1] != K.shape[-1]:
        raise ValueError(f"Q has d_k={Q.shape[-1]} but K has d_k={K.shape[-1]}")
    if K.shape[-2] != V.shape[-2]:
        raise ValueError(f"K has {K.shape[-2]} keys but V has {V.shape[-2]} values")
    if mask is not None and mask.dtype != torch.bool:
        raise TypeError(f"mask must be a boolean tensor, got {mask.dtype}")

    # Computed in at least float32 in the common dtype of Q, K and V (see
    # architecture.common); only the result is cast back to that common dtype,
    # so a float64 K or V is never narrowed to a float32 Q.
    (Q, K, V), in_dtype = upcast_all((Q, K, V), "scaled_dot_product_attention")

    # (..., queries, d_k) @ (..., d_k, keys) -> (..., queries, keys)
    d_k = Q.shape[-1]
    scores = (Q / math.sqrt(d_k)) @ K.transpose(-2, -1)

    # Masked positions become -inf so the softmax gives them zero probability.
    if mask is not None:
        scores.masked_fill_(~mask, float("-inf"))

    # (..., queries, keys) @ (..., keys, d_v) -> (..., queries, d_v)
    weights = softmax(scores, dim=-1)
    return (weights @ V).to(in_dtype)


class MultiHeadSelfAttention(nn.Module):
    """Causal multi-head self-attention: num_heads attention heads over d_model, with optional RoPE."""
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        rope: RotaryPositionalEmbedding | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        if d_model % num_heads != 0:
            raise ValueError(f"d_model={d_model} is not divisible by num_heads={num_heads}")
        if rope is not None and rope.d_k != d_model // num_heads:
            raise ValueError(
                f"rope.d_k={rope.d_k} must equal the head dimension {d_model // num_heads}"
            )
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.rope = rope

        # Q, K and V for all heads come from one fused projection (rows ordered
        # Q, then K, then V), so the input is read and multiplied once. It is
        # initialized as three separate d_model x d_model projections would be.
        self.qkv_proj = Linear(
            d_model, 3 * d_model, device=device, dtype=dtype, init_fans=(d_model, d_model)
        )
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)

        # The fusion stays out of checkpoints: state_dict() emits separate
        # q_proj, k_proj and v_proj weights and either layout loads. The one
        # visible trace is that named_parameters() lists qkv_proj.weight while
        # state_dict() lists the three.
        self.register_state_dict_post_hook(self._split_qkv_weights)
        self.register_load_state_dict_pre_hook(self._fuse_qkv_weights)

    def reset_parameters(self) -> None:
        self.qkv_proj.reset_parameters()
        self.output_proj.reset_parameters()

    # Plain functions rather than methods: torch tags state dict hooks with an
    # attribute, which a bound method cannot carry.
    @staticmethod
    def _split_qkv_weights(module, state_dict, prefix, local_metadata) -> None:
        # Absent when the parameter has been renamed (pruning, parametrization).
        fused = state_dict.pop(prefix + "qkv_proj.weight", None)
        if fused is None:
            return
        for name, weight in zip(_QKV_KEYS, fused.chunk(3, dim=0)):
            state_dict[prefix + name] = weight

    @staticmethod
    def _fuse_qkv_weights(
        module, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ) -> None:
        fused_key = prefix + "qkv_proj.weight"
        if fused_key in state_dict:
            return
        # Start from the current weights, so a checkpoint holding only some of
        # the three projections replaces those and leaves the others as they
        # are. Absent ones are reported under their checkpoint names.
        current = module.qkv_proj.weight.detach()
        chunks = list(current.chunk(3, dim=0))
        for i, name in enumerate(_QKV_KEYS):
            key = prefix + name
            if key not in state_dict:
                missing_keys.append(key)
                continue
            weight = state_dict.pop(key)
            if weight.shape != chunks[i].shape:
                error_msgs.append(
                    f"size mismatch for {key}: copying a param with shape "
                    f"{tuple(weight.shape)} from checkpoint, the shape in current "
                    f"model is {tuple(chunks[i].shape)}."
                )
                continue
            chunks[i] = weight.to(current.device, current.dtype)
        state_dict[fused_key] = torch.cat(chunks, dim=0)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        if x.dim() < 2 or x.shape[-1] != self.d_model:
            raise ValueError(
                f"expected x of shape (..., seq_len, d_model={self.d_model}), got {tuple(x.shape)}"
            )
        if token_positions is not None and self.rope is None:
            raise ValueError("token_positions were given but this module has no rope")
        seq_len = x.shape[-2]

        # One matmul gives (..., seq, 3 * d_model); each third becomes
        # (..., heads, seq, d_head).
        q, k, v = (self._split_heads(t) for t in self.qkv_proj(x).chunk(3, dim=-1))

        if self.rope is not None:
            if token_positions is None:
                # A Python int comparison, so no device sync: the generated
                # positions are 0..seq_len-1 and must all fit the rope table.
                if seq_len > self.rope.max_seq_len:
                    raise ValueError(
                        f"seq_len={seq_len} exceeds rope.max_seq_len={self.rope.max_seq_len}"
                    )
                token_positions = torch.arange(seq_len, device=x.device)
            # One table gather serves both rotations.
            cos, sin = self.rope.tables(token_positions, q.dim())
            q = self.rope.rotate(q, cos, sin)
            k = self.rope.rotate(k, cos, sin)

        # Each query attends to itself and earlier positions only. tril_ is in
        # place, so the mask is one allocation rather than two.
        causal = torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device).tril_()
        out = scaled_dot_product_attention(q, k, v, causal)

        # (..., heads, seq, d_head) -> (..., seq, d_model)
        out = out.transpose(-3, -2).flatten(-2)
        return self.output_proj(out)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        # (..., seq, d_model) -> (..., seq, heads, d_head) -> (..., heads, seq, d_head)
        return x.unflatten(-1, (self.num_heads, self.d_head)).transpose(-3, -2)

    def extra_repr(self) -> str:
        return f"d_model={self.d_model}, num_heads={self.num_heads}"
