"""
What the architecture modules share: the floating point input check and the
precision policy for the arithmetic inside them, and the table lookup that
the embedding and RoPE both gather rows with.

RMSNorm and softmax sum or exponentiate many terms, attention contracts Q
against K and the weights against V, and RoPE multiplies the input against
float32 cos and sin tables. In bf16 or fp16 that arithmetic loses precision
and can overflow, so every one of them validates that the input is floating
point, computes in at least float32, and casts only the final result back to
the input dtype. The policy is defined once here (float64 input stays
float64; bf16, fp16 and float32 compute in float32) so that the modules
cannot drift apart, and so that a change such as an upcast switch is made in
one place. `upcast` is the single-input form; `upcast_all` is the same policy
for a function of several tensors, which are promoted to one common dtype.

Embedding looks up a row of its weight per token id and RoPE looks up a row of
its cos and sin tables per position. Both go through `lookup`, so the id
validation and the choice of index_select over plain indexing are made once.
"""

import functools
from collections.abc import Sequence

import torch


def upcast(x: torch.Tensor, name: str) -> tuple[torch.Tensor, torch.dtype]:
    """Return (x promoted to at least float32, x's original dtype).

    Raises TypeError, naming the caller, if x is not a floating point tensor.
    The caller does its arithmetic on the returned tensor and casts the final
    result back with `.to(in_dtype)`.
    """
    if not x.is_floating_point():
        raise TypeError(f"{name} expects a floating point input, got {x.dtype}")
    in_dtype = x.dtype
    return x.to(torch.promote_types(in_dtype, torch.float32)), in_dtype


def upcast_all(
    xs: Sequence[torch.Tensor], name: str
) -> tuple[list[torch.Tensor], torch.dtype]:
    """Return (xs promoted to one common dtype of at least float32, the common dtype of xs).

    The common dtype is torch's promotion over the inputs' dtypes (bf16 with
    bf16 is bf16; float32 with float64 is float64), so no input is ever
    narrowed, and the returned dtype is what the caller casts its final
    result back to. Raises TypeError, naming the caller, if any x is not a
    floating point tensor. Each tensor is converted at most once.
    """
    for x in xs:
        if not x.is_floating_point():
            raise TypeError(f"{name} expects floating point inputs, got {x.dtype}")
    in_dtype = functools.reduce(torch.promote_types, (x.dtype for x in xs))
    compute_dtype = torch.promote_types(in_dtype, torch.float32)
    return [x.to(compute_dtype) for x in xs], in_dtype


def lookup(table: torch.Tensor, ids: torch.Tensor, name: str) -> torch.Tensor:
    """Return the rows of `table` at `ids`, with shape (*ids.shape, *table.shape[1:]).

    Raises TypeError, naming the caller, unless ids is int32 or int64: those
    are the only dtypes index_select accepts, and checking here gives the
    caller's name in the error rather than torch's generic one.

    index_select rather than `table[ids]`: plain indexing would let a negative
    id (say a -1 padding sentinel) silently wrap around to the end of the
    table, and would treat a bool tensor as a row mask. index_select raises on
    both, and on ids >= len(table) (a Python IndexError on the CPU, a
    device-side assert on an accelerator). Either way there is no
    device-to-host sync, which matters because RoPE runs this on Q and K in
    every attention layer.
    """
    if ids.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"{name} expects int32 or int64 ids, got {ids.dtype}")
    return table.index_select(0, ids.reshape(-1)).view(*ids.shape, *table.shape[1:])
