"""
The cross-entropy loss, as used to train the language model on next-token
prediction:

    l_i = -log softmax(o_i)[x_{i+1}]
        = log sum_j exp(o_ij - m_i) - (o_i[x_{i+1}] - m_i),   m_i = max_j o_ij

Input logits have shape (..., vocab_size) and targets shape (...) of token ids;
the output is the scalar mean of l_i over every batch-like position, returned
in the compute dtype (at least float32) rather than the logits' dtype, so a
bf16 or fp16 model does not get a quantized loss value. The max is
subtracted so exp cannot overflow, and the target's log-probability is taken in
closed form rather than as log(softmax), which would give log(0) = -inf when
the target's probability underflows. Only the exponentials are kept for
backward: the target logit is gathered by an autograd Function that saves just
the ids, and the shifted logits are then exponentiated in place.
"""

import torch

from architecture.common import check_ids, upcast


class _GatherTarget(torch.autograd.Function):
    """x.gather(-1, index) that saves only index for backward; torch's gather saves the whole x."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(index)
        ctx.shape = x.shape
        return x.gather(-1, index)

    @staticmethod
    def backward(ctx, grad: torch.Tensor) -> tuple[torch.Tensor, None]:
        (index,) = ctx.saved_tensors
        # What gather_backward does for a dense grad: zeros of x's shape with grad scattered at index.
        return grad.new_zeros(ctx.shape).scatter_add_(-1, index, grad), None


def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Mean of -log softmax(logits)[target] over all leading dimensions."""
    if logits.dim() == 0:
        raise ValueError("cross_entropy expects logits with a vocabulary dimension, got a 0-d tensor")
    if targets.shape != logits.shape[:-1]:
        raise ValueError(
            f"expected targets of shape {tuple(logits.shape[:-1])} for logits of shape "
            f"{tuple(logits.shape)}, got {tuple(targets.shape)}"
        )
    check_ids(targets, "cross_entropy")

    # The exponentials and the sum are computed in at least float32 (see architecture.common).
    logits, _ = upcast(logits, "cross_entropy")
    # The shift cancels in the loss, so it is taken off the graph: amax would otherwise save the full logits for backward.
    shifted = logits - logits.detach().amax(dim=-1, keepdim=True)
    # (..., vocab_size) -> (...), the shifted logit of each target id, gathered before the in-place exp below.
    # int64 is required: gather accepts int32 but returns wrong rows for an expanded int32 index on the CPU.
    target_logit = _GatherTarget.apply(shifted, targets.to(torch.int64).unsqueeze(-1)).squeeze(-1)
    # In place: nothing has saved shifted, so the exponentials are the only full-size tensor kept for backward.
    log_normalizer = shifted.exp_().sum(dim=-1).log()
    # Not cast back: the loss is a scalar that is logged and scaled, so it keeps the wider precision.
    return (log_normalizer - target_logit).mean()
