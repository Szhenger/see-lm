"""
Gradient clipping by the combined l2 norm of all gradients:

    g <- g * M / (||g||_2 + ε)   if ||g||_2 > M

where g is the concatenation of every parameter's gradient, M is the maximum
norm and ε keeps the scale finite. The gradients are modified in place;
parameters without a gradient are skipped.
"""

from collections.abc import Iterable

import torch


@torch.no_grad()
def clip_grad_norm_(
    parameters: Iterable[torch.nn.Parameter] | torch.nn.Parameter, max_l2_norm: float, *, eps: float = 1e-6
) -> torch.Tensor:
    """Scale the gradients of `parameters` in place so their combined l2 norm is at most max_l2_norm; return the norm before clipping."""
    if max_l2_norm <= 0:
        raise ValueError(f"max_l2_norm must be positive, got {max_l2_norm}")
    # A lone tensor would otherwise be iterated over its rows, which carry no grad, and silently clip nothing.
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return torch.zeros(())

    # Norm of the concatenation, as the norm of the per-gradient norms; accumulated in at least float32.
    # The per-gradient norms are gathered on one device, as the gradients may be spread over several.
    device = grads[0].device
    norms = torch.stack([torch.linalg.vector_norm(g, dtype=torch.promote_types(g.dtype, torch.float32)).to(device) for g in grads])
    total_norm = torch.linalg.vector_norm(norms)
    # Clamped rather than branched, so no device sync: a norm within the limit multiplies by exactly 1.
    scale = (max_l2_norm / (total_norm + eps)).clamp(max=1.0)
    for g in grads:
        g.mul_(scale.to(g.device))
    return total_norm
