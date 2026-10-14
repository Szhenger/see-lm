"""
Stochastic gradient descent with a decaying learning rate, as a
torch.optim.Optimizer:

    θ_{t+1} = θ_t - (α / sqrt(t + 1)) ∇L(θ_t),   t = 0, 1, 2, ...

α is the initial learning rate and t counts the steps taken on each parameter,
so the step size shrinks over training. t is kept per parameter in the
optimizer state, so it survives state_dict round-trips and a parameter whose
gradient was absent for a step is not advanced.
"""

import math
from collections.abc import Callable, Iterable

import torch


class SGD(torch.optim.Optimizer):
    """p -= lr / sqrt(t + 1) * p.grad, in place, for every parameter with a gradient."""
    def __init__(self, params: Iterable[torch.nn.Parameter] | Iterable[dict], lr: float = 1e-3) -> None:
        # The base class fills each group from defaults; a group dict may carry its own lr.
        super().__init__(params, defaults={"lr": lr})
        for group in self.param_groups:
            # `not 0 <= x` rather than `x < 0`, so a nan is rejected as well.
            if not 0 <= group["lr"]:
                raise ValueError(f"Invalid learning rate: {group['lr']}")

    @torch.no_grad()
    def step(self, closure: Callable[[], torch.Tensor] | None = None) -> torch.Tensor | None:
        loss = None
        if closure is not None:
            # The closure recomputes the loss and gradients, so it needs grad back on.
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                t = state.get("t", 0)
                # One fused in-place update: no scaled copy of the gradient is allocated.
                p.add_(p.grad, alpha=-lr / math.sqrt(t + 1))
                state["t"] = t + 1

        return loss
