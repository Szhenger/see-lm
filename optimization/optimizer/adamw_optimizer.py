"""
AdamW, as a torch.optim.Optimizer: Adam with the weight decay decoupled from
the gradient step.

    m_t = β1 m_{t-1} + (1 - β1) g_t
    v_t = β2 v_{t-1} + (1 - β2) g_t²
    α_t = α sqrt(1 - β2^t) / (1 - β1^t)
    θ_t = (1 - α λ) θ_{t-1} - α_t m_t / (sqrt(v_t) + ε),   t = 1, 2, ...

α is the learning rate, (β1, β2) the moment decays, ε the divide guard and λ
the weight decay, which shrinks the parameter by α λ directly rather than
entering the moments. The decay is applied to θ_{t-1}, as in the AdamW paper
and torch.optim.AdamW. The one difference from torch is where ε sits: torch
adds it after dividing sqrt(v_t) by sqrt(1 - β2^t), so the two agree to
rounding at the default ε and diverge only for a large ε. The step count and
both moments are kept per parameter in the optimizer state, so they survive
state_dict round-trips.
"""

import math
from collections.abc import Callable, Iterable

import torch


class AdamW(torch.optim.Optimizer):
    """Bias-corrected Adam step plus decoupled weight decay, in place, for every parameter with a gradient."""
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter] | Iterable[dict],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
    ) -> None:
        # The base class fills each group from defaults; a group dict may override any of them.
        super().__init__(params, defaults={"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay})
        for group in self.param_groups:
            # `not 0 <= x` rather than `x < 0`, so a nan is rejected as well.
            if not 0 <= group["lr"]:
                raise ValueError(f"Invalid learning rate: {group['lr']}")
            if not 0 <= group["eps"]:
                raise ValueError(f"Invalid epsilon value: {group['eps']}")
            beta1, beta2 = group["betas"]
            if not 0 <= beta1 < 1 or not 0 <= beta2 < 1:
                raise ValueError(f"Invalid betas: {group['betas']}")
            if not 0 <= group["weight_decay"]:
                raise ValueError(f"Invalid weight_decay value: {group['weight_decay']}")

    @torch.no_grad()
    def step(self, closure: Callable[[], torch.Tensor] | None = None) -> torch.Tensor | None:
        loss = None
        if closure is not None:
            # The closure recomputes the loss and gradients, so it needs grad back on.
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                # Rejected before any state is touched, as torch does: a failure mid-update would leave θ and m half-stepped.
                if grad.is_sparse or torch.is_complex(grad):
                    raise RuntimeError("AdamW supports dense gradients of real parameters only")
                state = self.state[p]
                if not state:
                    state["t"] = 0
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)
                t = state["t"] + 1
                m, v = state["m"], state["v"]

                # Decoupled decay on θ_{t-1}: scaled by lr, independent of the moments.
                if weight_decay != 0:
                    p.mul_(1 - lr * weight_decay)
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                # Bias correction folded into the step size, per the equations above.
                step_size = lr * math.sqrt(1 - beta2**t) / (1 - beta1**t)
                p.addcdiv_(m, v.sqrt().add_(eps), value=-step_size)
                state["t"] = t

        return loss
