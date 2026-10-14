"""
Cosine learning rate schedule with linear warmup:

    t < T_w            α_max t / T_w
    T_w <= t <= T_c    α_min + (1 + cos(π (t - T_w) / (T_c - T_w))) (α_max - α_min) / 2
    t > T_c            α_min

α_max and α_min are the peak and final learning rates, T_w the warmup
iterations and T_c the iteration at which the cosine reaches α_min.
"""

import math


def cosine_lr(it: int, max_learning_rate: float, min_learning_rate: float, warmup_iters: int, cosine_cycle_iters: int) -> float:
    """Learning rate at iteration `it` under warmup, cosine decay, then the floor."""
    if it < 0:
        raise ValueError(f"it must be non-negative, got {it}")
    if warmup_iters < 0 or cosine_cycle_iters < warmup_iters:
        raise ValueError(f"need 0 <= warmup_iters <= cosine_cycle_iters, got {warmup_iters} and {cosine_cycle_iters}")

    if it < warmup_iters:
        return max_learning_rate * it / warmup_iters
    if it > cosine_cycle_iters:
        return min_learning_rate
    # Warmup and cycle of equal length: the only in-range iteration is the peak.
    if cosine_cycle_iters == warmup_iters:
        return max_learning_rate
    progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
    return min_learning_rate + 0.5 * (1 + math.cos(math.pi * progress)) * (max_learning_rate - min_learning_rate)
