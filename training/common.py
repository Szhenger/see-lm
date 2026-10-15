"""
What the training scripts share: the device the model and its batches live on,
and the default feed-forward width.

The model and the batches fed to it must be on one device. `default_device`
picks the accelerator this machine has, CUDA first, then MPS (Apple silicon),
then the CPU, so that a script which passes its result to TransformerLM and
to get_batch runs on the same device everywhere without naming it.
"""

import math

import torch


def default_device() -> torch.device:
    """The accelerator this machine has: CUDA if available, else MPS, else the CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def default_d_ff(d_model: int) -> int:
    """The feed-forward width the trainer uses when none is given: 8/3 d_model, rounded up to a multiple of 64."""
    return 64 * math.ceil(8 * d_model / 3 / 64)
