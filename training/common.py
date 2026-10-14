"""
What the training scripts share: the device the model and its batches live on.

The model and the batches fed to it must be on one device. `default_device`
picks the accelerator this machine has, CUDA first, then MPS (Apple silicon),
then the CPU, so that a script which passes its result to TransformerLM and
to get_batch runs on the same device everywhere without naming it.
"""

import torch


def default_device() -> torch.device:
    """The accelerator this machine has: CUDA if available, else MPS, else the CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
