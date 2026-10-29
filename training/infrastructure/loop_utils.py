"""
What a training loop needs around the model: batches from an encoded corpus,
and checkpoints of the run.

Data. The corpus is a 1-D integer array of token ids, usually a uint16 .npy
file written by encode_datasets.py. `load_tokens` opens such a file as a
read-only memory map: the array stays on disk and only the pages a batch
touches are ever read, so a corpus far larger than memory costs nothing to
open. `get_batch` draws batch_size windows of context_length + 1 consecutive
tokens, each starting at an offset chosen uniformly at random; the first
context_length tokens of a window are the input and the last context_length
its next-token targets, so the returned tensors have shape
(batch_size, context_length). Offsets come from torch's CPU random number
generator (or the CPU generator passed in; torch rejects any other), so a run
that calls torch.manual_seed draws the same batches whatever the device. The
tensors go to the device given, or to the machine's accelerator
(training.common.default_device) when none is.

Checkpoints. A checkpoint is one torch.save file holding a dict of three
entries: the model's state dict, the optimizer's state dict and the iteration
count. Two guarantees: a checkpoint saved to a path is written to a .partial
file first and renamed into place, so a crash mid-save never leaves a
truncated file under the real name; and tensors are loaded onto the CPU
first, then moved to wherever the given model's parameters live, so a
checkpoint written on one device (MPS here) restores on another.
"""

import operator
import os
from typing import IO, BinaryIO

import numpy as np
import torch

from training.common import default_device

Target = str | os.PathLike | BinaryIO | IO[bytes]
_KEYS = ("model", "optimizer", "iteration")


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def load_tokens(path: str | os.PathLike) -> np.memmap:
    """Open the token array at path as a read-only memory map, without reading it."""
    tokens = np.load(path, mmap_mode="r")
    # np.load returns an open archive handle for an .npz, not an array.
    if not isinstance(tokens, np.memmap):
        kind = type(tokens).__name__
        if hasattr(tokens, "close"):
            tokens.close()
        raise TypeError(f"load_tokens expects a .npy array, {os.fspath(path)} holds a {kind}")
    if tokens.ndim != 1:
        raise ValueError(f"load_tokens expects a 1-d token array, {os.fspath(path)} has shape {tokens.shape}")
    if not np.issubdtype(tokens.dtype, np.integer):
        raise TypeError(f"load_tokens expects integer token ids, {os.fspath(path)} holds {tokens.dtype}")
    return tokens


def get_batch(
    dataset: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str | torch.device | None = None,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample (inputs, targets) of shape (batch_size, context_length) and dtype int64 from dataset, on device (default: the accelerator)."""
    if device is None:
        device = default_device()
    if dataset.ndim != 1:
        raise ValueError(f"get_batch expects a 1-d dataset, got shape {dataset.shape}")
    if not np.issubdtype(dataset.dtype, np.integer):
        raise TypeError(f"get_batch expects an integer dataset, got {dataset.dtype}")
    if batch_size < 1:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if context_length < 1:
        raise ValueError(f"context_length must be positive, got {context_length}")
    if len(dataset) <= context_length:
        raise ValueError(
            f"dataset holds {len(dataset)} tokens, but a batch needs at least "
            f"context_length + 1 = {context_length + 1}"
        )
    # A window of context_length + 1 tokens must end inside the dataset, so the
    # last valid offset is len(dataset) - context_length - 1 (randint's high is exclusive).
    starts = torch.randint(0, len(dataset) - context_length, (batch_size,), generator=generator)

    # One gather of (batch_size, context_length + 1) from the (possibly memory-mapped)
    # array, as int64 for the embedding lookup, then one copy to the device. The gather
    # already yields a fresh array, so copy=False only skips a second copy of int64 data.
    index = starts.numpy()[:, None] + np.arange(context_length + 1)
    window = torch.from_numpy(dataset[index].astype(np.int64, copy=False)).to(device)
    return window[:, :-1], window[:, 1:]


# --------------------------------------------------------------------------
# Checkpoints
# --------------------------------------------------------------------------
def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, iteration: int, out: Target) -> None:
    """Write the model and optimizer state dicts and the iteration count to out, a path or a binary file object."""
    # operator.index accepts any integer type (int, numpy ints, 0-d int tensors) and rejects floats.
    try:
        iteration = operator.index(iteration)
    except TypeError:
        raise TypeError(f"save_checkpoint expects an integer iteration, got {iteration!r}") from None
    if iteration < 0:
        raise ValueError(f"save_checkpoint expects a non-negative iteration, got {iteration}")
    checkpoint = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "iteration": iteration}

    if not isinstance(out, (str, os.PathLike)):
        torch.save(checkpoint, out)
        return
    path = os.fspath(out)
    partial = path + ".partial"
    try:
        torch.save(checkpoint, partial)
        os.replace(partial, path)
    finally:
        if os.path.exists(partial):
            os.unlink(partial)


def load_checkpoint(src: Target, model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> int:
    """Restore the model and optimizer from src, a path or a binary file object, and return the saved iteration count."""
    # Onto the CPU first: load_state_dict then copies each tensor to its parameter's device.
    checkpoint = torch.load(src, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"load_checkpoint expects a dict with keys {_KEYS}, got {type(checkpoint).__name__}")
    if missing := [key for key in _KEYS if key not in checkpoint]:
        raise ValueError(f"load_checkpoint expects a dict with keys {_KEYS}; missing {missing}")
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return int(checkpoint["iteration"])
