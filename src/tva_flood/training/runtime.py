"""Shared RNG, checkpoint, sampling and distributed helpers."""

import os
import random

import numpy as np
import torch
import torch.distributed as dist


def sample_windows(length, count):
    """Evenly spaced validation indices, including both ends; no target selection."""
    if length < 1 or count < 1:
        raise ValueError("Need positive validation length and sample count")
    return np.unique(np.rint(np.linspace(0, length - 1, min(count, length))).astype(int))


def save_atomic(obj, path):
    temp = path.with_name(path.name + ".part")
    with open(temp, "wb") as f:
        torch.save(obj, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def capture_rng(device):
    result = dict(
        python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state()
    )
    if device.type == "cuda":
        result["cuda"] = torch.cuda.get_rng_state(device)
    return result


def restore_rng(state, device):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if device.type == "cuda":
        torch.cuda.set_rng_state(state["cuda"], device)


def reduced(tensor, world):
    if world > 1:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor
