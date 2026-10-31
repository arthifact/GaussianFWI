"""Load an original model from data/ and resize it in memory for this run."""

from pathlib import Path

import numpy as np
import torch


def prepare_model(name, shape):
    if len(shape) != 2 or any(n < 2 or int(n) != n for n in shape):
        raise ValueError("shape must be (depth cells, horizontal cells), with at least two whole cells per axis")
    shape = tuple(int(n) for n in shape)
    root = Path(__file__).resolve().parents[1]
    velocity = torch.from_numpy(np.load(root / "data" / f"{name}.npy"))
    if velocity.ndim != 2 or not velocity.is_floating_point():
        raise ValueError("The velocity model must be a 2D floating-point array")
    if not torch.isfinite(velocity).all() or (velocity <= 0).any():
        raise ValueError("The velocity model must contain finite, positive velocities")
    # Interpolation expects [batch, channel, depth, x]; keep just the resized 2D field.
    velocity = torch.nn.functional.interpolate(velocity[None, None], size=shape, mode="area")[0, 0]
    return velocity
