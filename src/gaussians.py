"""A fixed background plus movable, anisotropic Gaussian velocity corrections."""

import math

import torch


def gaussian(dz, dx, long, short, angle):
    """Unit-height Gaussian; widths and offsets in metres, angle from x toward depth."""
    angle = torch.as_tensor(angle, dtype=dz.dtype, device=dz.device)
    along = (torch.cos(angle) * dx + torch.sin(angle) * dz) / long
    across = (torch.cos(angle) * dz - torch.sin(angle) * dx) / short
    return torch.exp(-0.5 * (along**2 + across**2))


class GaussianField:
    """Each parameter row is [z, x, log(long), log(short), angle, amplitude].

    Widths use logarithms so they stay positive during optimization.
    Grid shape, dtype and device come from the background.
    """

    def __init__(self, background, dx):
        if background.ndim != 2 or not background.is_floating_point() or dx <= 0:
            raise ValueError("background must be a floating-point 2D tensor and dx must be positive")
        self.background = background.detach().clone()
        self.dx = float(dx)
        self.params = background.new_zeros((0, 6))
        nz, nx = background.shape
        z = torch.arange(nz, dtype=background.dtype, device=background.device) * dx
        x = torch.arange(nx, dtype=background.dtype, device=background.device) * dx
        self.z, self.x = torch.meshgrid(z, x, indexing="ij")

    def __len__(self):
        return len(self.params)

    def velocity(self, params=None):
        parameters = self.params if params is None else params
        # Each parameter gets two grid axes so all Gaussians can be evaluated together.
        z, x, log_long, log_short, angle, amplitude = parameters.T[..., None, None]
        shapes = gaussian(self.z - z, self.x - x, log_long.exp(), log_short.exp(), angle)
        return self.background + (amplitude * shapes).sum(0)

    def add(self, z, x, long, short, angle, amp):
        row = self.params.new_tensor([[z, x, math.log(long), math.log(short), angle, amp]])
        self.params = torch.cat((self.params, row))

    def keep(self, mask):
        self.params = self.params[mask]
