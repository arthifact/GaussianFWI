"""Run this file for smoothed-gradient pixel FWI from an original model in data/."""

import json
import math
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import deepwave
import matplotlib
import numpy as np
import torch
import torch.nn.functional as F

matplotlib.use("Agg")  # Save figures without opening a window.
import matplotlib.pyplot as plt


def _smooth(image, sigma):
    """Gaussian smoothing in pixels, applied horizontally then vertically."""
    radius = min(int(3 * sigma), min(image.shape) - 1)
    if radius == 0:
        return image
    offsets = torch.arange(-radius, radius + 1, dtype=image.dtype, device=image.device)
    kernel = torch.exp(-0.5 * (offsets / sigma)**2)
    kernel = (kernel / kernel.sum())[None, None]
    padded = F.pad(image[None, None], (radius, radius, radius, radius), mode="reflect")
    horizontal = F.conv2d(padded, kernel[..., None, :])
    smoothed = F.conv2d(horizontal, kernel[..., :, None])
    return smoothed[0, 0]


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


class PixelFWI:
    """Own pixel model, survey and observations for this comparison."""

    def __init__(self, initial, survey):
        self.shape = tuple(survey["shape"])
        if len(self.shape) != 2 or any(n < 2 or int(n) != n for n in self.shape):
            raise ValueError("shape must contain two whole cell counts, each at least 2")
        self.shape = tuple(int(n) for n in self.shape)
        self.dx, self.dt = float(survey["dx"]), float(survey["dt"])
        self.nt, self.n_shots = int(survey["nt"]), int(survey["n_shots"])
        if self.nt != survey["nt"] or self.n_shots != survey["n_shots"]:
            raise ValueError("nt and n_shots must be whole numbers")
        self.source_frequency = float(survey["source_frequency"])
        self.bounds = tuple(survey["bounds"])
        if (len(self.bounds) != 2 or not all(math.isfinite(v) and v > 0 for v in self.bounds)
                or self.bounds[0] >= self.bounds[1]):
            raise ValueError("bounds must be positive (minimum, maximum) velocities in increasing order")
        nx = self.shape[1]
        n_receivers = nx if survey["n_receivers"] is None else int(survey["n_receivers"])
        if survey["n_receivers"] is not None and n_receivers != survey["n_receivers"]:
            raise ValueError("n_receivers must be a whole number or None")
        if not 1 <= self.n_shots <= nx or not 1 <= n_receivers <= nx:
            raise ValueError("Use between 1 and nx surface shots and receivers")
        if not all(math.isfinite(v) and v > 0 for v in (self.dx, self.dt, self.source_frequency)) or self.nt < 2:
            raise ValueError("Spacing, time step and source frequency must be finite and positive; nt must be at least 2")
        self.sources = torch.zeros(self.n_shots, 1, 2, dtype=torch.long)
        self.sources[:, 0, 1] = torch.linspace(0, nx - 1, self.n_shots).round().long()
        self.receivers = torch.zeros(1, n_receivers, 2, dtype=torch.long)
        self.receivers[0, :, 1] = torch.linspace(0, nx - 1, n_receivers).round().long()
        self.wavelet = deepwave.wavelets.ricker(
            self.source_frequency, self.nt, self.dt, 1.5 / self.source_frequency,
        ).reshape(1, 1, self.nt)
        if tuple(initial.shape) != self.shape:
            raise ValueError("The initial model must match the survey dimensions")
        self.background = initial.detach().clone()
        self.details = torch.zeros_like(initial, requires_grad=True)
        self.observed = None
        self._pool = ThreadPoolExecutor(min(self.n_shots, 8))

    def velocity(self):
        return self.background + self.details

    def _simulate(self, velocity, shot, accuracy=4):
        result = deepwave.scalar(
            velocity, self.dx, self.dt,
            source_amplitudes=self.wavelet,
            source_locations=self.sources[shot:shot + 1],
            receiver_locations=self.receivers,
            accuracy=accuracy, pml_width=20, pml_freq=self.source_frequency, max_vel=self.bounds[1],
        )
        # Deepwave returns receiver recordings last; remove the single-shot axis.
        return result[-1][0]

    def forward(self, velocity, accuracy=8):
        """Generate clean recordings [shot, receiver, time] from the true model."""
        velocity = velocity.detach()
        recordings = self._pool.map(
            lambda shot: self._simulate(velocity, shot, accuracy), range(self.n_shots),
        )
        return torch.stack(list(recordings))

    def add_noise(self, shots, snr_db, seed=0):
        """Add reproducible Gaussian noise with the source wavelet's spectrum."""
        white = torch.randn(shots.shape, generator=torch.Generator().manual_seed(seed))
        n_fft = 2 * shots.shape[-1]
        white_spectrum = torch.fft.rfft(white, n_fft)
        wavelet_spectrum = torch.fft.rfft(self.wavelet[0, 0], n_fft).abs()
        noise = torch.fft.irfft(white_spectrum * wavelet_spectrum, n_fft)
        noise = noise[..., :shots.shape[-1]]
        scaled_noise = noise * shots.norm() / noise.norm()
        return shots + scaled_noise / 10 ** (snr_db / 20)

    def _lowpass(self, recordings, frequency):
        n_fft = 2 * recordings.shape[-1]
        frequencies = torch.fft.rfftfreq(n_fft, self.dt)
        # Fade smoothly from 70% of the cutoff to zero at the cutoff.
        rolloff = ((frequencies - 0.7 * frequency) / (0.3 * frequency)).clamp(0, 1)
        taper = torch.cos(0.5 * torch.pi * rolloff)**2
        spectrum = torch.fft.rfft(recordings, n_fft)
        filtered = torch.fft.irfft(spectrum * taper, n_fft)
        return filtered[..., :recordings.shape[-1]]

    def _gradient(self, velocity, frequency):
        if self.observed is None:
            raise ValueError("Assign model.observed before fitting")
        expected_shape = (self.n_shots, self.receivers.shape[1], self.nt)
        if tuple(self.observed.shape) != expected_shape:
            raise ValueError(f"Observed recordings must have shape {expected_shape}")
        data_energy = (self._lowpass(self.observed, frequency)**2).sum()
        if not torch.isfinite(data_energy) or data_energy <= torch.finfo(data_energy.dtype).tiny:
            raise ValueError("Observed data have no usable energy in this frequency band")

        def shot_gradient(shot):
            shot_velocity = velocity.detach().requires_grad_()
            predicted = self._simulate(shot_velocity, shot)
            residual = self._lowpass(predicted - self.observed[shot], frequency)
            loss = (residual**2).sum()
            loss.backward()
            return loss.detach(), shot_velocity.grad

        losses, gradients = zip(*self._pool.map(shot_gradient, range(self.n_shots)))
        return float(sum(losses) / data_energy), sum(gradients) / data_energy

    def fit(self, frequency, iterations=340):
        """Fit one frequency with fresh Adam state; return the loss before each update."""
        if not math.isfinite(frequency) or frequency <= 0 or iterations < 0:
            raise ValueError("frequency must be finite and positive, and iterations nonnegative")
        losses = []
        if iterations == 0:
            return losses
        optimizer = torch.optim.Adam([self.details], lr=10.0, eps=1e-20)
        for _ in range(iterations):
            velocity = self.velocity().detach()
            loss, gradient = self._gradient(velocity, frequency)
            losses.append(loss)
            smoothing_width = 0.1 * float(velocity.mean()) / frequency / self.dx
            gradient = _smooth(gradient, smoothing_width)
            optimizer.zero_grad()
            self.details.grad = gradient
            optimizer.step()
            with torch.no_grad():
                self.details.clamp_(self.bounds[0] - self.background, self.bounds[1] - self.background)
        print(f"Smoothed pixels: {frequency:g} Hz, loss {loss:.3e}")
        return losses


def save_models(true, final, path):
    """Save this comparison's true and recovered models as a PNG."""
    fig, axes = plt.subplots(1, 2, figsize=(6, 4), layout="constrained")
    for ax, title, velocity in zip(axes, ("True", "Smoothed pixels"), (true, final)):
        image = ax.imshow(velocity.detach().cpu(), cmap="jet", vmin=float(true.min()), vmax=float(true.max()))
        ax.set(title=title, xlabel="x (cells)")
    axes[0].set_ylabel("Depth (cells)")
    fig.colorbar(image, ax=list(axes), label="Velocity (m/s)", shrink=0.8)
    fig.savefig(path, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    # Edit this comparison's settings, then press Run.
    model_name = "marmousi"  # marmousi, bp_salt, overthrust, seam, sigsbee
    survey = {
        "shape": (70, 70),       # depth cells, horizontal cells
        "dx": 10.0,             # cell spacing, metres
        "n_shots": 5,
        "n_receivers": None,    # None means one receiver per surface cell
        "dt": 0.001,            # seconds per sample
        "nt": 1000,             # samples per recording
        "source_frequency": 15.0,  # Ricker wavelet peak frequency, Hz
    }
    noise_db = 30  # None for clean observations
    noise_seed = 0
    frequencies = (6, 9, 13, 20, 30)
    iterations = 340  # Per frequency
    torch.set_num_threads(1)

    # Prepare this run entirely in memory from the original model.
    true = prepare_model(model_name, survey["shape"])
    survey["bounds"] = (1000.0, max(5500.0, float(true.max())))
    depth = torch.arange(true.shape[0])[:, None] * survey["dx"]
    initial = (1500 + 1500 * depth / depth.max()).expand_as(true).clone()
    model = PixelFWI(initial, survey)
    observed = model.forward(true, accuracy=8)
    if noise_db is not None:
        observed = model.add_noise(observed, noise_db, noise_seed)
    model.observed = observed

    progress = []
    for frequency in frequencies:
        started = perf_counter()
        losses = model.fit(frequency, iterations=iterations)
        seconds = perf_counter() - started
        error = model.velocity().detach() - true
        progress.append({
            "stage": "pixels", "frequency": frequency, "loss": losses,
            "seconds": seconds,
            "relative_l2": float(error.norm() / true.norm()),
            "rmse": float(error.square().mean().sqrt()),
        })
    final = model.velocity().detach()

    folder = Path(__file__).resolve().parent
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%fZ")
    output = folder / "outputs" / f"pixels_smoothed_{model_name}_{run_id}"
    output.mkdir(parents=True, exist_ok=False)
    settings = {
        "method": "pixels_smoothed",
        "model": model_name, "survey": survey, "noise_db": noise_db, "noise_seed": noise_seed,
        "frequencies": frequencies, "iterations": iterations,
        "initial_velocity": [1500.0, 3000.0],
        "versions": {name: version(name) for name in ("torch", "deepwave", "numpy")},
    }
    np.savez_compressed(
        output / "result.npz", true=true.numpy(), final=final.numpy(),
        settings=json.dumps(settings), progress=json.dumps(progress),
    )
    save_models(true, final, output / "result.png")
    print(f"Saved {output}")
