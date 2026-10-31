"""Deepwave simulation and data objective for Gaussian-background FWI."""

import math
from concurrent.futures import ThreadPoolExecutor

import deepwave
import torch


class FWI:
    """Survey simulation and misfit; assign observed recordings before fitting."""

    def __init__(self, survey):
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
        self.surface_velocity = float(survey["surface_velocity"])
        if not math.isfinite(self.surface_velocity) or self.surface_velocity <= 0:
            raise ValueError("surface_velocity must be finite and positive")
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
        self._pool = ThreadPoolExecutor(min(self.n_shots, 8))
        self.observed = None

    def simulate(self, velocity, shot, receivers=None, accuracy=4):
        if receivers is None:
            receivers = self.receivers
        result = deepwave.scalar(
            velocity, self.dx, self.dt, source_amplitudes=self.wavelet,
            source_locations=self.sources[shot:shot + 1], receiver_locations=receivers,
            accuracy=accuracy, pml_width=20, pml_freq=self.source_frequency, max_vel=self.bounds[1],
        )
        # Deepwave returns receiver recordings last; remove the single-shot axis.
        return result[-1][0]

    def forward(self, velocity, accuracy=4):
        """Generate clean recordings [shot, receiver, time]."""
        # Detach before starting worker threads; no_grad itself is thread-local.
        velocity = velocity.detach()
        recordings = self._pool.map(
            lambda s: self.simulate(velocity, s, accuracy=accuracy), range(self.n_shots),
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

    def lowpass(self, recordings, frequency):
        n_fft = 2 * recordings.shape[-1]
        frequencies = torch.fft.rfftfreq(n_fft, self.dt)
        # Fade smoothly from 70% of the cutoff to zero at the cutoff.
        rolloff = ((frequencies - 0.7 * frequency) / (0.3 * frequency)).clamp(0, 1)
        taper = torch.cos(0.5 * torch.pi * rolloff)**2
        spectrum = torch.fft.rfft(recordings, n_fft)
        filtered = torch.fft.irfft(spectrum * taper, n_fft)
        return filtered[..., :recordings.shape[-1]]

    def _data_energy(self, frequency):
        if self.observed is None:
            raise ValueError("Assign problem.observed before fitting")
        expected_shape = (self.n_shots, self.receivers.shape[1], self.nt)
        if tuple(self.observed.shape) != expected_shape:
            raise ValueError(f"Observed recordings must have shape {expected_shape}")
        energy = (self.lowpass(self.observed, frequency)**2).sum()
        if not torch.isfinite(energy) or energy <= torch.finfo(energy.dtype).tiny:
            raise ValueError("Observed data have no usable energy in this frequency band")
        return energy

    def loss(self, velocity, frequency):
        data_energy = self._data_energy(frequency)
        residual = self.lowpass(self.forward(velocity) - self.observed, frequency)
        return float((residual**2).sum() / data_energy)

    def gradient(self, velocity, frequency):
        data_energy = self._data_energy(frequency)

        def shot_gradient(shot):
            shot_velocity = velocity.detach().requires_grad_()
            predicted = self.simulate(shot_velocity, shot)
            residual = self.lowpass(predicted - self.observed[shot], frequency)
            loss = (residual**2).sum()
            loss.backward()
            return loss.detach(), shot_velocity.grad

        losses, gradients = zip(*self._pool.map(shot_gradient, range(self.n_shots)))
        return float(sum(losses) / data_energy), sum(gradients) / data_energy

    def illumination(self, velocity, frequency):
        """Source-side curvature approximation for proposing Gaussians."""
        velocity = velocity.detach()
        depth_cells = torch.arange(self.shape[0])
        horizontal_cells = torch.arange(self.shape[1])
        cells = torch.cartesian_prod(depth_cells, horizontal_cells)[None]

        def shot_energy(shot):
            wavefield = self.simulate(velocity, shot, cells)
            filtered = self.lowpass(wavefield, frequency)
            second_difference = torch.diff(filtered, n=2, dim=-1)
            return (second_difference**2).sum(-1)

        with torch.no_grad():
            energies = self._pool.map(shot_energy, range(self.n_shots))
            energy = sum(energies).reshape(self.shape) / self.dt**4
            return energy / velocity.clamp(*self.bounds)**6

    def birth_threshold(self, frequency):
        """Estimate a noise-based threshold for accepting a six-parameter Gaussian."""
        data_energy = self._data_energy(frequency)
        n_fft = 2 * self.nt
        frequencies = torch.fft.rfftfreq(n_fft, self.dt)
        rolloff = ((frequencies - 0.7 * frequency) / (0.3 * frequency)).clamp(0, 1)
        taper = torch.cos(0.5 * torch.pi * rolloff)**2
        weights = torch.full_like(frequencies, 2.0)
        weights[0] = weights[-1] = 1.0

        # Estimate noise from traces with at least 0.3 s before the direct arrival.
        offsets = (self.receivers[0, :, 1][None, :] - self.sources[:, 0, 1][:, None]).abs() * self.dx
        quiet_samples = (offsets / self.surface_velocity / self.dt).floor().long().clamp_max(self.nt)
        power, window_energy = torch.zeros_like(frequencies), 0.0
        minimum_samples = max(2, math.ceil(0.3 / self.dt))
        for shot, receiver in zip(*torch.where(quiet_samples >= minimum_samples)):
            sample_count = int(quiet_samples[shot, receiver])
            window = torch.hann_window(sample_count)
            noise_segment = self.observed[shot, receiver, :sample_count] * window
            power += torch.fft.rfft(noise_segment, n_fft).abs()**2
            window_energy += float((window**2).sum())
        if window_energy:
            noise_variance = float((weights * taper**2 * power).sum() / window_energy / n_fft)
        else:
            noise_variance = 0.0

        # Six fitted parameters; account for the reduced sample count after filtering.
        sample_count = self.observed.numel()
        effective_samples = sample_count * 0.8125 * frequency * 2 * self.dt
        penalty = 6 * math.log(max(effective_samples, 1.0))
        return penalty * noise_variance * (sample_count / effective_samples) / float(data_energy)
