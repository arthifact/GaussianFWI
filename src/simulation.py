"""Deepwave simulation and data objective for Gaussian-background FWI."""

import hashlib
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
        if self.source_frequency > 0.5 / self.dt:
            raise ValueError("The source peak frequency must not exceed Nyquist")
        self.sources = torch.zeros(self.n_shots, 1, 2, dtype=torch.long)
        self.sources[:, 0, 1] = torch.linspace(0, nx - 1, self.n_shots).round().long()
        self.receivers = torch.zeros(1, n_receivers, 2, dtype=torch.long)
        self.receivers[0, :, 1] = torch.linspace(0, nx - 1, n_receivers).round().long()
        self.wavelet = deepwave.wavelets.ricker(
            self.source_frequency, self.nt, self.dt, 1.5 / self.source_frequency,
        ).reshape(1, 1, self.nt)
        self._pool = ThreadPoolExecutor(min(self.n_shots, 8))
        self.observed = None

    @property
    def observed(self):
        return self._observed

    @observed.setter
    def observed(self, recordings):
        """Replace recordings and clear noise statistics belonging to the previous data."""
        self._observed = recordings
        self._noise_energy = {}
        self._noise_signature = None

    def _recording_signature(self):
        """Identify samples by value, including edits through shared NumPy memory."""
        if self.observed is None:
            return None
        samples = self.observed.detach().cpu().resolve_neg().contiguous()
        raw_bytes = samples.reshape(-1).view(torch.uint8).numpy()
        return samples.shape, samples.dtype, hashlib.sha256(raw_bytes).digest()

    @property
    def noise_energy(self):
        """Supplied band energies for the current recordings, keyed by cutoff in Hz."""
        signature = self._recording_signature()
        if signature != self._noise_signature:
            self._noise_energy = {}
            self._noise_signature = signature
        return self._noise_energy

    @noise_energy.setter
    def noise_energy(self, energies):
        self._noise_signature = self._recording_signature()
        self._noise_energy = energies

    def simulate(self, velocity, shot, receivers=None, accuracy=4):
        if receivers is None:
            receivers = self.receivers
        result = deepwave.scalar(
            velocity, self.dx, self.dt, source_amplitudes=self.wavelet.to(velocity),
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
        generator = torch.Generator(device=shots.device).manual_seed(seed)
        white = torch.randn(shots.shape, dtype=shots.dtype, device=shots.device, generator=generator)
        n_fft = 2 * shots.shape[-1]
        white_spectrum = torch.fft.rfft(white, n_fft)
        wavelet_spectrum = torch.fft.rfft(self.wavelet[0, 0].to(shots), n_fft).abs()
        noise = torch.fft.irfft(white_spectrum * wavelet_spectrum, n_fft)
        noise = noise[..., :shots.shape[-1]]
        scaled_noise = noise * shots.norm() / noise.norm()
        return shots + scaled_noise / 10 ** (snr_db / 20)

    def validate_cutoff(self, frequency):
        nyquist = 0.5 / self.dt
        if not math.isfinite(frequency) or not 0 < float(frequency) <= nyquist:
            raise ValueError(
                f"Invalid cutoff {frequency} Hz. The frequency must be finite and positive "
                f"and no higher than Nyquist ({nyquist} Hz)."
            )

    def _lowpass_taper(self, recordings, frequency):
        self.validate_cutoff(frequency)
        frequency = float(frequency)
        dtype = recordings.dtype if recordings.is_floating_point() else torch.get_default_dtype()
        frequencies = torch.fft.rfftfreq(2 * recordings.shape[-1], self.dt, dtype=dtype, device=recordings.device)
        # Fade smoothly from 70% of the cutoff to zero at the cutoff.
        rolloff = ((frequencies - 0.7 * frequency) / (0.3 * frequency)).clamp(0, 1)
        taper = torch.cos(0.5 * torch.pi * rolloff)**2
        taper[frequencies >= frequency] = 0.0  # cos(pi/2) leaves a roundoff residue.
        taper[0] = 1.0  # DC is always passed, even for a sub-bin cutoff.
        return taper

    def _band_power_weights(self, frequency):
        """Power in the full spectrum represented by each real-FFT bin."""
        weights = self._lowpass_taper(self.observed, frequency).square()
        weights[1:-1] *= 2  # Include the negative-frequency partners, except at DC and Nyquist.
        return weights

    def lowpass(self, recordings, frequency):
        n_fft = 2 * recordings.shape[-1]
        taper = self._lowpass_taper(recordings, frequency)
        spectrum = torch.fft.rfft(recordings, n_fft)
        filtered = torch.fft.irfft(spectrum * taper, n_fft)
        return filtered[..., :recordings.shape[-1]]

    def _data_energy(self, frequency):
        self.validate_cutoff(frequency)
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
        self.validate_cutoff(frequency)
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

    def _noise_variance(self, frequency):
        """Prefer supplied band energy; otherwise estimate variance before arrivals."""
        self.validate_cutoff(frequency)
        frequency = float(frequency)
        # JSON stores cutoff keys as strings, including "6" and "6.0" for the same band.
        energies = {
            float(value) for cutoff, value in self.noise_energy.items()
            if float(cutoff) == frequency
        }
        if len(energies) > 1:
            raise ValueError(f"Conflicting noise energies supplied for {frequency:g} Hz")
        if energies:
            energy = energies.pop()
            if not math.isfinite(energy) or energy < 0:
                raise ValueError("Supplied noise energy must be finite and nonnegative")
            return energy / self.observed.numel()

        n_fft = 2 * self.nt
        weights = self._band_power_weights(frequency)

        # Even a refracted arrival needs at least straight-line distance / maximum speed.
        offsets = torch.linalg.vector_norm((self.receivers - self.sources).double(), dim=-1) * self.dx
        max_speed = max(self.bounds[1], self.surface_velocity)
        quiet_samples = (offsets / (max_speed * self.dt)).floor().clamp_max(self.nt).long()
        power, window_energy = torch.zeros_like(weights), 0.0
        minimum_samples = max(2, math.ceil(0.3 / self.dt))
        for shot, receiver in zip(*torch.where(quiet_samples >= minimum_samples)):
            sample_count = int(quiet_samples[shot, receiver])
            window = torch.hann_window(sample_count, dtype=weights.dtype, device=weights.device)
            noise_segment = self.observed[shot, receiver, :sample_count] * window
            power += torch.fft.rfft(noise_segment, n_fft).abs()**2
            window_energy += float((window**2).sum())
        if not window_energy:
            raise ValueError(
                f"No usable pre-arrival noise windows at {frequency:g} Hz. "
                "A trace needs at least 0.3 s and two samples before the earliest possible arrival. "
                "Supply noise_energy[frequency] after assigning observed "
                "(0.0 for known clean data)."
            )
        return float((weights * power).sum() / window_energy / n_fft)

    def birth_threshold(self, frequency):
        """BIC-inspired growth threshold with an approximate effective sample count."""
        data_energy = self._data_energy(frequency)
        noise_variance = self._noise_variance(frequency)
        if noise_variance == 0:
            return 0.0
        # Scale the recorded sample count by the mean power gain on the padded FFT grid.
        sample_count = self.observed.numel()
        weights = self._band_power_weights(frequency)
        effective_samples = sample_count * float(weights.sum(dtype=torch.float64)) / (2 * self.nt)
        if effective_samples <= 1:
            raise ValueError(
                f"Too few effective samples ({effective_samples:g}) for a noisy Gaussian growth decision "
                f"at {frequency:g} Hz. Use a longer recording or a wider band."
            )
        penalty = 6 * math.log(effective_samples)
        return penalty * noise_variance * (sample_count / effective_samples) / float(data_energy)
