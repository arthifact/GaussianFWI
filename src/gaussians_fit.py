"""Adaptive Gaussian fitting: propose, accept, move and prune."""

import math

import torch

from .gaussians import gaussian


class _Candidates:
    """225 Gaussian shapes, scored at every grid cell using FFT correlations."""

    def __init__(self, field):
        self.z, self.x, self.dx = field.z, field.x, field.dx
        self.nz, self.nx = field.background.shape
        shapes = []
        for width_index in range(9):
            width = field.dx * 2 ** (width_index / 2)
            for aspect_ratio in (1, 2, 4):
                angles = range(12) if aspect_ratio > 1 else (0,)
                for angle_index in angles:
                    shapes.append((width * aspect_ratio, width, angle_index * math.pi / 12))
        shape_parameters = field.params.new_tensor(shapes)
        self.long, self.short, self.angle = shape_parameters.T[..., None, None]
        # Wrapped offsets on a doubled grid avoid circular wrap-around in the correlations.
        z_offsets = torch.fft.fftfreq(
            2 * self.nz, 1 / (2 * self.nz), device=self.z.device, dtype=self.z.dtype,
        )
        x_offsets = torch.fft.fftfreq(
            2 * self.nx, 1 / (2 * self.nx), device=self.x.device, dtype=self.x.dtype,
        )
        dz = z_offsets[:, None] * self.dx
        dx = x_offsets[None, :] * self.dx
        atoms = gaussian(dz, dx, self.long, self.short, self.angle)
        self.spectra = torch.fft.rfft2(atoms), torch.fft.rfft2(atoms**2)

    def correlate(self, image, spectra):
        padded_shape = (2 * self.nz, 2 * self.nx)
        image_spectrum = torch.fft.rfft2(image, s=padded_shape)
        correlation = torch.fft.irfft2(image_spectrum * spectra, s=padded_shape)
        return correlation[:, :self.nz, :self.nx]

    @torch.no_grad()
    def propose(self, gradient, illumination, velocity, frequency, max_candidates=6):
        slope = self.correlate(gradient, self.spectra[0])
        curvature = self.correlate(illumination, self.spectra[1])
        floor = (1e-12 * curvature.max()).clamp_min(torch.finfo(curvature.dtype).tiny)
        score = slope**2 / curvature.clamp_min(floor)
        # Keep proposed widths at least a quarter of the local wavelength.
        score[self.short < velocity / (4 * frequency)] = 0
        best_score, best_shape = score.max(0)
        proposals = []
        for _ in range(max_candidates):
            if best_score.max() <= 0:
                break
            iz, ix = torch.unravel_index(best_score.argmax(), best_score.shape)
            shape_index = best_shape[iz, ix]
            z, x = float(iz) * self.dx, float(ix) * self.dx
            long = float(self.long[shape_index])
            proposals.append(dict(
                z=z, x=x, long=long, short=float(self.short[shape_index]),
                angle=float(self.angle[shape_index]),
            ))
            best_score[(self.z - z)**2 + (self.x - x)**2 < long**2] = 0
        return proposals


def _adam_step(field, state, gradient):
    """Keep each Gaussian's optimizer history when other Gaussians are added or removed."""
    age, first_moment, second_moment = state.unbind(1)
    age += 1
    first_moment.mul_(0.9).add_(0.1 * gradient)
    second_moment.mul_(0.999).add_(0.001 * gradient**2)
    # Parameters: z, x, log(long), log(short), angle, amplitude.
    learning_rates = field.params.new_tensor([2.0, 2.0, 0.02, 0.02, 0.02, 10.0])
    corrected_mean = first_moment / (1 - 0.9**age)
    corrected_variance = second_moment / (1 - 0.999**age)
    field.params -= learning_rates * corrected_mean / (corrected_variance.sqrt() + 1e-20)


def _grow(field, problem, candidates, gradient, illumination, frequency, loss, threshold):
    """Check proposed Gaussians against the actual loss before adding them.

    Accepted additions update the model and loss. Later proposals still use the
    previous gradient, so their estimated amplitude needs a fresh forward check.
    """
    velocity = field.velocity()
    lower, upper = problem.bounds
    active = (velocity >= lower) & (velocity <= upper)
    proposals = candidates.propose(gradient * active, illumination, velocity.clamp(lower, upper), frequency)
    for candidate in proposals:
        shape = gaussian(
            field.z - candidate["z"], field.x - candidate["x"],
            candidate["long"], candidate["short"], candidate["angle"],
        )
        iz, ix = round(candidate["z"] / field.dx), round(candidate["x"] / field.dx)
        local_velocity = float(velocity[iz, ix].clamp(lower, upper))
        active = (velocity >= lower) & (velocity <= upper)
        slope = float((gradient * shape * active).sum())
        trial = math.copysign(0.05 * local_velocity, -slope)
        trial_loss = problem.loss((velocity + trial * shape).clamp(lower, upper), frequency)
        curvature = 2 * (trial_loss - loss - trial * slope) / trial**2
        amplitude = -slope / curvature if curvature > 0 else trial
        amplitude = max(-0.3 * local_velocity, min(0.3 * local_velocity, amplitude))
        new_loss = problem.loss((velocity + amplitude * shape).clamp(lower, upper), frequency)
        if new_loss <= loss - threshold:
            field.add(**candidate, amp=amplitude)
            velocity, loss = velocity + amplitude * shape, new_loss


def fit_gaussians(field, problem, frequency, iterations=325):
    """Fit one low-pass frequency and return the loss before each update."""
    if not math.isfinite(frequency) or frequency <= 0 or iterations < 0:
        raise ValueError("frequency must be finite and positive, and iterations nonnegative")
    losses = []
    if iterations == 0:
        return losses
    lower, upper = problem.bounds
    candidates = _Candidates(field)
    threshold = problem.birth_threshold(frequency)
    illumination = problem.illumination(field.velocity().clamp(lower, upper), frequency)
    adam = field.params.new_zeros((len(field), 3, 6))
    for iteration in range(iterations):
        params = field.params.clone().requires_grad_()
        velocity = field.velocity(params).clamp(lower, upper)
        loss, gradient = problem.gradient(velocity.detach(), frequency)
        losses.append(loss)
        if iteration % 10 == 0:
            # Add useful Gaussians and remove weak ones, keeping their Adam rows aligned.
            previous_count = len(field)
            _grow(field, problem, candidates, gradient, illumination, frequency, loss, threshold)
            new_state = adam.new_zeros((len(field) - previous_count, 3, 6))
            adam = torch.cat((adam, new_state))
            keep = field.params[:, 5].abs() > 5.0
            field.keep(keep)
            adam = adam[keep]
        else:
            # Backpropagate the velocity-grid gradient into the Gaussian parameters.
            velocity.backward(gradient)
            _adam_step(field, adam, params.grad)
            iz = (field.params[:, 0] / field.dx).round().long().clamp(0, velocity.shape[0] - 1)
            ix = (field.params[:, 1] / field.dx).round().long().clamp(0, velocity.shape[1] - 1)
            min_log_width = (velocity.detach()[iz, ix] / (4 * frequency)).clamp_min(field.dx).log()
            field.params[:, 2:4] = torch.maximum(field.params[:, 2:4], min_log_width[:, None])
    print(f"Gaussians: {frequency:g} Hz, {len(field)} atoms, loss {loss:.3e}")
    return losses
