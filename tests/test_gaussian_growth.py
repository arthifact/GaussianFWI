import unittest
from unittest.mock import Mock, patch

import torch

from src.gaussians import GaussianField, gaussian
from src.gaussians_fit import _Candidates, _grow, fit_gaussians
from src.simulation import FWI


class GaussianGrowthTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(1)

    def make_bounded_problem(self, true_velocity):
        # Match the clean, five-shot case that admitted four fully clipped atoms.
        problem = FWI({
            "shape": (30, 70), "dx": 10.0, "dt": 0.001, "nt": 1000,
            "n_shots": 5, "n_receivers": None, "source_frequency": 15.0,
            "bounds": (1500.0, 6000.0), "surface_velocity": 1500.0,
        })
        self.addCleanup(problem._pool.shutdown)
        problem.observed = problem.forward(true_velocity, accuracy=8)
        problem.noise_energy = {6: 0.0, 9: 0.0}
        return problem

    def test_fully_clipped_births_leave_the_raw_field_at_the_bound(self):
        initial = torch.full((30, 70), 1500.0)
        problem = self.make_bounded_problem(initial)
        field = GaussianField(initial, problem.dx)
        fit_gaussians(field, problem, 6, iterations=1)
        self.assertEqual(len(field), 0)
        self.assertTrue(torch.equal(field.velocity(), initial))

    def test_next_frequency_can_fit_an_anomaly_after_clipped_proposals(self):
        initial = torch.full((30, 70), 1500.0)
        problem = self.make_bounded_problem(initial)
        field = GaussianField(initial, problem.dx)
        fit_gaussians(field, problem, 6, iterations=1)

        # Fresh observations give the next band a real correction to recover.
        truth = initial.clone()
        truth[2:16, 22:48] += 300.0
        problem.observed = problem.forward(truth, accuracy=8)
        problem.noise_energy[9] = 0.0
        before = problem.loss(initial, 9)
        fit_gaussians(field, problem, 9, iterations=11)
        after = field.velocity().clamp(*problem.bounds)
        self.assertGreater(len(field), 0)
        self.assertFalse(torch.equal(after, initial))
        self.assertLess(problem.loss(after, 9), before)

    def test_failed_amplitude_estimate_keeps_the_passing_trial(self):
        # One narrowband shot recorded 1695 m away reproduces the 9 Hz overshoot.
        problem = FWI({
            "shape": (20, 114), "dx": 15.0, "dt": 0.001, "nt": 2700,
            "n_shots": 1, "n_receivers": 1, "source_frequency": 8.0,
            "bounds": (1000.0, 4000.0), "surface_velocity": 1500.0,
        })
        self.addCleanup(problem._pool.shutdown)
        problem.receivers[0, 0, 1] = 113
        time = torch.arange(problem.nt) * problem.dt - 0.65
        pulse = torch.sin(2 * torch.pi * 8 * time) * torch.exp(-0.5 * (time / 0.16)**2)
        problem.wavelet = pulse[None, None]
        initial = torch.full(problem.shape, 1500.0)
        field = GaussianField(initial, problem.dx)
        candidate = dict(z=0.0, x=840.0, long=960.0, short=240.0, angle=0.0)
        shape = gaussian(
            field.z, field.x - candidate["x"], candidate["long"], candidate["short"], candidate["angle"],
        )
        truth = initial + 75.0 * shape
        problem.observed = problem.forward(truth)
        problem.noise_energy[9] = 0.0

        waveform_loss = problem.loss
        measured = []

        def record_loss(velocity, frequency):
            value = waveform_loss(velocity, frequency)
            measured.append(value)
            return value

        with (
            patch.object(_Candidates, "propose", return_value=[candidate]),
            patch.object(problem, "loss", record_loss),
        ):
            history = fit_gaussians(field, problem, 9, iterations=1)

        trial_loss, estimated_loss = measured
        self.assertEqual(trial_loss, waveform_loss(field.velocity(), 9))
        self.assertLess(trial_loss, history[0] * 1e-6)
        self.assertGreater(estimated_loss, history[0])
        self.assertEqual(len(field), 1)
        self.assertEqual(float(field.params[0, 5]), 75.0)
        torch.testing.assert_close(field.velocity(), truth)

    def test_disabling_pruning_retains_a_weak_accepted_gaussian(self):
        problem = FWI({
            "shape": (32, 70), "dx": 10.0, "dt": 0.001, "nt": 800,
            "n_shots": 3, "n_receivers": 35, "source_frequency": 15.0,
            "bounds": (1000.0, 3000.0), "surface_velocity": 2000.0,
        })
        self.addCleanup(problem._pool.shutdown)
        initial = torch.full(problem.shape, 2000.0)
        kept = GaussianField(initial, problem.dx)
        truth = initial + 4 * gaussian(kept.z - 140, kept.x - 320, 240.0, 120.0, 0.25)
        problem.observed = problem.forward(truth)
        problem.noise_energy[6] = 0.0

        default = GaussianField(initial, problem.dx)
        fit_gaussians(default, problem, 6, iterations=1)
        fit_gaussians(kept, problem, 6, iterations=1, prune=False)
        self.assertEqual(len(default), 0)
        self.assertEqual(len(kept), 1)
        self.assertLessEqual(abs(kept.params[0, 5].item()), 5.0)
        self.assertLess(problem.loss(kept.velocity(), 6), problem.loss(default.velocity(), 6))

    def test_estimate_and_trial_must_each_clear_the_noise_threshold(self):
        initial = torch.full((3, 3), 2000.0)

        def objective(velocity):
            step = (velocity[1, 1] - 2000.0) / 100.0
            return ((1 - step) * (1 + step / 4)).square()

        velocity = initial.clone().requires_grad_()
        loss = objective(velocity)
        gradient = torch.autograd.grad(loss, velocity)[0]
        problem = Mock(bounds=(1000.0, 4000.0))
        problem.loss.side_effect = lambda v, f: float(objective(v))
        candidates = Mock()
        candidates.propose.return_value = [dict(z=10.0, x=10.0, long=80.0, short=80.0, angle=0.0)]

        # The estimated 150 m/s gains about 0.53 in loss. The 100 m/s trial gains exactly 1.
        for threshold, expected in ((0.5, [150.0]), (1.0, [100.0]), (1.01, [])):
            with self.subTest(threshold=threshold):
                field = GaussianField(initial, 10.0)
                _grow(field, problem, candidates, gradient, torch.ones_like(initial), 9, loss.item(), threshold)
                torch.testing.assert_close(field.params[:, 5], initial.new_tensor(expected))

    def test_one_ulp_loss_decrease_is_insufficient_on_clean_data(self):
        # A learnable center and a fixed residual put the best possible gain at one ULP.
        initial = torch.full((3, 3), 2000.0)
        field = GaussianField(initial, 10.0)
        eps = torch.finfo(initial.dtype).eps

        def objective(velocity):
            return 1.0 + eps * ((velocity[1, 1] - 2050.0) / 50.0).square()

        velocity = initial.clone().requires_grad_()
        loss = objective(velocity)
        gradient = torch.autograd.grad(loss, velocity)[0]
        problem = Mock(bounds=(1500.0, 6000.0))
        problem.loss.side_effect = lambda v, f: float(objective(v))
        candidates = Mock()
        candidates.propose.return_value = [
            dict(z=10.0, x=10.0, long=80.0, short=80.0, angle=0.0),
        ]
        _grow(
            field, problem, candidates, gradient, torch.ones_like(initial), 6,
            loss.item(), threshold=0.0,
        )
        self.assertEqual(len(field), 0)

    def test_cancelling_atoms_cannot_admit_a_birth_that_leaves_the_stored_grid_unchanged(self):
        initial = torch.full((3, 3), 2000.0)
        field = GaussianField(initial, 10.0)
        candidate = dict(z=10.0, x=10.0, long=80.0, short=80.0, angle=0.0)
        center = initial[1, 1]
        target = torch.nextafter(center, torch.tensor(float("inf")))
        half_ulp = float(target - center) / 2
        for amplitude in (50.0, -50.0 - half_ulp):
            field.add(**candidate, amp=amplitude)
        before = field.velocity()
        velocity = before.clone().requires_grad_()
        loss = (velocity[1, 1] - target).square()
        gradient = torch.autograd.grad(loss, velocity)[0]
        problem = Mock(bounds=(1000.0, 4000.0))
        problem.loss.side_effect = lambda grid, f: float((grid[1, 1] - target).square())
        candidates = Mock()
        candidates.propose.return_value = [candidate]

        _grow(field, problem, candidates, gradient, torch.ones_like(initial), 9, float(loss.detach()), 0.0)

        self.assertEqual(len(field), 2)
        self.assertTrue(torch.equal(field.velocity(), before))
