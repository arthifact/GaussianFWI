"""Regression checks for Gaussian widths at covered and uncovered centers."""

import math
import unittest

import torch

from src.gaussians import GaussianField, gaussian
from src.gaussians_fit import _Candidates, fit_gaussians
from src.simulation import FWI


class GaussianShortlistTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(1)

    def shortlist(self, velocity, dx, frequency, max_velocity):
        field = GaussianField(velocity, dx)
        bank = _Candidates(field, frequency, max_velocity)
        # This broad gradient gives wider shapes a high score.
        return bank.propose(1.0 + field.x / 1000.0, torch.ones_like(velocity), velocity, frequency)

    def test_covered_centers_preserve_the_original_ranked_shortlist(self):
        for dx, frequency, speed in ((10.0, 6, 2500.0), (10.0, 6, 3840.0), (5.0, 9, 2880.0)):
            with self.subTest(dx=dx, frequency=frequency, speed=speed):
                velocity = torch.full((30, 70), speed)
                # At this ceiling the bank contains exactly the nine original widths.
                original_ceiling = 4 * frequency * 16 * dx
                original = self.shortlist(velocity, dx, frequency, original_ceiling)
                self.assertTrue(original)
                self.assertEqual(self.shortlist(velocity, dx, frequency, 5500.0), original)

    def test_unrepresentable_gaussian_widths_are_rejected_before_building_the_bank(self):
        field = GaussianField(torch.full((2, 2), 2000.0), 10.0)
        for cutoff in (1e-36, math.nextafter(0.0, 1.0)):
            with self.subTest(cutoff=cutoff):
                with self.assertRaisesRegex(ValueError, "Gaussian widths"):
                    _Candidates(field, cutoff, 5500.0)

    def test_mixed_model_only_uses_extension_widths_at_fast_centers(self):
        velocity = torch.full((30, 70), 2500.0)
        velocity[:, :10] = 4000.0
        proposals = self.shortlist(velocity, 10.0, 6, 5500.0)
        speeds = set()
        for p in proposals:
            speed = velocity[round(p["z"] / 10), round(p["x"] / 10)].item()
            speeds.add(speed)
            self.assertGreaterEqual(p["short"], speed / 24)
            if speed == 2500.0:
                self.assertLessEqual(p["short"], 160.0)
        self.assertEqual(speeds, {2500.0, 4000.0})

    def test_fast_background_can_still_fit_a_broad_anomaly(self):
        initial = torch.full((70, 70), 4000.0)
        field = GaussianField(initial, 10.0)
        truth = initial + 250 * gaussian(field.z - 240, field.x - 350, 320.0, 200.0, 0.25)
        problem = FWI({
            "shape": initial.shape, "dx": 10.0, "dt": 0.001, "nt": 1000,
            "n_shots": 5, "n_receivers": None, "source_frequency": 15.0,
            "bounds": (1000.0, 5500.0), "surface_velocity": 4000.0,
        })
        self.addCleanup(problem._pool.shutdown)
        problem.observed = problem.forward(truth)
        problem.noise_energy[6] = 0.0
        before = problem.loss(initial, 6)
        fit_gaussians(field, problem, 6, iterations=1)
        self.assertGreater(len(field), 0)
        self.assertTrue((field.params[:, 3].exp() >= 4000.0 / 24).all())
        self.assertLess(problem.loss(field.velocity().clamp(*problem.bounds), 6), before)
