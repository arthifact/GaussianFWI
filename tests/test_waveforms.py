import math
import unittest

import torch

from comparisons.pixels_plain import PixelFWI as PlainPixels
from comparisons.pixels_smoothed import PixelFWI as SmoothedPixels
from src.simulation import FWI


class WaveformTests(unittest.TestCase):
    def setUp(self):
        self.survey = dict(shape=(10, 12), dx=10.0, dt=0.001, nt=128, n_shots=1, n_receivers=2,
                           source_frequency=15.0, bounds=(1000.0, 4000.0), surface_velocity=1500.0)
        self.initial = torch.full(self.survey["shape"], 1800.0)

    def problems(self, initial):
        for constructor in (FWI, PlainPixels, SmoothedPixels):
            problem = constructor(self.survey) if constructor is FWI else constructor(initial, self.survey)
            self.addCleanup(problem._pool.shutdown)
            yield problem

    def test_source_peak_above_nyquist_is_rejected_before_generating_the_wavelet(self):
        self.survey["source_frequency"] = math.nextafter(500.0, math.inf)
        for constructor in (FWI, PlainPixels, SmoothedPixels):
            with self.subTest(method=constructor.__module__):
                with self.assertRaisesRegex(ValueError, "source peak frequency.*Nyquist"):
                    if constructor is FWI:
                        constructor(self.survey)
                    else:
                        constructor(self.initial, self.survey)

    def test_dc_only_band_rejects_recordings_with_zero_mean(self):
        # The padded spectrum has zero DC. Every other bin is at or above this cutoff.
        cutoff = 1 / (2 * self.survey["nt"] * self.survey["dt"])
        alternating = (torch.arange(self.survey["nt"]) % 2 * 2 - 1).float().expand(1, 2, -1)
        for problem in self.problems(self.initial):
            with self.subTest(method=type(problem).__module__):
                problem.observed = alternating
                with self.assertRaisesRegex(ValueError, "no usable energy"):
                    if isinstance(problem, FWI):
                        problem.loss(self.initial, cutoff)
                    else:
                        problem.fit(cutoff, iterations=1)

    def test_smallest_positive_cutoff_preserves_dc_in_all_three_methods(self):
        cutoff = math.nextafter(0.0, 1.0)
        ones = torch.ones(1, 2, self.survey["nt"])
        for problem in self.problems(self.initial):
            with self.subTest(method=type(problem).__module__):
                lowpass = problem.lowpass if isinstance(problem, FWI) else problem._lowpass
                # A constant trace padded to twice its length has half its original mean.
                self.assertTrue(torch.equal(lowpass(ones, cutoff), ones * 0.5))
                if isinstance(problem, SmoothedPixels):
                    problem.observed = ones
                    self.assertTrue(math.isfinite(problem.fit(cutoff, iterations=1)[0]))
                    self.assertTrue(torch.isfinite(problem.velocity()).all())

    def test_double_precision_velocity_works_with_the_default_wavelet(self):
        model = self.initial.double()
        for problem in self.problems(model):
            with self.subTest(method=type(problem).__module__):
                recordings = problem.forward(model, accuracy=4)
                self.assertEqual(recordings.dtype, model.dtype)
                self.assertTrue(torch.isfinite(recordings).all())
                self.assertGreater(float(recordings.norm()), 0.0)

    def test_double_precision_filter_is_independent_of_the_global_default_dtype(self):
        original_dtype = torch.get_default_dtype()
        self.addCleanup(torch.set_default_dtype, original_dtype)
        recordings = torch.randn(1, 2, self.survey["nt"], dtype=torch.float64,
                                 generator=torch.Generator().manual_seed(11))
        for problem in self.problems(self.initial.double()):
            with self.subTest(method=type(problem).__module__):
                lowpass = problem.lowpass if isinstance(problem, FWI) else problem._lowpass
                torch.set_default_dtype(torch.float32)
                actual = lowpass(recordings, 125.0)
                torch.set_default_dtype(torch.float64)
                self.assertTrue(torch.equal(actual, lowpass(recordings, 125.0)))

    def test_double_precision_noise_estimates_scale_with_recording_amplitude(self):
        self.survey["shape"] = (10, 301)  # A far receiver has a usable pre-arrival window.
        self.survey["nt"] = 512
        shots = torch.randn(1, 2, self.survey["nt"], dtype=torch.float64,
                            generator=torch.Generator().manual_seed(7))
        scale = 1e-50
        for problem in self.problems(torch.full(self.survey["shape"], 1800.0, dtype=shots.dtype)):
            with self.subTest(method=type(problem).__module__):
                quiet = scale * shots
                noisy = problem.add_noise(quiet, 30)
                self.assertAlmostEqual(float((noisy - quiet).norm() / quiet.norm()), 10**(-30 / 20))
                if isinstance(problem, FWI):
                    problem.observed = shots
                    expected = problem._noise_variance(6)
                    problem.observed = quiet
                    self.assertGreater(expected, 0.0)
                    self.assertAlmostEqual(problem._noise_variance(6) / scale**2, expected)
