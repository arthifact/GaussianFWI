"""Regression checks for known, estimated and unavailable observation noise."""

import io
import json
import math
import unittest
from unittest.mock import patch

import numpy as np
import torch

from src.gaussians import GaussianField, gaussian
from src.gaussians_fit import _Candidates, fit_gaussians
from src.pixels import PixelField, fit_pixels
from src.simulation import FWI


class NoiseThresholdTests(unittest.TestCase):
    def make_problem(self, nx=16, surface_velocity=2000.0, dt=0.001):
        problem = FWI({
            "shape": (16, nx), "dx": 10.0, "dt": dt, "nt": 1000,
            "n_shots": 2, "n_receivers": 4, "source_frequency": 15.0,
            "bounds": (1000.0, 6000.0), "surface_velocity": surface_velocity,
        })
        self.addCleanup(problem._pool.shutdown)
        problem.observed = torch.randn(
            2, 4, 1000, generator=torch.Generator().manual_seed(0),
        )
        return problem

    def test_missing_windows_require_noise_information(self):
        problem = self.make_problem()
        with self.assertRaisesRegex(ValueError, "No usable pre-arrival noise windows"):
            problem.birth_threshold(6)

    def test_fast_layer_arrival_is_excluded_when_no_safe_window_remains(self):
        problem = FWI({
            "shape": (40, 70), "dx": 10.0, "dt": 0.001, "nt": 1000,
            "n_shots": 3, "n_receivers": 35, "source_frequency": 15.0,
            "bounds": (1000.0, 5500.0), "surface_velocity": 1500.0,
        })
        self.addCleanup(problem._pool.shutdown)
        velocity = torch.full(problem.shape, 4200.0)
        velocity[:3] = 1500.0
        clean = problem.forward(velocity, accuracy=8)
        far_trace = clean[0, -1].abs()
        arrival = int(torch.where(far_trace >= 0.05 * far_trace.max())[0][0]) * problem.dt
        offset = int(problem.receivers[0, -1, 1] - problem.sources[0, 0, 1]) * problem.dx
        self.assertLess(offset / problem.bounds[1], arrival)
        self.assertLess(arrival, offset / problem.surface_velocity)

        for snr in (None, 30):
            with self.subTest(snr=snr):
                problem.observed = clean if snr is None else problem.add_noise(clean, snr, seed=0)
                noise = problem.observed - clean
                for frequency in (6, 9):
                    with self.assertRaisesRegex(ValueError, "No usable pre-arrival noise windows.*Supply noise_energy"):
                        problem.birth_threshold(frequency)
                    problem.noise_energy[frequency] = float(problem.lowpass(noise, frequency).square().sum())
                    if snr is None:
                        self.assertEqual(problem.birth_threshold(frequency), 0.0)
                    else:
                        self.assertGreater(problem.birth_threshold(frequency), 0.0)

    def test_each_noise_window_excludes_samples_at_its_arrival_bound(self):
        problem = self.make_problem(nx=271, surface_velocity=1500.0)
        variance = problem._noise_variance(6)
        self.assertGreater(variance, 0.0)
        # At 6000 m/s the four offsets (0, 900, 1800, 2700 m) have these earliest samples.
        for receiver, sample in enumerate((0, 150, 300, 450)):
            problem.observed[0, receiver, sample] += 1e5
        self.assertEqual(problem._noise_variance(6), variance)
        problem.observed[0, -1, 100] += 1e5  # Inside the usable far-offset window.
        self.assertGreater(problem._noise_variance(6), variance)

    def test_explicit_clean_data_need_no_noise_windows(self):
        problem = self.make_problem()
        problem.noise_energy[6] = 0.0
        self.assertEqual(problem.birth_threshold(6), 0.0)

    def test_known_noise_requires_a_positive_gain(self):
        problem = self.make_problem()
        problem.noise_energy[6] = 1.0
        threshold = problem.birth_threshold(6)
        self.assertGreater(threshold, 0.0)
        problem.noise_energy[6] = 4.0
        self.assertAlmostEqual(problem.birth_threshold(6), 4 * threshold)

    def test_supplied_energy_overrides_available_window_estimate(self):
        problem = self.make_problem(nx=241, surface_velocity=1500.0)
        self.assertGreater(problem.birth_threshold(6), 0.0)
        problem.noise_energy[6] = 0.0
        self.assertEqual(problem.birth_threshold(6), 0.0)

    def test_saved_band_energies_reproduce_thresholds_after_reload(self):
        for nx in (16, 241):  # Without and with a usable pre-arrival window.
            with self.subTest(nx=nx):
                problem = self.make_problem(nx=nx, surface_velocity=1500.0)
                problem.noise_energy = {6: 0.0, 6.5: 1.0, 9.0: 2.0}
                expected = {f: problem.birth_threshold(f) for f in problem.noise_energy}
                # Use the same JSON-in-NPZ format as main.py.
                with io.BytesIO() as file:
                    np.savez_compressed(file, settings=json.dumps({"noise_energy": problem.noise_energy}))
                    file.seek(0)
                    with np.load(file, allow_pickle=False) as archive:
                        saved = json.loads(archive["settings"].item())
                replay = self.make_problem(nx=nx, surface_velocity=1500.0)
                replay.observed = problem.observed.clone()
                replay.noise_energy = saved["noise_energy"]
                for f, threshold in expected.items():
                    self.assertEqual(replay.birth_threshold(f), threshold)

    def test_mixed_frequency_keys_must_agree(self):
        problem = self.make_problem()
        problem.noise_energy[6] = 1.0
        expected = problem.birth_threshold(6)
        problem.noise_energy.update({"6.0": 1.0})
        self.assertEqual(problem.birth_threshold(6), expected)
        problem.noise_energy["6.0"] = 2.0
        with self.assertRaisesRegex(ValueError, "Conflicting noise energies"):
            problem.birth_threshold(6)

    def test_float32_cutoff_does_not_merge_nearby_noise_keys(self):
        problem = self.make_problem()
        problem.noise_energy = {"6": 1.0, "6.0000001": 2.0}
        expected = problem.birth_threshold(6.0)
        for cutoff in (np.float32(6.0), torch.tensor(6.0, dtype=torch.float32)):
            with self.subTest(scalar=type(cutoff).__name__):
                self.assertEqual(problem.birth_threshold(cutoff), expected)

    def test_each_frequency_needs_its_own_noise_information(self):
        problem = self.make_problem()
        problem.noise_energy[6] = 0.0
        with self.assertRaisesRegex(ValueError, "No usable pre-arrival noise windows"):
            problem.birth_threshold(9)

    def test_replacing_observations_clears_old_noise_information(self):
        problem = self.make_problem()
        problem.noise_energy[6] = 0.0
        problem.observed = problem.observed.clone()
        self.assertEqual(problem.noise_energy, {})
        with self.assertRaisesRegex(ValueError, "No usable pre-arrival noise windows"):
            problem.birth_threshold(6)

    def test_scaling_samples_invalidates_all_bands_and_fresh_energy_restores_threshold(self):
        problem = self.make_problem()
        problem.noise_energy = {6: 1.0, 9: 0.0}
        before = problem.birth_threshold(6)
        alias = problem.observed.detach()
        alias.mul_(2)
        for frequency in (6, 9):
            with self.assertRaisesRegex(ValueError, "No usable pre-arrival noise windows"):
                problem.birth_threshold(frequency)
        self.assertEqual(problem.noise_energy, {})
        problem.noise_energy = {"6": 4.0}
        self.assertEqual(problem.birth_threshold(6), before)

    def test_numpy_edit_uses_quiet_windows_from_the_current_recordings(self):
        problem = self.make_problem(nx=241, surface_velocity=1500.0)
        problem.noise_energy = {"6": 0.0}
        self.assertEqual(problem.birth_threshold(6), 0.0)
        # Change samples through shared memory without changing the total energy.
        problem.observed.numpy()[0] *= -1
        reference = self.make_problem(nx=241, surface_velocity=1500.0)
        reference.observed = problem.observed.clone()
        expected = reference.birth_threshold(6)
        self.assertGreater(expected, 0.0)
        self.assertEqual(problem.birth_threshold(6), expected)
        self.assertEqual(problem.noise_energy, {})

    def test_refreshing_one_band_after_an_edit_cannot_reuse_other_old_bands(self):
        problem = self.make_problem()
        problem.noise_energy = {6: 1.0, 9: 2.0}
        before = problem.birth_threshold(6)
        problem.observed.mul_(2)
        problem.noise_energy[6] = 4.0
        self.assertEqual(problem.birth_threshold(6), before)
        with self.assertRaisesRegex(ValueError, "No usable pre-arrival noise windows"):
            problem.birth_threshold(9)

    def test_later_growth_round_rechecks_noise_after_samples_change(self):
        problem = self.make_problem()
        problem.noise_energy[6] = 1.0
        field = GaussianField(torch.full(problem.shape, 2000.0), problem.dx)
        calls = 0

        def gradient(velocity, frequency):
            nonlocal calls
            calls += 1
            if calls == 2:  # After the first birth round, before the next one.
                problem.observed.mul_(2)
            return 0.0, torch.zeros_like(velocity)

        with patch.object(problem, "gradient", side_effect=gradient):
            with self.assertRaisesRegex(ValueError, "No usable pre-arrival noise windows"):
                fit_gaussians(field, problem, 6, iterations=11)

    def test_invalid_supplied_energy_is_rejected(self):
        problem = self.make_problem()
        for energy in (-1.0, float("nan"), float("inf")):
            with self.subTest(energy=energy):
                problem.noise_energy[6] = energy
                with self.assertRaisesRegex(ValueError, "finite and nonnegative"):
                    problem.birth_threshold(6)

    def test_invalid_frequency_is_rejected(self):
        problem = self.make_problem()
        for frequency in (0.0, -1.0, float("nan"), float("inf")):
            with self.subTest(frequency=frequency):
                with self.assertRaisesRegex(ValueError, "finite and positive"):
                    problem.birth_threshold(frequency)

    def test_two_all_pass_cutoffs_cannot_produce_different_birth_thresholds(self):
        problem = self.make_problem()
        cutoffs = (1000.0, 2000.0)
        problem.noise_energy = dict.fromkeys(cutoffs, 1.0)
        # Both tapers used to start above the highest recorded bin and pass identical data.
        highest_bin = torch.fft.rfftfreq(2 * problem.nt, problem.dt)[-1].item()
        self.assertLess(highest_bin, 0.7 * min(cutoffs))
        for cutoff in cutoffs:
            with self.subTest(cutoff=cutoff, operation="threshold"):
                with self.assertRaisesRegex(ValueError, "Nyquist.*500"):
                    problem.birth_threshold(cutoff)
            with self.subTest(cutoff=cutoff, operation="filter"):
                with self.assertRaisesRegex(ValueError, "Nyquist.*500"):
                    problem.lowpass(problem.observed, cutoff)

    def test_identical_sampled_bands_make_the_same_birth_decision(self):
        # At 128 samples the padded FFT bins are 3.90625 Hz apart. Both cutoffs
        # pass only DC and the first positive/negative frequency pair.
        problem = FWI({
            "shape": (16, 30), "dx": 10.0, "dt": 0.001, "nt": 128,
            "n_shots": 3, "n_receivers": 10, "source_frequency": 15.0,
            "bounds": (1000.0, 5500.0), "surface_velocity": 2000.0,
        })
        self.addCleanup(problem._pool.shutdown)
        initial = torch.full(problem.shape, 2000.0)
        field = GaussianField(initial, problem.dx)
        short = problem.dx * 2**3.5
        candidate = dict(z=0.0, x=150.0, long=2*short, short=short, angle=0.0)
        shape = gaussian(field.z, field.x - candidate["x"], 2*short, short, 0.0)
        clean = problem.forward(initial + 43.75 * shape)
        problem.observed = problem.add_noise(clean, 30, seed=0)
        noise = problem.observed - clean
        problem.noise_energy = {f: float(problem.lowpass(noise, f).square().sum()) for f in (6, 7)}
        data = problem.lowpass(problem.observed, 6)
        self.assertTrue(torch.equal(data, problem.lowpass(problem.observed, 7)))
        self.assertEqual(problem.noise_energy[6], problem.noise_energy[7])
        self.assertEqual(problem.birth_threshold(6), problem.birth_threshold(7))

        # Three of 256 Fourier bins contribute per trace, with no extra
        # independent samples attributed to the zero padding.
        effective = data.numel() * 3 / 256
        expected = 6 * math.log(effective) * problem.noise_energy[6] / (effective * data.square().sum().item())
        self.assertAlmostEqual(problem.birth_threshold(6), expected, places=10)
        with patch.object(_Candidates, "propose", return_value=[candidate]):
            fit_gaussians(field, problem, 6, iterations=1)
            second = GaussianField(initial, problem.dx)
            fit_gaussians(second, problem, 7, iterations=1)
        self.assertEqual(len(field), 1)
        self.assertTrue(torch.equal(field.params, second.params))

    def test_cutoff_at_nyquist_is_valid_but_the_next_float_is_not(self):
        for dt, nyquist in ((0.001, 500.0), (0.004, 125.0)):
            with self.subTest(dt=dt):
                problem = self.make_problem(dt=dt)
                problem.noise_energy[nyquist] = 1.0
                self.assertGreater(problem.birth_threshold(nyquist), 0.0)
                with self.assertRaisesRegex(ValueError, "Nyquist"):
                    problem.birth_threshold(math.nextafter(nyquist, math.inf))

    def test_dc_only_filters_have_the_same_noise_penalty(self):
        problem = self.make_problem()
        cutoffs = (0.05, 0.2)
        problem.noise_energy = dict.fromkeys(cutoffs, 1.0)
        self.assertTrue(torch.equal(problem.lowpass(problem.observed, cutoffs[0]),
                                    problem.lowpass(problem.observed, cutoffs[1])))
        first, second = (problem.birth_threshold(f) for f in cutoffs)
        self.assertGreater(first, 0.0)
        self.assertEqual(first, second)

    def test_equivalent_filters_agree_when_noise_is_estimated_from_windows(self):
        problem = self.make_problem(nx=241)
        self.assertTrue(torch.equal(problem.lowpass(problem.observed, 0.8), problem.lowpass(problem.observed, 0.9)))
        first = problem.birth_threshold(0.8)
        self.assertGreater(first, 0.0)
        self.assertEqual(first, problem.birth_threshold(0.9))

    def test_noisy_band_with_one_effective_sample_cannot_default_to_zero_penalty(self):
        problem = FWI({
            "shape": (8, 8), "dx": 10.0, "dt": 0.001, "nt": 128,
            "n_shots": 1, "n_receivers": 2, "source_frequency": 15.0,
            "bounds": (1000.0, 5500.0), "surface_velocity": 1500.0,
        })
        self.addCleanup(problem._pool.shutdown)
        problem.observed = torch.ones(1, 2, problem.nt)
        problem.noise_energy[1] = 1.0
        with self.assertRaisesRegex(ValueError, "Too few effective samples"):
            problem.birth_threshold(1)
        problem.noise_energy[1] = 0.0
        self.assertEqual(problem.birth_threshold(1), 0.0)

    def test_float32_cutoff_cannot_round_the_survey_limit_up(self):
        problem = self.make_problem(dt=0.003)
        nyquist = 0.5 / problem.dt
        for cutoff in (np.float32(nyquist), torch.tensor(nyquist, dtype=torch.float32)):
            with self.subTest(scalar=type(cutoff).__name__):
                self.assertGreater(float(cutoff), nyquist)
                with self.assertRaisesRegex(ValueError, "Nyquist"):
                    problem.lowpass(problem.observed, cutoff)

    def test_fitters_reject_invalid_cutoffs_even_for_empty_stages(self):
        problem = self.make_problem()
        initial = torch.full(problem.shape, 2000.0)
        fitters = ((fit_gaussians, GaussianField(initial, problem.dx)), (fit_pixels, PixelField(initial)))
        for fit, model in fitters:
            with self.subTest(fitter=fit.__name__):
                self.assertEqual(fit(model, problem, 500.0, iterations=0), [])
                with self.assertRaisesRegex(ValueError, "Nyquist"):
                    fit(model, problem, 1000.0, iterations=0)


if __name__ == "__main__":
    unittest.main()
