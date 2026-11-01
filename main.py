"""Edit the experiment settings below, then press Run."""

import argparse
import json
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from src.data_prepare import prepare_model
from src.gaussians import GaussianField
from src.gaussians_fit import fit_gaussians
from src.pixels import PixelField, fit_pixels
from src.simulation import FWI
from src.plotting import save_models

# Experiment settings.
model = "marmousi"  # marmousi, bp_salt, overthrust, seam, sigsbee
survey = {
    "shape": (70, 70),       # depth cells, horizontal cells
    "dx": 10.0,             # cell spacing, metres
    "n_shots": 5,
    "n_receivers": None,    # None means one receiver per surface cell
    "dt": 0.001,            # seconds per sample
    "nt": 1000,             # samples per recording
    "source_frequency": 15.0,  # Ricker wavelet peak frequency, Hz
}
noise_db = 30               # None for clean observations
noise_seed = 0
gaussian_frequencies = (6, 9)  # Low-pass cutoffs, Hz
pixel_frequencies = (13, 20, 30)
gaussian_iterations = 325     # Per frequency
pixel_iterations = 350
prune_gaussians = True       # Remove Gaussians with amplitude magnitude at or below 5 m/s.

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=model)
    parser.add_argument("--noise-seed", type=int, default=noise_seed)
    parser.add_argument("--pruning", action=argparse.BooleanOptionalAction, default=prune_gaussians)
    args = parser.parse_args()
    model, noise_seed, prune_gaussians = args.model, args.noise_seed, args.pruning

    # Prepare our survey and observations.
    torch.set_num_threads(1)  # Shots already run in parallel.
    true = prepare_model(model, survey["shape"])
    # Cover propagation speeds throughout the selected model, including buried layers.
    survey["bounds"] = (1000.0, max(5500.0, float(true.max())))
    survey["surface_velocity"] = float(true[0].max())
    problem = FWI(survey)
    for frequency in (*gaussian_frequencies, *pixel_frequencies):
        problem.validate_cutoff(frequency)
    clean = problem.forward(true, accuracy=8)
    observed = clean
    if noise_db is not None:
        observed = problem.add_noise(clean, noise_db, noise_seed)
    problem.observed = observed
    # Retain only the added-noise energy in each Gaussian frequency band.
    noise = observed - clean
    problem.noise_energy = {
        frequency: float(problem.lowpass(noise, frequency).square().sum())
        for frequency in dict.fromkeys(gaussian_frequencies)
    }
    del clean, noise

    # Start from a 1500–3000 m/s depth ramp.
    depth = torch.arange(true.shape[0])[:, None] * survey["dx"]
    initial = (1500 + 1500 * depth / depth.max()).expand_as(true).clone()

    # Keep a snapshot after each Gaussian frequency.
    gaussians = GaussianField(initial, survey["dx"])
    gaussian_backgrounds = []
    gaussian_params = {}
    progress = []
    for stage_index, frequency in enumerate(gaussian_frequencies):
        started = perf_counter()
        losses = fit_gaussians(gaussians, problem, frequency, iterations=gaussian_iterations, prune=prune_gaussians)
        seconds = perf_counter() - started
        snapshot = gaussians.velocity().clamp(*problem.bounds).detach().clone()
        gaussian_backgrounds.append((frequency, snapshot))
        # Stage indices preserve separate parameter rows for repeated cutoffs.
        gaussian_params[f"gaussian_params_{stage_index}"] = gaussians.params.detach().numpy().copy()
        error = snapshot - true
        progress.append({
            "stage": "gaussians", "frequency": frequency, "loss": losses,
            "post_update_loss": problem.loss(snapshot, frequency) if losses else None,
            "seconds": seconds, "gaussians": len(gaussians),
            "birth_threshold": problem.birth_threshold(frequency) if gaussian_iterations else None,
            "relative_l2": float(error.norm() / true.norm()),
            "rmse": float(error.square().mean().sqrt()),
        })
    background = gaussians.velocity().clamp(*problem.bounds).detach()

    # Pixel details on the fixed background.
    pixels = PixelField(background)
    pixel_models = []
    for frequency in pixel_frequencies:
        started = perf_counter()
        losses = fit_pixels(pixels, problem, frequency, iterations=pixel_iterations)
        seconds = perf_counter() - started
        snapshot = pixels.velocity().detach()
        pixel_models.append(snapshot)
        error = snapshot - true
        progress.append({
            "stage": "pixels", "frequency": frequency, "loss": losses,
            "post_update_loss": problem.loss(snapshot, frequency) if losses else None,
            "seconds": seconds,
            "relative_l2": float(error.norm() / true.norm()),
            "rmse": float(error.square().mean().sqrt()),
        })
    final = pixels.velocity().detach()

    # Keep each completed run together: one figure and one compressed results file.
    root = Path(__file__).resolve().parent
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%fZ")
    output = root / "outputs" / f"{model}_{run_id}"
    output.mkdir(parents=True, exist_ok=False)
    settings = {
        "method": "gaussian_background_pixels",
        "model": model, "survey": survey, "noise_db": noise_db, "noise_seed": noise_seed,
        "noise_energy": problem.noise_energy, "noise_energy_source": "synthetic_added_noise",
        "birth_threshold_method": "sampled_filter_power", "gaussian_pruning": prune_gaussians,
        "gaussian_frequencies": gaussian_frequencies, "pixel_frequencies": pixel_frequencies,
        "gaussian_iterations": gaussian_iterations, "pixel_iterations": pixel_iterations,
        "initial_velocity": [1500.0, 3000.0],
        "versions": {name: version(name) for name in ("torch", "deepwave", "numpy")},
    }
    empty_stages = true.new_empty((0, *true.shape))
    saved_backgrounds = torch.stack([grid for _, grid in gaussian_backgrounds]) if gaussian_backgrounds else empty_stages
    saved_pixels = torch.stack(pixel_models) if pixel_models else empty_stages
    np.savez_compressed(
        output / "result.npz",
        true=true.numpy(), initial=initial.numpy(), final=final.numpy(),
        gaussian_frequencies=gaussian_frequencies,
        gaussian_backgrounds=saved_backgrounds.numpy(), pixel_models=saved_pixels.numpy(),
        settings=json.dumps(settings), progress=json.dumps(progress), **gaussian_params,
    )
    save_models(true, gaussian_backgrounds, final, output / "result.png")
    print(f"Saved {output}")
