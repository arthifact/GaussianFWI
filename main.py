"""Edit the experiment settings below, then press Run."""

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
noise_db = 30               # 'None' for clean observations
noise_seed = 0
gaussian_frequencies = (6, 9)  # Low-pass cutoffs, Hz
pixel_frequencies = (13, 20, 30)
gaussian_iterations = 325     # Per frequency
pixel_iterations = 350

# Prepare our survey and observations.
torch.set_num_threads(1)  # Shots already run in parallel.
true = prepare_model(model, survey["shape"])
# Cover the selected model's velocities; use the surface speed for the noise window.
survey["bounds"] = (1000.0, max(5500.0, float(true.max())))
survey["surface_velocity"] = float(true[0].max())
problem = FWI(survey)
observed = problem.forward(true, accuracy=8)
if noise_db is not None:
    observed = problem.add_noise(observed, noise_db, noise_seed)
problem.observed = observed

# Start from a 1500–3000 m/s depth ramp.
depth = torch.arange(true.shape[0])[:, None] * survey["dx"]
initial = (1500 + 1500 * depth / depth.max()).expand_as(true).clone()

# Keep a snapshot after each Gaussian frequency.
gaussians = GaussianField(initial, survey["dx"])
gaussian_backgrounds = {}
progress = []
for frequency in gaussian_frequencies:
    started = perf_counter()
    losses = fit_gaussians(gaussians, problem, frequency, iterations=gaussian_iterations)
    seconds = perf_counter() - started
    snapshot = gaussians.velocity().clamp(*problem.bounds).detach().clone()
    gaussian_backgrounds[frequency] = snapshot
    error = snapshot - true
    progress.append({
        "stage": "gaussians", "frequency": frequency, "loss": losses,
        "seconds": seconds, "gaussians": len(gaussians),
        "relative_l2": float(error.norm() / true.norm()),
        "rmse": float(error.square().mean().sqrt()),
    })
background = gaussians.velocity().clamp(*problem.bounds).detach()

# Pixel details on the fixed background.
pixels = PixelField(background)
for frequency in pixel_frequencies:
    started = perf_counter()
    losses = fit_pixels(pixels, problem, frequency, iterations=pixel_iterations)
    seconds = perf_counter() - started
    error = pixels.velocity().detach() - true
    progress.append({
        "stage": "pixels", "frequency": frequency, "loss": losses,
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
    "gaussian_frequencies": gaussian_frequencies, "pixel_frequencies": pixel_frequencies,
    "gaussian_iterations": gaussian_iterations, "pixel_iterations": pixel_iterations,
    "initial_velocity": [1500.0, 3000.0],
    "versions": {name: version(name) for name in ("torch", "deepwave", "numpy")},
}
np.savez_compressed(
    output / "result.npz",
    true=true.numpy(), final=final.numpy(),
    gaussian_frequencies=gaussian_frequencies,
    gaussian_backgrounds=torch.stack(list(gaussian_backgrounds.values())).numpy(),
    settings=json.dumps(settings), progress=json.dumps(progress),
)
save_models(true, gaussian_backgrounds, final, output / "result.png")
print(f"Saved {output}")
