# GaussianFWI

Gaussian background + plain pixel refinement using Deepwave.

The original `.npy` models are included in `data/`.

Edit the settings at the top of `main.py` and press **Run**.
Set `noise_db = None` for clean observations.
Run the regression tests with `uv run --frozen python -m unittest discover -q`.
The source peak frequency and stage cutoffs must be positive and no higher than `0.5 / dt` Hz
(500 Hz at the default 1 ms sample interval). A cutoff exactly at Nyquist is valid.
For a matched pruning comparison, run `python main.py --model overthrust --noise-seed 1`
and repeat with `--no-pruning`. The default removes Gaussians with amplitudes
at or below 5 m/s in magnitude at growth rounds.

The pruning comparison uses 70 × 70 Marmousi and Overthrust models at 10 m
spacing, five training shots, 30 dB noise and seeds 0 and 1. The four held-out
shots are at horizontal grid indices 8, 26, 43 and 61, with the same receivers
as the training survey. Held-out loss is the sum of squared filtered residuals
divided by the sum of squared filtered clean recordings.
Clean reference recordings use an eighth-order spatial stencil and predictions
use fourth order.

Each pair below lists the 5 m/s cutoff first and pruning disabled second.
The 9 Hz results are from the Gaussian background. The 30 Hz results and
final velocity RMSE are from the pixel refinement.

| Model | Noise seed | Gaussians at 9 Hz | Held-out 9 Hz loss (×10⁻³) | Held-out 30 Hz loss (×10⁻³) | Final velocity RMSE (m/s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Marmousi | 0 | 30 / 75 | 1.955 / 2.423 | 1.060 / 1.045 | 506.8 / 598.6 |
| Marmousi | 1 | 28 / 65 | 1.280 / 2.858 | 1.051 / 1.069 | 579.0 / 465.7 |
| Overthrust | 0 | 30 / 47 | 0.510 / 0.692 | 0.446 / 0.444 | 315.2 / 313.7 |
| Overthrust | 1 | 46 / 60 | 0.613 / 0.586 | 0.436 / 0.435 | 321.3 / 327.3 |

The cutoff kept fewer Gaussians in all four comparisons. Held-out waveform
loss improved in three at 9 Hz and one at 30 Hz. Final velocity RMSE improved
in two. The default reflects the preference for a compact Gaussian background.

- `src/` — our method.
- `data/` — original velocity models: `.npy`, m/s, `(depth, x)`.
- `comparisons/` — independent methods; edit and run each file separately.

Each run resizes the model and generates shots in memory. Use matching settings
and noise seeds when comparing methods.

The low-to-high frequency schedule follows multiscale waveform inversion.
Recoverable spatial scales depend on frequency and acquisition geometry. Here
a quarter of the local wavelength sets the minimum proposed Gaussian width
(one standard deviation). Later velocity updates can change that wavelength.
[Bunks et al. (1995)](https://doi.org/10.1190/1.1443880) · [Sirgue and Pratt (2004)](https://doi.org/10.1190/1.1649391)

Chen et al. used a fixed Gaussian basis to estimate long-wavelength FWI
starting models. Here the centers, widths, orientations and amplitudes can
all change during fitting. [Chen et al. (2020)](https://doi.org/10.1190/segam2020-3418905.1)

The fitter ranks Gaussian shapes using the velocity gradient and source
illumination. Before adding a row, it evaluates the nonlinear waveform loss on
the bounded grid rebuilt from the proposed parameters. A birth must reduce
that loss beyond roundoff and clear the noise threshold.

Gaussian growth uses the known added-noise energy in each frequency band, with
zero added noise for clean runs. Schwarz's model dimension penalty motivates
charging for six new parameters. For these filtered traces, we approximate the
effective sample count by scaling the recorded sample count by the mean squared
filter weight on the padded FFT grid. This approximation treats traces as
independent. [Schwarz (1978)](https://doi.org/10.1214/aos/1176344136)

Positive frequencies and their negative-frequency partners contribute equally.
DC and Nyquist each contribute once. [One-sided spectral power](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.periodogram.html)

Identical sampled filters and noise energies therefore yield the same threshold.
A noisy band needs more than one effective sample for this calculation.

New archives record the method as `sampled_filter_power`. To replay its
threshold, assign the same recordings to `problem.observed`,
then set `problem.noise_energy = settings["noise_energy"]` from the decoded run settings.
For older archives without `birth_threshold_method`, use the Gaussian stage's
recorded `birth_threshold` when available.
Changing the recording samples invalidates the supplied energies for all bands.
For recordings aligned to shot time, the fitter can estimate noise before the
first arrival. Each trace contributes only whole sample intervals before its
source–receiver distance divided by the maximum propagation speed. The fitter
uses the larger of the survey's upper velocity bound and surface speed. This
ceiling must cover the whole medium because a faster buried layer can send
energy to a receiver ahead of the direct surface arrival. [Layered travel times](https://earthweb.ess.washington.edu/vidale/Reprints/BSSA/1988_Vidale.pdf) · [Seismic noise windows](https://doi.org/10.1007/s10518-017-0131-9)

The fitter requires at least 0.3 s and two eligible samples on a trace.
Supply band noise energies for the current recordings if no trace qualifies,
the speed ceiling is uncertain, or a time shift or filter has moved signal
ahead of its physical arrival.

Each `main.py` run saves two files in a new folder.

- `result.png` — velocity models at 300 dpi; no window opens.
- `result.npz` — reference, initial and reconstructed models, settings,
  loss histories, timings and errors. For Gaussian stage index `i` (starting
  at zero), `gaussian_params_<i>` stores each surviving Gaussian as
  `[z_m, x_m, log(width_along_angle_m), log(perpendicular_width_m), angle_rad, amplitude_m/s]`.
  The angle is measured from horizontal toward depth. The first axis starts
  as the longer one, but either width can become larger during fitting.
  Rebuild the field on `initial` with the survey spacing and clip it to the
  survey bounds to reproduce `gaussian_backgrounds[i]`. Stage indices also
  distinguish repeated frequencies. Pixel stage grids are in `pixel_models`.
  Each progress entry keeps the pre-update `loss` history and a separately
  measured `post_update_loss` on the resulting grid (null for zero updates).

Our results go in `outputs/`. The independent pixel comparisons save their
reference and final grids, settings and per-stage metrics in `comparisons/outputs/`.

Data credits: BP (BP 2004), IFP (Marmousi), SEG/EAGE (Overthrust),
SMAART (Sigsbee). SEAM Open Data © 2013 SEG Advanced Modeling Corporation
([CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)); its P-wave
velocity slice was converted from SEG-Y to NumPy.
