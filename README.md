# GaussianFWI

Gaussian background + plain pixel refinement using Deepwave.

The original `.npy` models are included in `data/`.

Edit the settings at the top of `main.py` and press **Run**.
Set `noise_db = None` for clean observations.

- `src/` — our method.
- `data/` — original velocity models: `.npy`, m/s, `(depth, x)`.
- `comparisons/` — independent methods; edit and run each file separately.

Each run resizes the model and generates shots in memory. Use matching settings
and noise seeds when comparing methods.

Each run saves a new folder with just two files:

- `result.png` — velocity models at 300 dpi; no window opens.
- `result.npz` — reference and reconstructed models, settings, loss history,
  timings and errors.

Our results go in `outputs/`; comparison results go in `comparisons/outputs/`.

Data credits: BP (BP 2004), IFP (Marmousi), SEG/EAGE (Overthrust),
SMAART (Sigsbee). SEAM Open Data © 2013 SEG Advanced Modeling Corporation
([CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)); its P-wave
velocity slice was converted from SEG-Y to NumPy.
