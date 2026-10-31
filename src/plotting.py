"""Save the true model, Gaussian stages and final pixel refinement as a PNG."""

import matplotlib

matplotlib.use("Agg")  # Save figures without opening a window.
import matplotlib.pyplot as plt


def save_models(true, gaussian_backgrounds, final, path):
    models = [("True", true)]
    for frequency, background in gaussian_backgrounds:
        models.append((f"Gaussian background\n{frequency:g} Hz", background))
    models.append(("Background + pixels", final))

    fig, axes = plt.subplots(
        1, len(models), figsize=(3 * len(models), 4),
        layout="constrained", squeeze=False,
    )
    for ax, (title, model) in zip(axes.flat, models):
        image = ax.imshow(model.detach().cpu(), cmap="jet", vmin=float(true.min()), vmax=float(true.max()))
        ax.set(title=title, xlabel="x (cells)")
    axes[0, 0].set_ylabel("Depth (cells)")
    fig.colorbar(image, ax=list(axes.flat), label="Velocity (m/s)", shrink=0.8)
    fig.savefig(path, dpi=300)
    plt.close(fig)
