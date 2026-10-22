"""
Chapter 2 — Statistical Surface Inspection (the 3-sigma tolerance rule)
=======================================================================
Machine-shop idea: a tolerance band is set from GOOD parts, then every new
part is checked against it.

Computer-vision version:
    1. Measure the normal gray-level mean (mu) and spread (sigma) on healthy parts.
    2. Flag any pixel of an incoming part where |pixel - mu| > 3 * sigma.

The tolerance is learned from healthy reference parts only — if it were
computed from the defective part itself, the defect would inflate sigma and
partly hide itself.

Run:  python src/02_statistical_inspection.py
Output: assets/02_statistical_inspection.png
"""

import matplotlib.pyplot as plt
import numpy as np

from common import load_labels, load_part, save_figure

SIGMA_LIMIT = 3.0
N_REFERENCE = 20  # healthy parts used to set the tolerance band


def main() -> None:
    df = load_labels()

    healthy_files = df.loc[df["defect_type"] == "healthy", "filename"].tolist()
    reference = np.stack([load_part(f) for f in healthy_files[:N_REFERENCE]])
    mu, sigma = reference.mean(), reference.std()
    lower, upper = mu - SIGMA_LIMIT * sigma, mu + SIGMA_LIMIT * sigma
    print(f"Tolerance from {N_REFERENCE} healthy parts: "
          f"mu = {mu:.1f}, sigma = {sigma:.1f}  ->  band [{lower:.0f}, {upper:.0f}]")

    # One unseen part of each type
    samples = {"healthy": healthy_files[N_REFERENCE]}
    for defect in ["chatter", "scratch", "pitting"]:
        samples[defect] = df.loc[df["defect_type"] == defect, "filename"].iloc[0]

    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    for col, (defect, fname) in enumerate(samples.items()):
        part = load_part(fname)
        mask = np.abs(part - mu) > SIGMA_LIMIT * sigma
        n_bad = int(mask.sum())
        print(f"  {defect:8s} {fname}: {n_bad:5d} out-of-tolerance pixels")

        axes[0, col].imshow(part, cmap="gray", vmin=0, vmax=255)
        axes[0, col].set_title(f"{defect.capitalize()} ({fname})", fontsize=10)
        axes[1, col].imshow(part, cmap="gray", vmin=0, vmax=255)
        axes[1, col].imshow(np.ma.masked_where(~mask, mask), cmap="autumn", alpha=0.9)
        axes[1, col].set_title(f"Out of tolerance: {n_bad} px", fontsize=10)
        for ax in axes[:, col]:
            ax.axis("off")

    fig.suptitle(
        f"3-Sigma Inspection  (tolerance band {lower:.0f}–{upper:.0f} learned from healthy parts)",
        fontweight="bold",
    )
    out = save_figure(fig, "02_statistical_inspection.png")
    print(f"[DONE] Saved: {out}")


if __name__ == "__main__":
    main()
