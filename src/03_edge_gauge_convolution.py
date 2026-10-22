"""
Chapter 3 — Convolution as a Sliding Gauge (the foundation of CNNs)
===================================================================
Machine-shop idea: slide a contour / radius gauge along a part to find where
the profile changes suddenly.

Computer-vision version: a 3x3 kernel (Sobel filter) is slid over every pixel.

        [-1  0  +1]
        [-2  0  +2]     flat metal  -> left and right cancel  -> ~0 (dark)
        [-1  0  +1]     scratch     -> big step 25 -> 128     -> strong (bright)

A Convolutional Neural Network (CNN) uses exactly this operation — the only
difference is that a CNN LEARNS the numbers inside its kernels from data
instead of having them designed by hand.

Run:  python src/03_edge_gauge_convolution.py
Output: assets/03_edge_gauge_convolution.png
"""

import matplotlib.pyplot as plt

from common import edge_response, load_labels, load_part, save_figure

N_REFERENCE = 20  # healthy parts used to calibrate the alarm level
SAFETY_FACTOR = 1.5


def main() -> None:
    df = load_labels()

    # Calibrate the alarm: strongest edge ever seen on healthy metal x safety factor
    healthy_files = df.loc[df["defect_type"] == "healthy", "filename"].tolist()
    healthy_peak = max(edge_response(load_part(f)).max() for f in healthy_files[:N_REFERENCE])
    alarm_level = healthy_peak * SAFETY_FACTOR
    print(f"Highest edge response on healthy parts: {healthy_peak:.0f} "
          f"-> alarm level set at {alarm_level:.0f}")

    healthy_fn = healthy_files[N_REFERENCE]
    scratch_fn = df.loc[df["defect_type"] == "scratch", "filename"].iloc[0]

    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    for row, (label, fname) in enumerate([("Healthy", healthy_fn), ("Scratch", scratch_fn)]):
        part = load_part(fname)
        response = edge_response(part)
        alarm = response > alarm_level
        print(f"  {label:8s} {fname}: peak response {response.max():6.0f}, "
              f"alarm pixels {int(alarm.sum())}")

        axes[row, 0].imshow(part, cmap="gray", vmin=0, vmax=255)
        axes[row, 0].set_title(f"{label} part ({fname})")
        axes[row, 1].imshow(response, cmap="hot", vmin=0, vmax=response.max() if row else alarm_level * 2)
        axes[row, 1].set_title(f"Gauge response (peak {response.max():.0f})")
        axes[row, 2].imshow(alarm, cmap="binary")
        axes[row, 2].set_title(f"Alarm > {alarm_level:.0f}: {int(alarm.sum())} px")
        for ax in axes[row]:
            ax.axis("off")

    fig.suptitle("Sobel Edge Gauge: smooth metal stays dark, scratches light up", fontweight="bold")
    out = save_figure(fig, "03_edge_gauge_convolution.png")
    print(f"[DONE] Saved: {out}")


if __name__ == "__main__":
    main()
