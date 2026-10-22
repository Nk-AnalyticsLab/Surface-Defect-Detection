"""
Chapter 6 — "Golden Master" Unsupervised Anomaly Detection (PCA)
================================================================
In real factories defects are rare and new defect types appear without
warning, so labelled defect images are often unavailable. The Golden Master
approach learns ONLY what a good part looks like:

1. Train PCA on 120 healthy parts — it memorises the normal surface texture.
2. For any incoming part, reconstruct it through the Golden Master.
       residual = |actual part - reconstruction|
   Healthy metal reconstructs well (residual ~ 0); a defect cannot be
   reconstructed, so it remains in the residual and glows.
3. Anomaly score = mean residual. The PASS/FAIL limit is set from the
   training (healthy) scores only:  limit = mean + 3 * std.
4. Evaluate on the 180 parts never seen in training
   (30 healthy + 150 defective). Defect labels are used ONLY for evaluation.

Run:  python src/06_golden_master_anomaly.py
Output: assets/06_golden_master_anomaly.png
"""

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score

from common import CLASS_NAMES, IMAGES_DIR, RANDOM_STATE, load_labels, save_figure

IMG_SIZE = 64        # downsample for speed
N_COMPONENTS = 12    # too many components -> the model starts "reconstructing" noise too
N_TRAIN_HEALTHY = 120


def load_vector(filename: str) -> np.ndarray:
    img = Image.open(IMAGES_DIR / filename).convert("L").resize((IMG_SIZE, IMG_SIZE))
    return np.asarray(img, dtype=np.float32).ravel()


def main() -> None:
    df = load_labels()

    healthy = df[df["defect_type"] == "healthy"]
    train_files = healthy["filename"].iloc[:N_TRAIN_HEALTHY]
    test_df = df[~df["filename"].isin(train_files)].reset_index(drop=True)

    X_train = np.stack([load_vector(f) for f in train_files])
    X_test = np.stack([load_vector(f) for f in test_df["filename"]])

    master = PCA(n_components=N_COMPONENTS, random_state=RANDOM_STATE).fit(X_train)
    print(f"Golden Master trained on {len(X_train)} healthy parts "
          f"({N_COMPONENTS} components, {master.explained_variance_ratio_.sum():.1%} variance)")

    def residuals(X):
        return np.abs(X - master.inverse_transform(master.transform(X)))

    train_scores = residuals(X_train).mean(axis=1)
    test_res = residuals(X_test)
    test_scores = test_res.mean(axis=1)
    limit = train_scores.mean() + 3 * train_scores.std()

    y_true = test_df["is_defective"].to_numpy()
    flagged = test_scores > limit
    auc = roc_auc_score(y_true, test_scores)
    print(f"Anomaly limit (mean + 3 std of healthy training scores): {limit:.2f}")
    print(f"ROC-AUC on {len(test_df)} unseen parts: {auc:.3f}")
    for defect in CLASS_NAMES:
        sel = (test_df["defect_type"] == defect).to_numpy()
        rate = flagged[sel].mean()
        what = "false-alarm rate" if defect == "healthy" else "detection rate"
        print(f"  {defect:8s} n={sel.sum():3d}  {what}: {rate:.1%}")

    # ---- Figure: 4 example parts (rows) + score distribution ----------------
    fig = plt.figure(figsize=(14, 13))
    grid = fig.add_gridspec(5, 3, height_ratios=[1, 1, 1, 1, 1.1])

    for row, defect in enumerate(CLASS_NAMES):
        i = int(np.flatnonzero(test_df["defect_type"] == defect)[0])
        actual = X_test[i].reshape(IMG_SIZE, IMG_SIZE)
        recon = master.inverse_transform(master.transform(X_test[i:i + 1])).reshape(IMG_SIZE, IMG_SIZE)
        verdict = "FAIL" if flagged[i] else "PASS"
        panels = [
            (actual, "gray", (0, 255), f"{defect.capitalize()} ({test_df.at[i, 'filename']})"),
            (recon, "gray", (0, 255), "Golden Master reconstruction"),
            (test_res[i].reshape(IMG_SIZE, IMG_SIZE), "hot", (0, 50),
             f"Residual — score {test_scores[i]:.1f} → {verdict}"),
        ]
        for col, (img, cmap, (lo, hi), title) in enumerate(panels):
            ax = fig.add_subplot(grid[row, col])
            ax.imshow(img, cmap=cmap, vmin=lo, vmax=hi)
            ax.set_title(title, fontsize=10, fontweight="bold" if col == 2 else None)
            ax.axis("off")

    ax_hist = fig.add_subplot(grid[4, :])
    bins = np.linspace(0, test_scores.max() * 1.05, 60)
    colours = {"healthy": "green", "chatter": "tab:orange", "scratch": "tab:blue", "pitting": "tab:purple"}
    for defect in CLASS_NAMES:
        sel = (test_df["defect_type"] == defect).to_numpy()
        ax_hist.hist(test_scores[sel], bins=bins, alpha=0.7, color=colours[defect], label=defect)
    ax_hist.axvline(limit, color="red", linestyle="--", linewidth=2, label=f"limit = {limit:.1f}")
    ax_hist.set_xlabel("Anomaly score (mean residual)")
    ax_hist.set_ylabel("Number of parts")
    ax_hist.set_title(f"Anomaly scores on {len(test_df)} unseen parts — ROC-AUC {auc:.3f}",
                      fontweight="bold")
    ax_hist.legend()

    fig.tight_layout()
    out = save_figure(fig, "06_golden_master_anomaly.png")
    print(f"[DONE] Saved: {out}")


if __name__ == "__main__":
    main()
