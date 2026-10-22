"""
Chapter 4 — Automated PASS / FAIL Quality-Control Classifier
============================================================
1. Measure two physical features for all 300 parts:
       Roughness (std-dev)  and  Max Edge (Sobel peak)
2. Split 80 / 20 into 240 training parts and 60 UNSEEN test parts (stratified).
3. Train a Logistic Regression (features standardised first, so both
   measurements count equally despite very different units/scales).
4. Evaluate on the unseen batch. On a production line the costliest error is
   a FALSE NEGATIVE — a defective part shipped to the customer — so recall on
   the FAIL class is reported explicitly.

Run:  python src/04_binary_qc_classifier.py
Output: assets/04_binary_qc_classifier.png
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import FEATURE_NAMES, RANDOM_STATE, build_feature_table, load_labels, save_figure


def main() -> None:
    df = load_labels()
    print(f"Extracting features from {len(df)} parts...")
    X = build_feature_table(df)[:, :2]  # roughness, max edge
    y = df["is_defective"].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    print(f"Train: {len(X_train)} parts | Unseen test: {len(X_test)} parts")

    model = make_pipeline(StandardScaler(), LogisticRegression(random_state=RANDOM_STATE))
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    cm = confusion_matrix(y_test, pred)
    tn, fp, fn, tp = cm.ravel()
    print("\n" + classification_report(y_test, pred, target_names=["PASS", "FAIL"], digits=3))
    print(f"False negatives (defects shipped): {fn}   False alarms (good parts scrapped): {fp}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Decision regions drawn by asking the model to classify a fine grid of points
    pad = 0.05 * (X.max(axis=0) - X.min(axis=0))
    xx, yy = np.meshgrid(
        np.linspace(X[:, 0].min() - pad[0], X[:, 0].max() + pad[0], 300),
        np.linspace(X[:, 1].min() - pad[1], X[:, 1].max() + pad[1], 300),
    )
    zz = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    axes[0].contourf(xx, yy, zz, levels=[-0.5, 0.5, 1.5], colors=["#d9f2d9", "#f9d6d5"], alpha=0.8)
    axes[0].contour(xx, yy, zz, levels=[0.5], colors="k", linestyles="--", linewidths=1.5)
    axes[0].scatter(*X_test[y_test == 0].T, c="green", edgecolors="k", s=50, label="PASS (healthy)")
    axes[0].scatter(*X_test[y_test == 1].T, c="red", marker="x", s=60, label="FAIL (defective)")
    axes[0].set_xlabel(FEATURE_NAMES[0])
    axes[0].set_ylabel(FEATURE_NAMES[1])
    axes[0].set_title("Decision boundary on 60 unseen test parts")
    axes[0].legend(loc="upper left")

    ConfusionMatrixDisplay(cm, display_labels=["PASS", "FAIL"]).plot(
        ax=axes[1], cmap="Blues", colorbar=False
    )
    acc = (tp + tn) / cm.sum()
    axes[1].set_title(f"Test confusion matrix — accuracy {acc:.1%}, FN = {fn}")

    out = save_figure(fig, "04_binary_qc_classifier.png")
    print(f"[DONE] Saved: {out}")


if __name__ == "__main__":
    main()
