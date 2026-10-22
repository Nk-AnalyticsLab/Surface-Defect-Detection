"""
Chapter 5 — Multi-Class Defect Triage (root-cause classification)
=================================================================
Knowing a part FAILED is not enough — maintenance needs to know WHY:

    healthy  -> ship it
    chatter  -> check spindle bearings / clamping
    scratch  -> inspect tool edge, handling and conveyors
    pitting  -> raise a quality issue with the casting supplier

A Decision Tree learns a readable troubleshooting flowchart from three
measurements (roughness, max edge, cavity area). Trained on 240 parts and
evaluated on 60 unseen parts.

Run:  python src/05_multiclass_defect_triage.py
Output: assets/05_multiclass_defect_triage.png
"""

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

from common import (
    CLASS_NAMES,
    FEATURE_NAMES,
    RANDOM_STATE,
    build_feature_table,
    load_labels,
    load_part,
    save_figure,
)


def main() -> None:
    df = load_labels()
    print(f"Extracting features from {len(df)} parts...")
    X = build_feature_table(df)
    y = df["defect_type"].map({name: i for i, name in enumerate(CLASS_NAMES)}).to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    print(f"Train: {len(X_train)} parts | Unseen test: {len(X_test)} parts")

    tree = DecisionTreeClassifier(max_depth=3, random_state=RANDOM_STATE)
    tree.fit(X_train, y_train)
    pred = tree.predict(X_test)

    labels = [c.capitalize() for c in CLASS_NAMES]
    print("\nLearned troubleshooting flowchart:")
    print(export_text(tree, feature_names=FEATURE_NAMES, class_names=labels, decimals=1))
    print(classification_report(y_test, pred, target_names=labels, digits=3))

    fig = plt.figure(figsize=(16, 9))

    for i, defect in enumerate(CLASS_NAMES):
        fname = df.loc[df["defect_type"] == defect, "filename"].iloc[0]
        ax = plt.subplot2grid((2, 4), (0, i))
        ax.imshow(load_part(fname), cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"Class {i}: {defect.capitalize()}\n({fname})", fontsize=10, fontweight="bold")
        ax.axis("off")

    ax_tree = plt.subplot2grid((2, 4), (1, 0), colspan=3)
    plot_tree(tree, feature_names=FEATURE_NAMES, class_names=labels,
              filled=True, rounded=True, impurity=False, ax=ax_tree, fontsize=8)
    ax_tree.set_title("Learned troubleshooting flowchart (240 training parts)", fontweight="bold")

    ax_cm = plt.subplot2grid((2, 4), (1, 3))
    cm = confusion_matrix(y_test, pred)
    ConfusionMatrixDisplay(cm, display_labels=labels).plot(
        ax=ax_cm, cmap="Blues", colorbar=False, xticks_rotation=45
    )
    acc = cm.trace() / cm.sum()
    ax_cm.set_title(f"Unseen test set (N = {len(y_test)}) — {acc:.1%}", fontweight="bold")

    out = save_figure(fig, "05_multiclass_defect_triage.png")
    print(f"[DONE] Saved: {out}")


if __name__ == "__main__":
    main()
