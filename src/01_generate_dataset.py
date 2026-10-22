"""
Chapter 1 — Synthetic Factory Dataset & CSV Ground Truth
========================================================
Simulates an optical inspection camera on a machining line:

* All 300 part images go into ONE folder (data/images/) with serial-number
  filenames (part_001.png ... part_300.png) — the filename gives nothing away.
* The true inspection result is recorded in a separate ledger (data/labels.csv).

Defect classes (128 x 128 grayscale, steel-grey surface with machining grain):
    healthy  150  PASS  – normal milling grain + sensor noise
    chatter   50  FAIL  – high-frequency vibration bands (spindle / loose clamp)
    scratch   50  FAIL  – thin diagonal gouge with shadow + burr edge
    pitting   50  FAIL  – 2–4 dark circular cavities (casting porosity)

Run:  python src/01_generate_dataset.py
"""

import numpy as np
import pandas as pd
from PIL import Image

from common import IMAGES_DIR, LABELS_CSV, RANDOM_STATE

SIZE = 128


def healthy_surface(rng: np.random.Generator) -> np.ndarray:
    base = np.full((SIZE, SIZE), 128.0, dtype=np.float32)
    grain = (
        np.sin(np.linspace(0, rng.uniform(12, 18) * np.pi, SIZE))[:, None]
        * rng.uniform(8, 13)
    )
    noise = rng.normal(0, rng.uniform(3, 5), (SIZE, SIZE))
    return base + grain + noise


def chatter_surface(rng: np.random.Generator) -> np.ndarray:
    img = healthy_surface(rng)
    freq = rng.uniform(50, 70)
    bands = np.sin(np.linspace(0, freq * np.pi, SIZE))[None, :] * rng.uniform(25, 40)
    return img + bands


def scratch_surface(rng: np.random.Generator) -> np.ndarray:
    img = healthy_surface(rng)
    start_y, start_x = rng.integers(15, 45, size=2)
    length = rng.integers(40, 75)
    slope = rng.uniform(0.8, 1.3)
    for step in range(length):
        y = min(SIZE - 1, start_y + step)
        x = min(SIZE - 1, start_x + int(step * slope))
        img[y, max(0, x - 1): min(SIZE, x + 2)] = 25   # gouge shadow
        if x + 2 < SIZE:
            img[y, x + 2] = 235                         # raised burr
    return img


def pitting_surface(rng: np.random.Generator) -> np.ndarray:
    img = healthy_surface(rng)
    yy, xx = np.ogrid[:SIZE, :SIZE]
    for _ in range(rng.integers(2, 5)):
        cy, cx = rng.integers(25, 100, size=2)
        r = rng.integers(4, 8)
        img[(xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2] = 20
    return img


GENERATORS = {
    # defect_type: (qc_status, generator, count)
    "healthy": ("PASS", healthy_surface, 150),
    "chatter": ("FAIL", chatter_surface, 50),
    "scratch": ("FAIL", scratch_surface, 50),
    "pitting": ("FAIL", pitting_surface, 50),
}


def main() -> None:
    rng = np.random.default_rng(RANDOM_STATE)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    # Build the production run, then shuffle so serial numbers reveal nothing
    batch = [
        (defect, status, gen)
        for defect, (status, gen, count) in GENERATORS.items()
        for _ in range(count)
    ]
    order = rng.permutation(len(batch))

    print(f"Generating {len(batch)} parts into: {IMAGES_DIR}")
    records = []
    for part_id, idx in enumerate(order, start=1):
        defect, status, gen = batch[idx]
        filename = f"part_{part_id:03d}.png"
        pixels = np.clip(gen(rng), 0, 255).astype(np.uint8)
        Image.fromarray(pixels).save(IMAGES_DIR / filename)
        records.append(
            {
                "part_id": part_id,
                "filename": filename,
                "qc_status": status,
                "defect_type": defect,
                "is_defective": int(status == "FAIL"),
            }
        )

    df = pd.DataFrame(records)
    df.to_csv(LABELS_CSV, index=False)

    print(f"[DONE] Labels saved to: {LABELS_CSV}\n")
    print("Class balance:")
    print(df["defect_type"].value_counts().to_string())
    print("\nFirst 5 rows of labels.csv:")
    print(df.head().to_string(index=False))


if __name__ == "__main__":
    main()
