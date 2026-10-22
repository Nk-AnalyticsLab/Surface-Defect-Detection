"""
Shared helpers used by every chapter script.

Keeping paths, image loading and feature extraction in one place means each
chapter script stays short and every script measures parts the same way —
the software equivalent of every inspector on the shop floor using the same
calibrated gauge.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # render plots to file; no display window needed

import numpy as np
import pandas as pd
from PIL import Image
from scipy.signal import convolve2d

# --------------------------------------------------------------------------
# Project paths (resolved relative to this file, so scripts run from anywhere)
# --------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
IMAGES_DIR = DATA_DIR / "images"
LABELS_CSV = DATA_DIR / "labels.csv"
ASSETS_DIR = ROOT_DIR / "assets"

RANDOM_STATE = 42
CLASS_NAMES = ["healthy", "chatter", "scratch", "pitting"]

# 3x3 Sobel "gauges" — measure how steeply brightness changes left-right / up-down
SOBEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
SOBEL_Y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)

# A pixel darker than this is treated as a deep cavity (pit or gouge shadow)
CAVITY_THRESHOLD = 40


def load_labels() -> pd.DataFrame:
    """Read the ground-truth inspection ledger, with a helpful error if missing."""
    if not LABELS_CSV.exists():
        raise FileNotFoundError(
            f"Dataset not found at {LABELS_CSV}.\n"
            "Generate it first with:  python src/01_generate_dataset.py"
        )
    return pd.read_csv(LABELS_CSV)


def load_part(filename: str) -> np.ndarray:
    """Load one part image as a 2-D float32 grayscale matrix (0–255)."""
    return np.array(Image.open(IMAGES_DIR / filename).convert("L"), dtype=np.float32)


def edge_response(image: np.ndarray) -> np.ndarray:
    """Slide the Sobel gauges over the image and return gradient magnitude."""
    gx = convolve2d(image, SOBEL_X, mode="same", boundary="symm")
    gy = convolve2d(image, SOBEL_Y, mode="same", boundary="symm")
    return np.hypot(gx, gy)


def extract_features(image: np.ndarray) -> list[float]:
    """
    Three physical measurements per part:
      1. roughness   – std-dev of gray levels (like Ra surface roughness)
      2. max_edge    – strongest Sobel response (sharpest scratch/edge)
      3. cavity_area – number of very dark pixels (pits / deep gouges)
    """
    roughness = float(np.std(image))
    max_edge = float(np.max(edge_response(image)))
    cavity_area = float(np.sum(image < CAVITY_THRESHOLD))
    return [roughness, max_edge, cavity_area]


FEATURE_NAMES = ["Roughness (std)", "Max Edge (Sobel)", "Cavity Area (px)"]


def build_feature_table(df: pd.DataFrame) -> np.ndarray:
    """Extract features for every row in the labels table."""
    return np.array([extract_features(load_part(f)) for f in df["filename"]])


def save_figure(fig, name: str) -> Path:
    """Save a matplotlib figure into assets/ and return its path."""
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = ASSETS_DIR / name
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    return out_path
