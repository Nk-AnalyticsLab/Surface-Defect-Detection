"""
Run the whole pipeline in order: generate the dataset, then chapters 2–6.

Usage:  python run_all.py
"""

import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent / "src"
SCRIPTS = sorted(p for p in SRC.glob("[0-9][0-9]_*.py"))

for script in SCRIPTS:
    print(f"\n{'=' * 70}\n>>> {script.name}\n{'=' * 70}")
    result = subprocess.run([sys.executable, str(script)], cwd=SRC)
    if result.returncode != 0:
        sys.exit(f"Stopped: {script.name} failed.")

print("\nAll chapters finished. Figures are in the assets/ folder.")
