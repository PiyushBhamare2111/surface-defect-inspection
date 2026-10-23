"""Copy 2 held-out TEST images per class into samples/ for the demo app.

NEU-CLS is CC BY 4.0, so redistributing a few images with attribution is allowed.
Usage:  python scripts/make_samples.py
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from defect_inspect.config import CLASSES, SAMPLES_DIR  # noqa: E402
from defect_inspect.data import discover, stratified_split  # noqa: E402


def main() -> None:
    _, _, test = stratified_split(discover())
    SAMPLES_DIR.mkdir(exist_ok=True)
    for j, c in enumerate(CLASSES):
        for s in [s for s in test if s.label == j][:2]:
            shutil.copy2(s.path, SAMPLES_DIR / Path(s.path).name)
            print("copied", Path(s.path).name)


if __name__ == "__main__":
    main()
