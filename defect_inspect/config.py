"""Project-wide constants: class names, paths, image settings."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
IMAGE_DIR = DATA_DIR / "NEU-CLS"          # data/NEU-CLS/<class>/<file>.jpg
MODELS_DIR = ROOT / "models"
DEFAULT_WEIGHTS = MODELS_DIR / "resnet18_neu.pt"
DOCS_DIR = ROOT / "docs"
SAMPLES_DIR = ROOT / "samples"
REPORTS_DIR = ROOT / "reports"

# The six NEU-CLS defect classes (order = model output index).
CLASSES: list[str] = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]

PRETTY = {
    "crazing": "Crazing",
    "inclusion": "Inclusion",
    "patches": "Patches",
    "pitted_surface": "Pitted surface",
    "rolled-in_scale": "Rolled-in scale",
    "scratches": "Scratches",
}

# Illustrative disposition spec: which defect types are "critical" (part must be
# rejected) vs. "minor" (cosmetic, acceptable per this example customer spec).
# This is configurable in the app and CLI - real specs come from the customer / MRB.
DEFAULT_CRITICAL: list[str] = ["crazing", "inclusion", "pitted_surface", "scratches"]

IMG_SIZE = 224
# NEU images are grayscale; we replicate to 3 channels and use ImageNet stats so
# the pretrained backbone sees inputs in its familiar range.
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

SEED = 42
