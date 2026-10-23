"""Dataset discovery, stratified split, transforms and PyTorch Dataset."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import transforms as T

from .config import CLASSES, IMAGE_DIR, IMG_SIZE, MEAN, STD, SEED

IMG_EXTS = {".jpg", ".jpeg", ".bmp", ".png"}


@dataclass(frozen=True)
class Sample:
    path: str
    label: int


def label_from_name(name: str) -> int:
    """NEU file names look like 'pitted_surface_17.jpg' -> class index."""
    stem = Path(name).stem
    prefix = stem.rsplit("_", 1)[0]
    if prefix not in CLASSES:
        raise ValueError(f"Unrecognised class prefix in {name!r}")
    return CLASSES.index(prefix)


def discover(image_dir: Path = IMAGE_DIR) -> list[Sample]:
    """Return all labelled images under image_dir/<class>/*, sorted for determinism."""
    image_dir = Path(image_dir)
    if not image_dir.exists():
        raise FileNotFoundError(
            f"{image_dir} not found - run `python scripts/download_data.py` first."
        )
    samples = []
    for p in sorted(image_dir.rglob("*")):
        if p.suffix.lower() in IMG_EXTS and p.is_file():
            samples.append(Sample(str(p), label_from_name(p.name)))
    return samples


def stratified_split(
    samples: Sequence[Sample],
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = SEED,
) -> tuple[list[Sample], list[Sample], list[Sample]]:
    """Deterministic, class-stratified train/val/test split."""
    samples = sorted(samples, key=lambda s: s.path)
    labels = [s.label for s in samples]
    trainval, test = train_test_split(
        samples, test_size=test_frac, stratify=labels, random_state=seed
    )
    tv_labels = [s.label for s in trainval]
    rel_val = val_frac / (1.0 - test_frac)
    train, val = train_test_split(
        trainval, test_size=rel_val, stratify=tv_labels, random_state=seed
    )
    return list(train), list(val), list(test)


def save_split(split: dict[str, list[Sample]], path: Path, root: Path) -> None:
    out = {k: [str(Path(s.path).relative_to(root)) for s in v] for k, v in split.items()}
    Path(path).write_text(json.dumps(out, indent=1))


def eval_transform(img_size: int = IMG_SIZE) -> T.Compose:
    return T.Compose([
        T.Grayscale(num_output_channels=3),
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
    ])


def train_transform(img_size: int = IMG_SIZE) -> T.Compose:
    # Steel strip textures have no canonical orientation, so flips are safe.
    # Mild brightness/contrast jitter mimics lighting drift on a real line, and
    # random Gaussian blur mimics a slightly defocused lens / vibration. Without
    # the blur, a 1-px defocus dropped test accuracy from 100% to ~57%.
    return T.Compose([
        T.Grayscale(num_output_channels=3),
        T.RandomResizedCrop(img_size, scale=(0.75, 1.0), ratio=(0.9, 1.1)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.ColorJitter(brightness=0.25, contrast=0.25),
        T.RandomApply([T.GaussianBlur(kernel_size=9, sigma=(0.1, 3.0))], p=0.4),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
    ])


def load_image(path_or_file) -> Image.Image:
    """Open an image (path or file-like) and return a grayscale PIL image."""
    if hasattr(path_or_file, "seek"):  # re-readable uploads (Streamlit reruns)
        path_or_file.seek(0)
    img = Image.open(path_or_file)
    img.load()
    return img.convert("L")


class DefectDataset(Dataset):
    def __init__(self, samples: Sequence[Sample], transform=None):
        self.samples = list(samples)
        self.transform = transform or eval_transform()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        return self.transform(load_image(s.path)), s.label
