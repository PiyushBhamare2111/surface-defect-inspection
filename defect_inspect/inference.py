"""High-level inspection API used by the Streamlit app and batch CLI."""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd
import torch
from PIL import Image

from .config import CLASSES, DEFAULT_CRITICAL, DEFAULT_WEIGHTS
from .data import eval_transform, load_image
from .decision import decide
from .gradcam import GradCAM, overlay
from .model import load_model


class Inspector:
    def __init__(self, weights: Path = DEFAULT_WEIGHTS, device: torch.device | None = None):
        self.model = load_model(weights, device)
        self.transform = eval_transform()

    def probs(self, img: Image.Image) -> np.ndarray:
        x = self.transform(img.convert("L")).unsqueeze(0)
        with torch.no_grad():
            dev = next(self.model.parameters()).device
            return torch.softmax(self.model(x.to(dev)), 1)[0].cpu().numpy()

    def explain(self, img: Image.Image, alpha: float = 0.45) -> tuple[Image.Image, np.ndarray]:
        """Return (Grad-CAM overlay image, class probabilities)."""
        img = img.convert("L")
        cam = GradCAM(self.model)
        try:
            heat, _, probs = cam(self.transform(img).unsqueeze(0))
        finally:
            cam.remove()
        return overlay(img, heat, alpha), probs

    def inspect_batch(
        self,
        items: Iterable[tuple[str, object]],
        accept_threshold: float = 0.20,
        reject_threshold: float = 0.80,
        critical: Iterable[str] = DEFAULT_CRITICAL,
        min_confidence: float = 0.0,
        preprocess: Callable[[Image.Image], Image.Image] | None = None,
    ) -> pd.DataFrame:
        """items: (name, path-or-file) pairs -> one inspection-log row per image."""
        critical = list(critical)
        rows = []
        for name, src in items:
            ts = datetime.now().isoformat(timespec="seconds")
            try:
                im = load_image(src)
                p = self.probs(preprocess(im) if preprocess else im)
            except Exception as e:  # unreadable image -> route to a human
                rows.append({"timestamp": ts, "image": name, "verdict": "MANUAL REVIEW",
                             "predicted_class": "", "confidence": np.nan, "p_critical": np.nan,
                             "reason": f"could not read image: {e}"})
                continue
            d = decide(p, accept_threshold, reject_threshold, critical, min_confidence)
            row = {"timestamp": ts, "image": name, **asdict(d)}
            row.update({f"p_{c}": float(v) for c, v in zip(CLASSES, p)})
            rows.append(row)
        df = pd.DataFrame(rows)
        lead = ["timestamp", "image", "verdict", "predicted_class", "confidence", "p_critical", "reason"]
        return df[lead + [c for c in df.columns if c not in lead]]
