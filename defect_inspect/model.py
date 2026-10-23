"""Model construction, checkpoint save/load and single-image prediction."""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models

from .config import CLASSES, DEFAULT_WEIGHTS


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model(num_classes: int = len(CLASSES), pretrained: bool = True) -> nn.Module:
    """ResNet18 backbone (ImageNet weights) with a new classification head."""
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    m = models.resnet18(weights=weights)
    m.fc = nn.Sequential(nn.Dropout(0.2), nn.Linear(m.fc.in_features, num_classes))
    return m


def save_checkpoint(model: nn.Module, path: Path, meta: dict | None = None) -> None:
    """Save weights in fp16 to halve the file size (~22 MB); loaded back as fp32."""
    state = {k: (v.half() if v.is_floating_point() else v)
             for k, v in model.state_dict().items()}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": state, "classes": CLASSES, "arch": "resnet18",
                "meta": meta or {}}, path)


def load_model(path: Path = DEFAULT_WEIGHTS, device: torch.device | None = None) -> nn.Module:
    device = device or get_device()
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model = build_model(len(ckpt["classes"]), pretrained=False)
    model.load_state_dict({k: v.float() if v.is_floating_point() else v
                           for k, v in ckpt["state_dict"].items()})
    return model.to(device).eval()


@torch.no_grad()
def predict_proba(model: nn.Module, batch: torch.Tensor) -> torch.Tensor:
    device = next(model.parameters()).device
    return torch.softmax(model(batch.to(device)), dim=1).cpu()
