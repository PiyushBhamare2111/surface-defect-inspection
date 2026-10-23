"""Grad-CAM (Selvaraju et al., 2017): which pixels drove the prediction?

Hooks the last convolutional block, weights its feature maps by the average
gradient of the chosen class score, and upsamples the result to image size.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module | None = None):
        self.model = model
        self.target_layer = target_layer or model.layer4[-1]
        self._acts = None
        self._grads = None
        self._h1 = self.target_layer.register_forward_hook(self._save_acts)
        self._h2 = self.target_layer.register_full_backward_hook(self._save_grads)

    def _save_acts(self, _m, _inp, out):
        self._acts = out.detach()

    def _save_grads(self, _m, _gin, gout):
        self._grads = gout[0].detach()

    def remove(self) -> None:
        self._h1.remove()
        self._h2.remove()

    def __call__(self, x: torch.Tensor, class_idx: int | None = None) -> tuple[np.ndarray, int, np.ndarray]:
        """x: (1, 3, H, W). Returns (cam HxW in [0,1], class_idx, probs)."""
        self.model.eval()
        device = next(self.model.parameters()).device
        x = x.to(device).requires_grad_(True)
        with torch.enable_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, 1)[0].detach().cpu().numpy()
            idx = int(logits.argmax(1)) if class_idx is None else int(class_idx)
            self.model.zero_grad(set_to_none=True)
            logits[0, idx].backward()
        weights = self._grads.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self._acts).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam.cpu().numpy(), idx, probs


def overlay(img: Image.Image, cam: np.ndarray, alpha: float = 0.45) -> Image.Image:
    """Blend a jet-coloured heatmap over the (grayscale) inspection image."""
    import matplotlib.cm as cm

    base = img.convert("RGB").resize(cam.shape[::-1])
    heat = (cm.jet(cam)[..., :3] * 255).astype(np.uint8)
    out = (1 - alpha) * np.asarray(base, dtype=float) + alpha * heat
    return Image.fromarray(out.clip(0, 255).astype(np.uint8))
