"""Render 1920x1080 showcase slides into docs/media/ (for a project video / LinkedIn).

Needs the dataset (data/), trained weights and reports/metrics.json.
Usage:  python scripts/make_media.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from defect_inspect.config import CLASSES, PRETTY, REPORTS_DIR  # noqa: E402
from defect_inspect.data import discover, eval_transform, load_image, stratified_split  # noqa: E402
from defect_inspect.decision import tradeoff_curve  # noqa: E402
from defect_inspect.evaluate import (ACCENT, BLUE, GRID, INK, INK2, PANEL, STRESS_BLUR,  # noqa: E402
                                     SURFACE, _Blur, predict_all)
from defect_inspect.gradcam import GradCAM, overlay  # noqa: E402
from defect_inspect.model import load_model  # noqa: E402

OUT = ROOT / "docs" / "media"
W, H, DPI = 19.2, 10.8, 100


def canvas():
    fig = plt.figure(figsize=(W, H), dpi=DPI, facecolor=SURFACE)
    return fig


def header(fig, title, sub):
    fig.text(0.05, 0.92, title, fontsize=40, fontweight="bold", color=INK, va="center")
    fig.text(0.05, 0.86, sub, fontsize=19, color=INK2, va="center")
    fig.add_artist(plt.Line2D([0.05, 0.13], [0.82, 0.82], color=ACCENT, lw=5))


def footer(fig):
    fig.text(0.05, 0.035, "Piyush Bhamare  |  github.com/PiyushBhamare2111/surface-defect-inspection",
             fontsize=14, color=INK2)
    fig.text(0.95, 0.035, "NEU-CLS steel surface defects  |  PyTorch  |  Streamlit", fontsize=14,
             color=INK2, ha="right")


def tile(ax, img, title=None, color=INK, cmap="gray"):
    ax.imshow(img, cmap=cmap)
    ax.set_xticks([]), ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor(GRID)
    if title:
        ax.set_title(title, fontsize=17, color=color, pad=8)


def slide_title(samples):
    fig = canvas()
    header(fig, "Surface Defect Inspection with Deep Learning",
           "Classifying hot-rolled steel strip defects - and deciding accept / reject / manual review")
    for j, c in enumerate(CLASSES):
        picks = [s for s in samples if s.label == j][:2]
        for i, s in enumerate(picks):
            ax = fig.add_axes([0.055 + j * 0.152, 0.42 - i * 0.29, 0.13, 0.25])
            tile(ax, load_image(s.path), PRETTY[c] if i == 0 else None)
    footer(fig)
    fig.savefig(OUT / "01_title_defect_classes.png", dpi=DPI, facecolor=SURFACE)
    plt.close(fig)


def slide_gradcam(model, test):
    fig = canvas()
    header(fig, "Grad-CAM: the model shows its evidence",
           "Heatmaps highlight the pixels that drove each prediction - like an inspector pointing at the flaw")
    cam = GradCAM(model)
    tf = eval_transform()
    for j, c in enumerate(CLASSES):
        s = [s for s in test if s.label == j][1]
        img = load_image(s.path)
        heat, idx, probs = cam(tf(img).unsqueeze(0))
        x = 0.05 + j * 0.15
        tile(fig.add_axes([x, 0.45, 0.135, 0.24]), img, PRETTY[c])
        tile(fig.add_axes([x, 0.15, 0.135, 0.24]), overlay(img, heat))
        fig.text(x + 0.0675, 0.115, f"{PRETTY[CLASSES[idx]]}  {probs[idx]:.0%}", ha="center",
                 fontsize=15, color=ACCENT)
    cam.remove()
    footer(fig)
    fig.savefig(OUT / "02_gradcam_evidence.png", dpi=DPI, facecolor=SURFACE)
    plt.close(fig)


def slide_tradeoff(model, test, m):
    import torchvision.transforms as T
    fig = canvas()
    header(fig, "Escapes vs false rejects - the quality engineer's trade-off",
           f"Same model, camera defocused by {STRESS_BLUR}px: moving the reject threshold trades shipped defects "
           "for scrapped product")
    p, y = predict_all(model, test, T.Compose([T.Lambda(_Blur(STRESS_BLUR)), eval_transform()]))
    c = tradeoff_curve(p, y)
    ax = fig.add_axes([0.07, 0.15, 0.55, 0.6], facecolor=SURFACE)
    ax.grid(axis="y", color=GRID)
    ax.plot(c["threshold"], c["escape_rate"] * 100, color=BLUE, lw=4, label="Escape rate (critical defect shipped)")
    ax.plot(c["threshold"], c["false_reject_rate"] * 100, color=ACCENT, lw=4,
            label="False-reject rate (acceptable part scrapped)")
    ax.set_xlim(0, 1), ax.set_ylim(-2, 102)
    ax.set_xlabel("Reject threshold on P(critical defect)", fontsize=17, color=INK2)
    ax.set_ylabel("Rate (%)", fontsize=17, color=INK2)
    ax.tick_params(labelsize=14, colors=INK2)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    for sp in ["left", "bottom"]:
        ax.spines[sp].set_color(GRID)
    ax.legend(fontsize=15, facecolor=SURFACE, edgecolor=GRID, labelcolor=INK, loc="upper center")
    st = m["stressed"]
    cards = [
        ("Single gate  t = 0.50", f"{st['single_threshold']['escape_rate']:.1%}", "escape rate", BLUE),
        ("Guard band  0.20 - 0.80", f"{st['guard_band']['escape_rate']:.1%}", "escape rate", ACCENT),
        ("Guard band cost", f"{st['guard_band']['review_rate']:.1%}", "to manual review", INK2),
    ]
    for i, (t, v, lab, col) in enumerate(cards):
        y0 = 0.6 - i * 0.2
        fig.patches.append(plt.Rectangle((0.67, y0), 0.28, 0.16, transform=fig.transFigure,
                                         facecolor=PANEL, edgecolor=GRID, lw=1.5))
        fig.text(0.69, y0 + 0.125, t, fontsize=16, color=INK2)
        fig.text(0.69, y0 + 0.045, v, fontsize=42, color=col, fontweight="bold")
        fig.text(0.82, y0 + 0.06, lab, fontsize=16, color=INK2)
    footer(fig)
    fig.savefig(OUT / "03_escape_vs_false_reject.png", dpi=DPI, facecolor=SURFACE)
    plt.close(fig)


def slide_results(m):
    base = json.loads((REPORTS_DIR / "metrics_baseline_no_blur_aug.json").read_text())
    fig = canvas()
    header(fig, "Results", "ResNet18 fine-tuned in ~1.5 min on an RTX 4060 Laptop GPU  |  held-out test set of 270 images")
    stats = [
        (f"{m['accuracy']:.1%}", "test accuracy\n(6 defect classes)"),
        (f"{m['macro_f1']:.3f}", "macro F1"),
        (f"{base['robustness']['blur']['1'] * 100:.0f}% -> {m['robustness']['blur']['1'] * 100:.0f}%",
         "accuracy under 1px defocus\nbefore -> after blur augmentation"),
        (f"{m['stressed']['single_threshold']['escape_rate']:.0%} -> {m['stressed']['guard_band']['escape_rate']:.1%}",
         f"escapes on a {STRESS_BLUR}px-defocused camera:\npass/fail gate -> guard band"),
    ]
    for i, (v, lab) in enumerate(stats):
        x = 0.05 + i * 0.228
        fig.patches.append(plt.Rectangle((x, 0.38), 0.21, 0.34, transform=fig.transFigure,
                                         facecolor=PANEL, edgecolor=GRID, lw=1.5))
        fig.text(x + 0.105, 0.6, v, fontsize=44 if len(v) < 9 else 34, color=ACCENT, ha="center",
                 fontweight="bold", va="center")
        fig.text(x + 0.105, 0.46, lab, fontsize=14, color=INK2, ha="center", va="center", linespacing=1.4)
    lines = ["Transfer learning (ImageNet ResNet18) + flips, crops, lighting jitter, defocus augmentation",
             "Grad-CAM heatmaps for every decision  |  configurable critical-defect spec",
             "Streamlit station: single-part view + batch CSV inspection log"]
    for i, t in enumerate(lines):
        fig.text(0.05, 0.27 - i * 0.055, "-  " + t, fontsize=18, color=INK)
    footer(fig)
    fig.savefig(OUT / "04_results.png", dpi=DPI, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    m = json.loads((REPORTS_DIR / "metrics.json").read_text())
    samples = discover()
    _, _, test = stratified_split(samples)
    model = load_model()
    slide_title(samples)
    slide_gradcam(model, test)
    slide_tradeoff(model, test, m)
    slide_results(m)
    print("Wrote slides to", OUT)


if __name__ == "__main__":
    main()
