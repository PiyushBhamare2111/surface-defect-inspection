"""Evaluate the trained model on the held-out test split and write figures.

Usage:  python -m defect_inspect.evaluate [--weights models/resnet18_neu.pt]

Outputs
  reports/metrics.json                 accuracy, per-class P/R/F1, inspection stats
  docs/sample_grid.png                 2 examples per class
  docs/confusion_matrix.png
  docs/gradcam_examples.png
  docs/threshold_tradeoff.png          escape rate vs false-reject rate
  docs/robustness.png                  accuracy under simulated camera drift
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageFilter  # noqa: E402
from sklearn.metrics import (accuracy_score, classification_report,  # noqa: E402
                             confusion_matrix)
from torch.utils.data import DataLoader  # noqa: E402

from .config import (CLASSES, DEFAULT_CRITICAL, DEFAULT_WEIGHTS, DOCS_DIR,  # noqa: E402
                     PRETTY, REPORTS_DIR)
from .data import DefectDataset, discover, eval_transform, load_image, stratified_split  # noqa: E402
from .decision import band_stats, tradeoff_curve  # noqa: E402
from .gradcam import GradCAM, overlay  # noqa: E402
from .model import load_model  # noqa: E402

# Dark theme shared by every figure; accent #ff8a3d matches the README badges.
SURFACE, PANEL, INK, INK2, GRID = "#0b0c0f", "#15171c", "#f0efec", "#a9a8a0", "#2a2c33"
ACCENT, BLUE = "#ff8a3d", "#3987e5"
ORANGE = ACCENT
STRESS_BLUR = 3  # px Gaussian blur used for the "degraded camera" scenario
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
SEQ = LinearSegmentedColormap.from_list("accent", ["#15171c", "#5a3420", ACCENT, "#ffd2b3"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "font.family": "DejaVu Sans", "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
    "axes.titleweight": "bold", "axes.titlesize": 12,
})


def predict_all(model, samples, transform=None, batch_size=64) -> tuple[np.ndarray, np.ndarray]:
    dl = DataLoader(DefectDataset(samples, transform or eval_transform()), batch_size=batch_size)
    device = next(model.parameters()).device
    probs, labels = [], []
    with torch.no_grad():
        for x, y in dl:
            probs.append(torch.softmax(model(x.to(device)), 1).cpu().numpy())
            labels.append(y.numpy())
    return np.concatenate(probs), np.concatenate(labels)


def fig_sample_grid(samples, path: Path) -> None:
    fig, axes = plt.subplots(2, 6, figsize=(12, 4.6))
    for j, c in enumerate(CLASSES):
        picks = [s for s in samples if s.label == j][:2]
        for i, s in enumerate(picks):
            ax = axes[i, j]
            ax.imshow(load_image(s.path), cmap="gray", vmin=0, vmax=255)
            ax.set_xticks([]), ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if i == 0:
                ax.set_title(PRETTY[c], fontsize=11)
    fig.suptitle("NEU-CLS hot-rolled steel strip defects (200x200 grayscale)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_confusion(panels: list[tuple[str, np.ndarray]], path: Path) -> None:
    """panels: [(title, confusion matrix), ...]"""
    fig, axes = plt.subplots(1, len(panels), figsize=(6.2 * len(panels), 5.6), squeeze=False)
    names = [PRETTY[c] for c in CLASSES]
    for ax, (title, cm) in zip(axes[0], panels):
        ax.imshow(cm, cmap=SEQ, vmin=0, vmax=cm.sum(1).max())
        ax.set_xticks(range(len(names)), names, rotation=35, ha="right")
        ax.set_yticks(range(len(names)), names)
        thresh = cm.sum(1).max() / 2
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=11,
                        color=SURFACE if cm[i, j] > thresh else (INK if cm[i, j] else INK2))
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(title)
        for sp in ax.spines.values():
            sp.set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_gradcam(model, samples, path: Path) -> None:
    cam = GradCAM(model)
    tf = eval_transform()
    fig, axes = plt.subplots(2, 6, figsize=(12, 4.8))
    for j, c in enumerate(CLASSES):
        s = [s for s in samples if s.label == j][0]
        img = load_image(s.path)
        heat, idx, probs = cam(tf(img).unsqueeze(0))
        axes[0, j].imshow(img, cmap="gray")
        axes[0, j].set_title(PRETTY[c], fontsize=11)
        axes[1, j].imshow(overlay(img, heat))
        axes[1, j].set_title(f"pred: {PRETTY[CLASSES[idx]]}\n{probs[idx]:.0%}", fontsize=9, color=INK2)
        for ax in axes[:, j]:
            ax.set_xticks([]), ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
    cam.remove()
    fig.suptitle("Grad-CAM: where the model looks (top: input, bottom: heatmap overlay)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_tradeoff(panels: list[tuple[str, dict]], op_t: float, path: Path, critical) -> None:
    """panels: [(title, tradeoff_curve(...)), ...] drawn side by side on a shared y-scale."""
    fig, axes = plt.subplots(1, len(panels), figsize=(6.2 * len(panels), 4.6), sharey=True, squeeze=False)
    for ax, (title, curve) in zip(axes[0], panels):
        t, esc, frr = curve["threshold"], curve["escape_rate"] * 100, curve["false_reject_rate"] * 100
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.plot(t, esc, color=BLUE, lw=2, label="Escape rate (critical defect accepted)")
        ax.plot(t, frr, color=ACCENT, lw=2, label="False-reject rate (minor defect scrapped)")
        k = int(np.argmin(np.abs(t - op_t)))
        ax.axvline(op_t, color=INK2, lw=1, ls="--")
        ax.annotate(f"t = {op_t:.2f}\nescape {esc[k]:.1f}%\nfalse reject {frr[k]:.1f}%",
                    xy=(op_t, 50), xytext=(op_t + 0.04, 40), fontsize=9, color=INK)
        ax.set_xlabel("Reject threshold on P(critical defect)")
        ax.set_ylim(-2, 102)
        ax.set_xlim(0, 1)
        ax.set_title(title)
    axes[0, 0].set_ylabel("Rate on test set (%)")
    axes[0, 0].legend(loc="upper center", bbox_to_anchor=(0.5, 0.93), fontsize=9,
                      facecolor=SURFACE, edgecolor=GRID, framealpha=1)
    crit = ", ".join(PRETTY[c] for c in critical)
    fig.suptitle("Inspection trade-off: escapes vs false rejects as the reject threshold moves",
                 fontsize=12, fontweight="bold")
    fig.text(0.01, 0.01, f"Critical = {crit}. Other classes treated as acceptable (illustrative spec).",
             fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_training(history_path: Path, path: Path) -> None:
    if not history_path.exists():
        return
    h = json.loads(history_path.read_text())
    ep = [r["epoch"] for r in h["history"]]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, key, title in [(axes[0], "loss", "Loss (label-smoothed CE)"), (axes[1], "acc", "Accuracy")]:
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ys_tr = [r[f"train_{key}"] * (100 if key == "acc" else 1) for r in h["history"]]
        ys_va = [r[f"val_{key}"] * (100 if key == "acc" else 1) for r in h["history"]]
        ax.plot(ep, ys_tr, color=BLUE, lw=2, label="train")
        ax.plot(ep, ys_va, color=ACCENT, lw=2, label="validation")
        ax.axvline(h["best_epoch"], color=INK2, lw=1, ls="--")
        ax.set_xlabel("Epoch")
        ax.set_title(title)
        ax.legend(frameon=False)
    axes[0].annotate(f"best epoch {h['best_epoch']}\n(early stop)", (h["best_epoch"], axes[0].get_ylim()[1]),
                     xytext=(-8, -30), textcoords="offset points", ha="right", fontsize=8, color=INK2)
    axes[1].set_ylabel("%")
    fig.suptitle(f"Training curves - {h['device']}, {h['seconds']/60:.1f} min", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


class _Blur:
    def __init__(self, r): self.r = r
    def __call__(self, im): return im.filter(ImageFilter.GaussianBlur(self.r)) if self.r else im


class _Gain:
    def __init__(self, g): self.g = g
    def __call__(self, im): return Image.eval(im, lambda v: max(0, min(255, int(v * self.g))))


def robustness(model, test) -> dict:
    """Accuracy when the 'camera' drifts: defocus blur and lighting gain changes."""
    import torchvision.transforms as T
    base = eval_transform()
    out = {"blur": {}, "gain": {}}
    for r in [0, 1, 2, 3]:
        p, y = predict_all(model, test, T.Compose([T.Lambda(_Blur(r)), base]))
        out["blur"][r] = float((p.argmax(1) == y).mean())
    for g in [0.6, 0.8, 1.0, 1.2, 1.4]:
        p, y = predict_all(model, test, T.Compose([T.Lambda(_Gain(g)), base]))
        out["gain"][g] = float((p.argmax(1) == y).mean())
    return out


def fig_robustness(rob: dict, path: Path, baseline: dict | None = None) -> None:
    """rob / baseline: {"blur": {x: acc}, "gain": {x: acc}}; baseline = model trained without blur aug."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, key, xl, title in [(axes[0], "blur", "Gaussian blur radius (px)", "Defocus"),
                               (axes[1], "gain", "Brightness gain (x)", "Lighting drift")]:
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        series = [("with blur augmentation (shipped model)", rob, ACCENT)]
        if baseline:
            series.insert(0, ("without blur augmentation", baseline, BLUE))
        for label, r, color in series:
            xs = [str(x) for x in r[key].keys()]
            ys = [v * 100 for v in r[key].values()]
            ax.plot(xs, ys, color=color, lw=2, marker="o", ms=7, label=label)
            for x, y in zip(xs, ys):
                ax.annotate(f"{y:.0f}", (x, y), textcoords="offset points", xytext=(0, 8 if color == ACCENT else -15),
                            ha="center", fontsize=8, color=INK2)
        ax.set_xlabel(xl)
        ax.set_title(title)
    axes[0].set_ylabel("Test accuracy (%)")
    axes[0].set_ylim(0, 110)
    axes[0].legend(loc="lower left", fontsize=8, facecolor=SURFACE, edgecolor=GRID, framealpha=1)
    fig.suptitle("Robustness to simulated camera drift", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    ap.add_argument("--accept", type=float, default=0.20)
    ap.add_argument("--reject", type=float, default=0.80)
    ap.add_argument("--threshold", type=float, default=0.50, help="single pass/fail point to annotate")
    args = ap.parse_args(argv)

    DOCS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)
    samples = discover()
    train, val, test = stratified_split(samples)
    model = load_model(args.weights)
    probs, labels = predict_all(model, test)
    preds = probs.argmax(1)

    acc = accuracy_score(labels, preds)
    report = classification_report(labels, preds, target_names=CLASSES, output_dict=True, digits=4)
    cm = confusion_matrix(labels, preds, labels=range(len(CLASSES)))
    print(f"Test accuracy: {acc:.4f}  (n={len(labels)})")
    print(classification_report(labels, preds, target_names=CLASSES, digits=4))

    curve = tradeoff_curve(probs, labels, DEFAULT_CRITICAL)
    k = int(np.argmin(np.abs(curve["threshold"] - args.threshold)))
    band = band_stats(probs, labels, args.accept, args.reject, DEFAULT_CRITICAL)
    print(f"Single threshold {args.threshold:.2f}: escape {curve['escape_rate'][k]:.3%}, "
          f"false reject {curve['false_reject_rate'][k]:.3%}")
    print(f"Guard band [{args.accept}, {args.reject}]: {band}")

    rob = robustness(model, test)
    print("Robustness:", rob)

    # Same trade-off on deliberately degraded images, where the model does make mistakes.
    import torchvision.transforms as T
    probs_s, labels_s = predict_all(model, test, T.Compose([T.Lambda(_Blur(STRESS_BLUR)), eval_transform()]))
    curve_stress = tradeoff_curve(probs_s, labels_s, DEFAULT_CRITICAL)
    band_s = band_stats(probs_s, labels_s, args.accept, args.reject, DEFAULT_CRITICAL)
    print(f"Stressed (blur {STRESS_BLUR}): acc {(probs_s.argmax(1) == labels_s).mean():.4f}, "
          f"t={args.threshold}: escape {curve_stress['escape_rate'][k]:.3%}, "
          f"FRR {curve_stress['false_reject_rate'][k]:.3%}; band {band_s}")

    metrics = {
        "n_test": int(len(labels)), "split_sizes": [len(train), len(val), len(test)],
        "accuracy": acc, "macro_f1": report["macro avg"]["f1-score"],
        "per_class": {c: {k2: report[c][k2] for k2 in ("precision", "recall", "f1-score", "support")}
                      for c in CLASSES},
        "confusion_matrix": cm.tolist(),
        "critical_classes": DEFAULT_CRITICAL,
        "single_threshold": {"t": args.threshold, "escape_rate": float(curve["escape_rate"][k]),
                             "false_reject_rate": float(curve["false_reject_rate"][k])},
        "guard_band": {"accept": args.accept, "reject": args.reject, **band},
        "stressed": {"blur_radius": STRESS_BLUR,
                     "accuracy": float((probs_s.argmax(1) == labels_s).mean()),
                     "single_threshold": {"t": args.threshold,
                                          "escape_rate": float(curve_stress["escape_rate"][k]),
                                          "false_reject_rate": float(curve_stress["false_reject_rate"][k])},
                     "guard_band": {"accept": args.accept, "reject": args.reject, **band_s}},
        "stressed_confusion_matrix": confusion_matrix(labels_s, probs_s.argmax(1)).tolist(),
        "robustness": {"blur": {str(a): b for a, b in rob["blur"].items()},
                       "gain": {str(a): b for a, b in rob["gain"].items()}},
    }
    (REPORTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=1))

    fig_sample_grid(samples, DOCS_DIR / "sample_grid.png")
    cm_s = confusion_matrix(labels_s, probs_s.argmax(1), labels=range(len(CLASSES)))
    fig_confusion([(f"Clean test set (n={cm.sum()}) - {acc:.1%}", cm),
                   (f"Degraded camera, blur r={STRESS_BLUR}px - {(probs_s.argmax(1) == labels_s).mean():.1%}", cm_s)],
                  DOCS_DIR / "confusion_matrix.png")
    fig_gradcam(model, test, DOCS_DIR / "gradcam_examples.png")
    fig_tradeoff([("Clean images", curve), (f"Degraded camera (blur r={STRESS_BLUR}px)", curve_stress)],
                 args.threshold, DOCS_DIR / "threshold_tradeoff.png", DEFAULT_CRITICAL)
    base_path = REPORTS_DIR / "metrics_baseline_no_blur_aug.json"
    baseline = json.loads(base_path.read_text())["robustness"] if base_path.exists() else None
    fig_robustness({k2: {str(a): b for a, b in v.items()} for k2, v in rob.items()},
                   DOCS_DIR / "robustness.png", baseline)
    fig_training(args.weights.parent / "train_history.json", DOCS_DIR / "training_curves.png")
    print(f"Wrote {REPORTS_DIR / 'metrics.json'} and figures in {DOCS_DIR}")


if __name__ == "__main__":
    main()
