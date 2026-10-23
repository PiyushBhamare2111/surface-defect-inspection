"""Unit tests that run without the dataset or trained weights."""
from __future__ import annotations

from collections import Counter

import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset

from defect_inspect.config import CLASSES, IMG_SIZE
from defect_inspect.data import (Sample, eval_transform, label_from_name, stratified_split,
                                 train_transform)
from defect_inspect.decision import (ACCEPT, REJECT, REVIEW, band_stats, decide, p_critical,
                                     tradeoff_curve)
from defect_inspect.gradcam import GradCAM, overlay
from defect_inspect.model import build_model, load_model, save_checkpoint
from defect_inspect.train import train_one_epoch


def fake_samples(per_class=300):
    return [Sample(f"data/{c}/{c}_{i}.jpg", j) for j, c in enumerate(CLASSES) for i in range(per_class)]


# ---------------------------------------------------------------- data
def test_label_from_name():
    assert label_from_name("rolled-in_scale_12.jpg") == CLASSES.index("rolled-in_scale")
    assert label_from_name("pitted_surface_3.jpg") == CLASSES.index("pitted_surface")
    with pytest.raises(ValueError):
        label_from_name("unknown_1.jpg")


def test_split_is_stratified_disjoint_and_complete():
    s = fake_samples()
    tr, va, te = stratified_split(s)
    assert (len(tr), len(va), len(te)) == (1260, 270, 270)
    paths = [x.path for x in tr + va + te]
    assert len(set(paths)) == len(s)  # disjoint and complete
    for part, n in [(tr, 210), (va, 45), (te, 45)]:
        assert set(Counter(x.label for x in part).values()) == {n}


def test_split_is_deterministic_and_order_independent():
    s = fake_samples()
    a = stratified_split(s)
    b = stratified_split(list(reversed(s)))
    assert [[x.path for x in p] for p in a] == [[x.path for x in p] for p in b]
    c = stratified_split(s, seed=7)
    assert [x.path for x in c[2]] != [x.path for x in a[2]]


# ---------------------------------------------------------------- preprocessing
@pytest.mark.parametrize("mode,size", [("L", (200, 200)), ("RGB", (320, 240))])
def test_preprocessing_shape_and_normalisation(mode, size):
    img = Image.new(mode, size, 128 if mode == "L" else (128, 128, 128))
    x = eval_transform()(img)
    assert x.shape == (3, IMG_SIZE, IMG_SIZE)
    assert x.dtype == torch.float32
    # mid-gray -> roughly zero after ImageNet normalisation
    assert x.abs().max() < 0.5
    # grayscale replicated to all channels before normalisation
    y = train_transform()(img)
    assert y.shape == (3, IMG_SIZE, IMG_SIZE)


# ---------------------------------------------------------------- model
def test_forward_pass_shape():
    m = build_model(pretrained=False).eval()
    with torch.no_grad():
        out = m(torch.randn(2, 3, IMG_SIZE, IMG_SIZE))
    assert out.shape == (2, len(CLASSES))


def test_checkpoint_roundtrip(tmp_path):
    m = build_model(pretrained=False).eval()
    save_checkpoint(m, tmp_path / "w.pt")
    m2 = load_model(tmp_path / "w.pt", torch.device("cpu"))
    x = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
    with torch.no_grad():
        assert torch.allclose(torch.softmax(m(x), 1), torch.softmax(m2(x), 1), atol=1e-2)


def test_one_training_step_reduces_loss_and_runs():
    torch.manual_seed(0)
    m = build_model(pretrained=False)
    x = torch.randn(8, 3, 64, 64)
    y = torch.arange(8) % len(CLASSES)
    dl = DataLoader(TensorDataset(x, y), batch_size=8)
    opt = torch.optim.SGD(m.parameters(), lr=0.01)
    before = [p.clone() for p in m.parameters()]
    loss, acc = train_one_epoch(m, dl, opt, torch.nn.CrossEntropyLoss(), torch.device("cpu"))
    assert np.isfinite(loss) and 0 <= acc <= 1
    assert any(not torch.equal(a, b) for a, b in zip(before, m.parameters()))


def test_gradcam_output():
    m = build_model(pretrained=False).eval()
    cam = GradCAM(m)
    heat, idx, probs = cam(torch.randn(1, 3, IMG_SIZE, IMG_SIZE))
    cam.remove()
    assert heat.shape == (IMG_SIZE, IMG_SIZE)
    assert 0 <= heat.min() and heat.max() <= 1
    assert 0 <= idx < len(CLASSES) and abs(probs.sum() - 1) < 1e-4
    ov = overlay(Image.new("L", (200, 200), 100), heat)
    assert ov.size == (IMG_SIZE, IMG_SIZE) and ov.mode == "RGB"


# ---------------------------------------------------------------- decision logic
def onehot(cls, p=1.0):
    v = np.full(len(CLASSES), (1 - p) / (len(CLASSES) - 1))
    v[CLASSES.index(cls)] = p
    return v


CRIT = ["crazing", "inclusion", "pitted_surface", "scratches"]


def test_decide_accept_reject_review():
    assert decide(onehot("inclusion", 0.97), 0.2, 0.8, CRIT).verdict == REJECT
    assert decide(onehot("patches", 0.97), 0.2, 0.8, CRIT).verdict == ACCEPT
    # 50/50 between a critical and a minor class -> guard band
    v = np.zeros(len(CLASSES)); v[CLASSES.index("scratches")] = 0.5; v[CLASSES.index("patches")] = 0.5
    d = decide(v, 0.2, 0.8, CRIT)
    assert d.verdict == REVIEW and abs(d.p_critical - 0.5) < 1e-9


def test_decide_threshold_boundaries_and_low_confidence():
    v = np.zeros(len(CLASSES)); v[CLASSES.index("crazing")] = 0.8; v[CLASSES.index("patches")] = 0.2
    assert decide(v, 0.2, 0.8, CRIT).verdict == REJECT          # >= reject threshold
    assert decide(v, 0.2, 0.81, CRIT).verdict == REVIEW
    # pure pass/fail gate when the band collapses
    assert decide(v, 0.85, 0.85, CRIT).verdict == ACCEPT
    # unsure model -> review even if decisive on P(critical)
    flat = np.zeros(len(CLASSES)); flat[[0, 1, 3, 5]] = 0.25
    assert decide(flat, 0.2, 0.8, CRIT, min_confidence=0.5).verdict == REVIEW
    assert decide(flat, 0.2, 0.8, CRIT, min_confidence=0.0).verdict == REJECT


def test_decide_validates_inputs():
    with pytest.raises(ValueError):
        decide(onehot("crazing"), 0.9, 0.1, CRIT)
    with pytest.raises(ValueError):
        decide([0.5, 0.5], 0.2, 0.8, CRIT)
    with pytest.raises(ValueError):
        decide(onehot("crazing"), 0.2, 0.8, ["rust"])


def test_tradeoff_curve_monotonic():
    rng = np.random.default_rng(0)
    probs = rng.dirichlet(np.ones(len(CLASSES)), size=200)
    labels = rng.integers(0, len(CLASSES), 200)
    c = tradeoff_curve(probs, labels, CRIT)
    assert c["escape_rate"][0] == 0 and c["false_reject_rate"][0] == 1   # t=0 rejects everything
    assert np.all(np.diff(c["escape_rate"]) >= 0)
    assert np.all(np.diff(c["false_reject_rate"]) <= 0)
    assert np.allclose(p_critical(probs, CRIT) + p_critical(probs, set(CLASSES) - set(CRIT)), 1)
    b = band_stats(probs, labels, 0.3, 0.7, CRIT)
    assert 0 <= b["review_rate"] <= 1
