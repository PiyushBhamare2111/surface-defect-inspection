"""Accept / reject / manual-review logic, written the way an inline vision
inspection station is configured.

The NEU dataset contains only defective strips, so "accept" here means "the
defect is a minor/cosmetic type allowed by the spec" and "reject" means "a
critical defect type is present". The score used for the decision is

    p_critical = sum of softmax probabilities over the critical classes

and two thresholds create a guard band:

    p_critical >= reject_threshold   -> REJECT
    p_critical <= accept_threshold   -> ACCEPT
    otherwise                        -> MANUAL REVIEW

A wide band sends more parts to a human (lower escapes and false rejects, more
labour); collapsing the band (accept == reject) gives a pure pass/fail gate.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from .config import CLASSES, DEFAULT_CRITICAL

ACCEPT, REJECT, REVIEW = "ACCEPT", "REJECT", "MANUAL REVIEW"


@dataclass
class Decision:
    verdict: str
    predicted_class: str
    confidence: float
    p_critical: float
    reason: str


def critical_mask(critical: Iterable[str], classes: Sequence[str] = CLASSES) -> np.ndarray:
    crit = set(critical)
    unknown = crit - set(classes)
    if unknown:
        raise ValueError(f"Unknown classes in critical list: {sorted(unknown)}")
    return np.array([c in crit for c in classes], dtype=bool)


def p_critical(probs: np.ndarray, critical: Iterable[str] = DEFAULT_CRITICAL) -> np.ndarray:
    """probs: (N, C) or (C,) softmax outputs -> probability that a critical defect is present."""
    probs = np.asarray(probs, dtype=float)
    return probs[..., critical_mask(critical)].sum(axis=-1)


def decide(
    probs: Sequence[float],
    accept_threshold: float = 0.20,
    reject_threshold: float = 0.80,
    critical: Iterable[str] = DEFAULT_CRITICAL,
    min_confidence: float = 0.0,
    classes: Sequence[str] = CLASSES,
) -> Decision:
    """Turn one image's class probabilities into an inspection verdict.

    min_confidence: if the top-class probability is below this, the part goes to
    manual review even when the critical score is decisive (an "unsure what this
    is" guard, useful for unseen defect types or bad images).
    """
    if not 0.0 <= accept_threshold <= reject_threshold <= 1.0:
        raise ValueError("Need 0 <= accept_threshold <= reject_threshold <= 1")
    probs = np.asarray(probs, dtype=float)
    if probs.shape != (len(classes),):
        raise ValueError(f"Expected {len(classes)} probabilities, got shape {probs.shape}")
    critical = list(critical)
    top = int(probs.argmax())
    conf = float(probs[top])
    pc = float(p_critical(probs, critical))
    if conf < min_confidence:
        return Decision(REVIEW, classes[top], conf, pc,
                        f"top-class confidence {conf:.2f} < {min_confidence:.2f}")
    if pc >= reject_threshold:
        return Decision(REJECT, classes[top], conf, pc,
                        f"P(critical)={pc:.2f} >= reject threshold {reject_threshold:.2f}")
    if pc <= accept_threshold:
        return Decision(ACCEPT, classes[top], conf, pc,
                        f"P(critical)={pc:.2f} <= accept threshold {accept_threshold:.2f}")
    return Decision(REVIEW, classes[top], conf, pc,
                    f"P(critical)={pc:.2f} inside guard band")


def tradeoff_curve(
    probs: np.ndarray,
    labels: np.ndarray,
    critical: Iterable[str] = DEFAULT_CRITICAL,
    thresholds: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Single pass/fail threshold sweep.

    escape_rate       = critical parts that were ACCEPTED / all critical parts
                        (defects shipped to the customer)
    false_reject_rate = minor parts that were REJECTED / all minor parts
                        (good-enough product scrapped - the "false reject" cost)
    """
    thresholds = np.linspace(0, 1, 201) if thresholds is None else thresholds
    mask = critical_mask(critical)
    is_crit = mask[np.asarray(labels)]
    pc = p_critical(probs, critical)
    esc, frr = [], []
    for t in thresholds:
        rej = pc >= t
        esc.append(((~rej) & is_crit).sum() / max(is_crit.sum(), 1))
        frr.append((rej & ~is_crit).sum() / max((~is_crit).sum(), 1))
    return {"threshold": thresholds, "escape_rate": np.array(esc),
            "false_reject_rate": np.array(frr)}


def band_stats(
    probs: np.ndarray,
    labels: np.ndarray,
    accept_threshold: float,
    reject_threshold: float,
    critical: Iterable[str] = DEFAULT_CRITICAL,
) -> dict[str, float]:
    """Escape / false-reject / manual-review rates for a two-threshold guard band."""
    mask = critical_mask(critical)
    is_crit = mask[np.asarray(labels)]
    pc = p_critical(probs, critical)
    rej = pc >= reject_threshold
    acc = pc <= accept_threshold
    rev = ~(rej | acc)
    n_crit, n_minor = max(is_crit.sum(), 1), max((~is_crit).sum(), 1)
    return {
        "escape_rate": float((acc & is_crit).sum() / n_crit),
        "false_reject_rate": float((rej & ~is_crit).sum() / n_minor),
        "review_rate": float(rev.mean()),
    }
