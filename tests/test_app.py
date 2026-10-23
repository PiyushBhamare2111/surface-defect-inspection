"""End-to-end checks that use the committed weights and sample images."""
from __future__ import annotations

from pathlib import Path

import pytest

from defect_inspect.config import CLASSES, DEFAULT_WEIGHTS, SAMPLES_DIR

ROOT = Path(__file__).resolve().parent.parent
needs_weights = pytest.mark.skipif(not DEFAULT_WEIGHTS.exists(), reason="trained weights not present")


@needs_weights
def test_trained_model_classifies_bundled_samples():
    from defect_inspect.inference import Inspector

    files = sorted(SAMPLES_DIR.glob("*.jpg"))
    assert len(files) == 12
    log = Inspector().inspect_batch((p.name, p) for p in files)
    truth = [p.name.rsplit("_", 1)[0] for p in files]
    assert (log["predicted_class"] == truth).all()
    assert set(log["verdict"]) <= {"ACCEPT", "REJECT", "MANUAL REVIEW"}
    assert all(f"p_{c}" in log.columns for c in CLASSES)


@needs_weights
def test_streamlit_app_runs_single_and_batch():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120).run()
    assert not at.exception, at.exception
    verdicts = [m.value for m in at.markdown if "ACCEPT" in m.value or "REJECT" in m.value
                or "MANUAL REVIEW" in m.value]
    assert verdicts, "no verdict rendered for the default sample"
    at.button[0].click().run()          # batch tab: run on bundled samples
    assert not at.exception, at.exception
    assert len(at.dataframe) == 1 and len(at.dataframe[0].value) == 12
