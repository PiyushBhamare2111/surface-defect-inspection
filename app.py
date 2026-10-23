"""Streamlit inspection station for steel surface defects.

Run:  streamlit run app.py   ->  http://localhost:8501
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image, ImageFilter

from defect_inspect.config import CLASSES, DEFAULT_CRITICAL, DEFAULT_WEIGHTS, PRETTY, SAMPLES_DIR
from defect_inspect.data import load_image
from defect_inspect.decision import ACCEPT, REJECT, decide

st.set_page_config(page_title="Surface Defect Inspection", page_icon="🔍", layout="wide")

VERDICT_COLOR = {ACCEPT: "#1baf7a", REJECT: "#e34948", "MANUAL REVIEW": "#eda100"}


@st.cache_resource(show_spinner="Loading model...")
def get_inspector():
    from defect_inspect.inference import Inspector
    return Inspector(DEFAULT_WEIGHTS)


def apply_blur(img: Image.Image, radius: float) -> Image.Image:
    return img.filter(ImageFilter.GaussianBlur(radius)) if radius else img


def sample_files() -> list[Path]:
    return sorted(p for p in SAMPLES_DIR.glob("*") if p.suffix.lower() in {".jpg", ".png", ".bmp"})


st.title("Surface Defect Inspection")
st.caption("ResNet18 transfer learning on the NEU-CLS hot-rolled steel strip dataset - "
           "defect class, Grad-CAM evidence and an accept / reject / manual-review disposition.")

if not DEFAULT_WEIGHTS.exists():
    st.error(f"Model weights not found at `{DEFAULT_WEIGHTS}`. Train with "
             "`python -m defect_inspect.train` (or re-clone: the weights are committed).")
    st.stop()

with st.sidebar:
    st.header("Disposition rules")
    critical = st.multiselect("Critical defect types (must reject)", CLASSES, default=DEFAULT_CRITICAL,
                              format_func=lambda c: PRETTY[c])
    accept_t, reject_t = st.slider(
        "Guard band on P(critical)", 0.0, 1.0, (0.20, 0.80), 0.01,
        help="Below the lower value -> ACCEPT. Above the upper value -> REJECT. "
             "In between -> MANUAL REVIEW. Drag both handles together for a pure pass/fail gate.")
    min_conf = st.slider("Min. top-class confidence", 0.0, 1.0, 0.50, 0.01,
                         help="Images the model is unsure about go to manual review.")
    alpha = st.slider("Heatmap opacity", 0.0, 1.0, 0.45, 0.05)
    st.header("Camera simulation")
    blur = st.slider("Defocus blur (px)", 0.0, 4.0, 0.0, 0.5,
                     help="Simulate an out-of-focus lens to see how confidence and the "
                          "disposition degrade - the reason vision stations need routine calibration.")
    st.markdown("---")
    st.caption("Raise the reject threshold -> fewer false rejects, more escapes. "
               "Widen the band -> fewer of both, more operator time. "
               "See `docs/threshold_tradeoff.png`.")

inspector = get_inspector()
tab_single, tab_batch = st.tabs(["Single part", "Batch inspection log"])

with tab_single:
    samples = sample_files()
    src = st.radio("Image source", ["Sample image", "Upload"], horizontal=True)
    img, name = None, None
    if src == "Sample image" and samples:
        choice = st.selectbox("Sample", samples, format_func=lambda p: p.name)
        img, name = load_image(choice), choice.name
    elif src == "Upload":
        up = st.file_uploader("Upload a grayscale or colour image", type=["jpg", "jpeg", "png", "bmp"])
        if up is not None:
            try:
                img, name = load_image(up), up.name
            except Exception as e:
                st.error(f"Could not read image: {e}")

    if img is not None:
        img = apply_blur(img, blur)
        cam_img, probs = inspector.explain(img, alpha)
        d = decide(probs, accept_t, reject_t, critical, min_conf)
        c1, c2, c3 = st.columns([1, 1, 1.3])
        c1.image(img.resize((400, 400)), caption=f"Input: {name}" + (f" (blur {blur}px)" if blur else ""),
                 width="stretch")
        c2.image(cam_img.resize((400, 400)), caption="Grad-CAM overlay", width="stretch")
        with c3:
            st.markdown(
                f"<div style='background:{VERDICT_COLOR[d.verdict]};color:white;padding:14px 18px;"
                f"border-radius:8px;font-size:1.6rem;font-weight:700;text-align:center'>{d.verdict}</div>",
                unsafe_allow_html=True)
            st.metric("Predicted defect", PRETTY[d.predicted_class])
            m1, m2 = st.columns(2)
            m1.metric("Confidence", f"{d.confidence:.1%}")
            m2.metric("P(critical)", f"{d.p_critical:.1%}")
            st.caption(d.reason)
            st.bar_chart(pd.DataFrame({"probability": probs}, index=[PRETTY[c] for c in CLASSES]),
                         horizontal=True, color="#ff8a3d", height=230)

with tab_batch:
    st.write("Inspect many images at once and download a CSV inspection log.")
    ups = st.file_uploader("Upload images", type=["jpg", "jpeg", "png", "bmp"],
                           accept_multiple_files=True, key="batch")
    use_samples = st.checkbox("Use the bundled sample images", value=not ups)
    if st.button("Run inspection", type="primary"):
        items = [(u.name, u) for u in ups or []]
        if use_samples:
            items += [(p.name, p) for p in sample_files()]
        if not items:
            st.warning("No images selected.")
        else:
            log = inspector.inspect_batch(items, accept_t, reject_t, critical, min_conf,
                                          preprocess=lambda im: apply_blur(im, blur))
            st.session_state["log"] = log
    log = st.session_state.get("log")
    if log is not None:
        counts = log["verdict"].value_counts()
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Inspected", len(log))
        k2.metric("Accepted", int(counts.get(ACCEPT, 0)))
        k3.metric("Rejected", int(counts.get(REJECT, 0)))
        k4.metric("Manual review", int(counts.get("MANUAL REVIEW", 0)))
        st.dataframe(log, width="stretch", hide_index=True)
        st.download_button("Download inspection log (CSV)", log.to_csv(index=False).encode(),
                           file_name=f"inspection_log_{datetime.now():%Y%m%d_%H%M%S}.csv",
                           mime="text/csv")
