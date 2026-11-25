"""
Construction Safety Detection - Streamlit front-end
Run:  streamlit run app.py
"""

import io
import os
import shutil
import subprocess
import tempfile
import time
from collections import Counter

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO

# ----------------------------------------------------------------------------
# CONFIG - edit these to match your project
# ----------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH = os.path.join(BASE_DIR, "best.pt")

# Classes that mean "unsafe". Any class whose name starts with "NO-" or "no_"
# is also treated as a violation automatically (see is_violation()).
DEFAULT_VIOLATION_CLASSES = [
    "no_helmet",
    "no_safebelt",
    "smoking",
    "NO-Hardhat",
    "NO-Safety Vest",
    "NO-Mask",
]

PAGE_TITLE = "Site Safety Check"
CREATOR_NAME = "DARBAR NAGURU"

st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon="🦺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------
# STYLING
# ----------------------------------------------------------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@600;700&family=Barlow:wght@400;500;600&display=swap');

:root {
    --concrete: #EEF0F2;
    --steel: #1E2A35;
    --steel-soft: #5B6875;
    --hivis: #F2B705;
    --ok: #1F8A5B;
    --alert: #D9381E;
    --card: #FFFFFF;
    --line: #D8DDE2;
}

html, body, [class*="css"], .stApp {
    font-family: 'Barlow', sans-serif;
    color: var(--steel);
}
.stApp { background: var(--concrete); }

h1, h2, h3, .display {
    font-family: 'Barlow Condensed', sans-serif !important;
    letter-spacing: 0.01em;
    color: var(--steel);
}

/* Header band with hazard stripe */
.hero {
    background: var(--steel);
    border-radius: 6px;
    padding: 1.6rem 2rem 1.4rem 2rem;
    margin-bottom: 1.2rem;
    border-bottom: 10px solid transparent;
    border-image: repeating-linear-gradient(
        -45deg, var(--hivis) 0 14px, var(--steel) 14px 28px) 10;
}
.hero h1 {
    color: #fff !important;
    font-size: 2.6rem;
    margin: 0;
    line-height: 1.05;
}
.hero p {
    color: #B9C3CC;
    margin: .5rem 0 0 0;
    font-size: 1.02rem;
    max-width: 62ch;
}

/* Verdict banner */
.verdict {
    border-radius: 6px;
    padding: 1rem 1.3rem;
    margin: .2rem 0 1rem 0;
    display: flex; align-items: center; gap: .9rem;
    border: 1px solid var(--line);
    background: var(--card);
}
.verdict .dot { width: 14px; height: 14px; border-radius: 50%; flex: none; }
.verdict .title {
    font-family: 'Barlow Condensed', sans-serif;
    font-weight: 700; font-size: 1.5rem; line-height: 1.1;
}
.verdict .sub { color: var(--steel-soft); font-size: .95rem; }
.verdict.safe  { border-left: 8px solid var(--ok); }
.verdict.safe .dot { background: var(--ok); }
.verdict.unsafe { border-left: 8px solid var(--alert); }
.verdict.unsafe .dot { background: var(--alert); }
.verdict.idle { border-left: 8px solid var(--line); }
.verdict.idle .dot { background: var(--line); }

/* Metrics */
[data-testid="stMetric"] {
    background: var(--card);
    border: 1px solid var(--line);
    border-radius: 6px;
    padding: .8rem 1rem;
}
[data-testid="stMetricLabel"] { color: var(--steel-soft); }
[data-testid="stMetricValue"] {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 2.2rem;
}

/* Sidebar */
[data-testid="stSidebar"] { background: var(--card); border-right: 1px solid var(--line); }

/* Tabs */
.stTabs [data-baseweb="tab-list"] { gap: 6px; }
.stTabs [data-baseweb="tab"] {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 1.15rem; font-weight: 600;
    padding: .5rem 1.1rem;
}
.stTabs [aria-selected="true"] { border-bottom: 3px solid var(--hivis) !important; }

/* Buttons */
.stButton > button, .stDownloadButton > button {
    background: var(--hivis); color: var(--steel);
    border: 0; border-radius: 4px; font-weight: 600;
    padding: .55rem 1.2rem;
}
.stButton > button:hover, .stDownloadButton > button:hover {
    background: #FFC81F; color: var(--steel);
}
.stButton > button:focus-visible, .stDownloadButton > button:focus-visible {
    outline: 3px solid var(--steel); outline-offset: 2px;
}

/* Creator footer label */
.creator-label {
    margin-top: 2.5rem;
    padding: .85rem 1.1rem;
    border-top: 1px solid var(--line);
    color: var(--steel-soft);
    font-size: .92rem;
    text-align: center;
}
.creator-label strong {
    color: var(--steel);
    letter-spacing: 0.03em;
}

footer, #MainMenu { visibility: hidden; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# HELPERS
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model…")
def load_model(path: str) -> YOLO:
    return YOLO(path)


def is_violation(name: str, violation_set: set) -> bool:
    n = name.lower()
    return name in violation_set or n.startswith("no-") or n.startswith("no_")


def prepare_rgb_image(pil_img: Image.Image) -> np.ndarray:
    """Ensure a valid 3-channel uint8 RGB NumPy array (H, W, 3)."""
    rgb = pil_img.convert("RGB")
    if rgb.width <= 0 or rgb.height <= 0:
        raise ValueError("Uploaded image has invalid dimensions.")
    arr = np.asarray(rgb, dtype=np.uint8)
    if arr.ndim != 3 or arr.shape[2] != 3:
        arr = np.stack([arr[:, :, 0]] * 3, axis=-1).astype(np.uint8)
    return arr


def run_detection_rgb(model: YOLO, rgb_arr: np.ndarray, conf: float, iou: float, imgsz: int = 640):
    """Run YOLO prediction on an (H, W, 3) uint8 RGB image and return (annotated_rgb, detections)."""
    results = model.predict(
        source=rgb_arr,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        verbose=False,
    )
    result = results[0]
    annotated_bgr = result.plot()  # BGR uint8
    annotated_rgb = cv2