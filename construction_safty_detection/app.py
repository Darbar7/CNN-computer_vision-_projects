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
DEFAULT_MODEL_PATH = "best.pt"  # path to your trained YOLO weights

# Classes that mean "unsafe". Any class whose name starts with "NO-" / "no_"
# is also treated as a violation automatically (see is_violation()).
DEFAULT_VIOLATION_CLASSES = ["NO-Hardhat", "NO-Safety Vest", "NO-Mask"]

PAGE_TITLE = "Site Safety Check"

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


def run_detection(model, bgr, conf, iou):
    """Return annotated RGB image and a list of (class_name, confidence)."""
    result = model.predict(bgr, conf=conf, iou=iou, verbose=False)[0]
    annotated = result.plot()  # BGR
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    detections = []
    if result.boxes is not None and len(result.boxes) > 0:
        for c, p in zip(result.boxes.cls.tolist(), result.boxes.conf.tolist()):
            detections.append((result.names[int(c)], float(p)))
    return annotated_rgb, detections


def summarize(detections, violation_set):
    counts = Counter(name for name, _ in detections)
    violations = {k: v for k, v in counts.items() if is_violation(k, violation_set)}
    return counts, violations


def render_verdict(detections, violation_set):
    if not detections:
        st.markdown(
            '<div class="verdict idle"><div class="dot"></div><div>'
            '<div class="title">Nothing detected</div>'
            '<div class="sub">Try a clearer image or lower the confidence threshold.</div>'
            "</div></div>",
            unsafe_allow_html=True,
        )
        return
    counts, violations = summarize(detections, violation_set)
    if violations:
        total = sum(violations.values())
        items = ", ".join(f"{v} × {k}" for k, v in violations.items())
        st.markdown(
            f'<div class="verdict unsafe"><div class="dot"></div><div>'
            f'<div class="title">{total} safety violation{"s" if total != 1 else ""} found</div>'
            f'<div class="sub">{items}</div></div></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="verdict safe"><div class="dot"></div><div>'
            '<div class="title">No violations found</div>'
            '<div class="sub">All detected workers are wearing required gear.</div>'
            "</div></div>",
            unsafe_allow_html=True,
        )


def render_metrics(detections, violation_set):
    counts, violations = summarize(detections, violation_set)
    c1, c2, c3 = st.columns(3)
    c1.metric("Objects detected", sum(counts.values()))
    c2.metric("Violations", sum(violations.values()))
    avg = np.mean([p for _, p in detections]) if detections else 0
    c3.metric("Avg. confidence", f"{avg:.0%}")


def render_table(detections, violation_set):
    if not detections:
        return
    counts, _ = summarize(detections, violation_set)
    confs = {}
    for n, p in detections:
        confs.setdefault(n, []).append(p)
    rows = [
        {
            "Class": n,
            "Count": c,
            "Avg. confidence": f"{np.mean(confs[n]):.0%}",
            "Status": "Violation" if is_violation(n, violation_set) else "OK",
        }
        for n, c in sorted(counts.items(), key=lambda x: -x[1])
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def to_png_bytes(rgb) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    return buf.getvalue()


def reencode_for_browser(src: str, dst: str) -> bool:
    """Convert to H.264 so st.video can play it. Needs ffmpeg on PATH."""
    if shutil.which("ffmpeg") is None:
        return False
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", src, "-vcodec", "libx264",
             "-pix_fmt", "yuv420p", dst],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        return True
    except Exception:
        return False


# ----------------------------------------------------------------------------
# SIDEBAR
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Settings")

    uploaded_weights = st.file_uploader(
        "Custom model weights (.pt)", type=["pt"],
        help="Leave empty to use the default model path.",
    )
    model_path = DEFAULT_MODEL_PATH
    if uploaded_weights is not None:
        tmp_dir = tempfile.mkdtemp()
        model_path = os.path.join(tmp_dir, uploaded_weights.name)
        with open(model_path, "wb") as f:
            f.write(uploaded_weights.getbuffer())

    conf = st.slider("Confidence threshold", 0.05, 0.95, 0.40, 0.05)
    iou = st.slider("Overlap threshold (IoU)", 0.1, 0.95, 0.45, 0.05)

    st.markdown("### Violation classes")
    model_ok = os.path.exists(model_path)
    all_names = []
    model = None
    if model_ok:
        try:
            model = load_model(model_path)
            all_names = list(model.names.values())
        except Exception as e:
            st.error(f"Could not load model: {e}")
    violation_choice = st.multiselect(
        "Classes counted as unsafe",
        options=all_names or DEFAULT_VIOLATION_CLASSES,
        default=[c for c in DEFAULT_VIOLATION_CLASSES if c in all_names]
        if all_names else DEFAULT_VIOLATION_CLASSES,
        help="Classes starting with “NO-” are always counted as unsafe.",
    )
    violation_set = set(violation_choice)

    if model is not None:
        with st.expander("Classes this model detects"):
            st.write(", ".join(all_names))

# ----------------------------------------------------------------------------
# HEADER
# ----------------------------------------------------------------------------
st.markdown(
    f"""
    <div class="hero">
        <h1>{PAGE_TITLE}</h1>
        <p>Upload a site photo, video or camera snapshot. The model checks for hard hats,
        safety vests and other required gear, then flags anything missing.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if model is None:
    st.warning(
        f"Model not found at `{DEFAULT_MODEL_PATH}`. Place your trained weights "
        "next to `app.py` or upload them in the sidebar."
    )
    st.stop()

tab_img, tab_vid, tab_cam = st.tabs(["Image", "Video", "Camera"])

# ----------------------------------------------------------------------------
# IMAGE TAB
# ----------------------------------------------------------------------------
def image_flow(pil_img: Image.Image, key: str):
    bgr = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)
    with st.spinner("Checking image…"):
        annotated, dets = run_detection(model, bgr, conf, iou)

    render_verdict(dets, violation_set)
    left, right = st.columns(2)
    left.image(pil_img, caption="Original", use_container_width=True)
    right.image(annotated, caption="Detections", use_container_width=True)

    render_metrics(dets, violation_set)
    render_table(dets, violation_set)
    st.download_button(
        "Download annotated image",
        data=to_png_bytes(annotated),
        file_name="safety_check.png",
        mime="image/png",
        key=f"dl_{key}",
    )


with tab_img:
    files = st.file_uploader(
        "Drop a site photo here", type=["jpg", "jpeg", "png", "webp"],
        key="img_up",
    )
    if files:
        image_flow(Image.open(files), "img")
    else:
        render_verdict([], violation_set)

# ----------------------------------------------------------------------------
# VIDEO TAB
# ----------------------------------------------------------------------------
with tab_vid:
    vid = st.file_uploader(
        "Drop a site video here", type=["mp4", "mov", "avi", "mkv"], key="vid_up"
    )
    frame_skip = st.select_slider(
        "Speed vs. detail",
        options=[1, 2, 3, 5],
        value=2,
        format_func=lambda x: "Every frame" if x == 1 else f"Every {x}rd frame"
        if x == 3 else f"Every {x}th frame" if x > 3 else f"Every {x}nd frame",
        help="Skipping frames makes processing faster. Skipped frames reuse the last result.",
    )

    if vid and st.button("Process video", key="run_vid"):
        src = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        src.write(vid.read())
        src.close()

        cap = cv2.VideoCapture(src.name)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        raw_out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
        writer = cv2.VideoWriter(
            raw_out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
        )

        verdict_slot = st.empty()
        preview = st.empty()
        bar = st.progress(0.0, text="Starting…")

        all_counts = Counter()
        frames_with_violation = 0
        processed = 0
        last_rgb, last_dets = None, []
        idx = 0
        t0 = time.time()

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % frame_skip == 0 or last_rgb is None:
                last_rgb, last_dets = run_detection(model, frame, conf, iou)
                processed += 1
                all_counts.update(n for n, _ in last_dets)
                if any(is_violation(n, violation_set) for n, _ in last_dets):
                    frames_with_violation += 1
                preview.image(last_rgb, use_container_width=True)
                with verdict_slot.container():
                    render_verdict(last_dets, violation_set)
            writer.write(cv2.cvtColor(last_rgb, cv2.COLOR_RGB2BGR))
            idx += 1
            bar.progress(min(idx / total, 1.0), text=f"Frame {idx} of {total}")

        cap.release()
        writer.release()
        bar.empty()

        elapsed = time.time() - t0
        st.success(f"Done. Checked {processed} frames in {elapsed:.1f}s.")

        m1, m2, m3 = st.columns(3)
        m1.metric("Frames checked", processed)
        m2.metric("Frames with violations", frames_with_violation)
        m3.metric(
            "Violation rate",
            f"{(frames_with_violation / processed):.0%}" if processed else "0%",
        )

        if all_counts:
            st.dataframe(
                pd.DataFrame(
                    [{"Class": k, "Total detections": v,
                      "Status": "Violation" if is_violation(k, violation_set) else "OK"}
                     for k, v in all_counts.most_common()]
                ),
                use_container_width=True, hide_index=True,
            )

        final_path = raw_out.replace(".mp4", "_h264.mp4")
        playable = reencode_for_browser(raw_out, final_path)
        out_path = final_path if playable else raw_out
        if playable:
            preview.empty()
            st.video(out_path)
        with open(out_path, "rb") as f:
            st.download_button(
                "Download annotated video", f.read(),
                file_name="safety_check.mp4", mime="video/mp4",
            )
        if not playable:
            st.caption(
                "Install ffmpeg to enable in-browser playback of the result. "
                "The download works either way."
            )

# ----------------------------------------------------------------------------
# CAMERA TAB
# ----------------------------------------------------------------------------
with tab_cam:
    st.caption("Take a snapshot with your device camera and check it instantly.")
    snap = st.camera_input("Camera", label_visibility="collapsed")
    if snap:
        image_flow(Image.open(snap), "cam")