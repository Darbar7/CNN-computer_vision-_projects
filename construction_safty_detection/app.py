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
# CONFIG
# ----------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH = os.path.join(BASE_DIR, "best.pt")

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
# SAFE, SCOPED STYLING (no wildcard [class*="css"] overrides)
# ----------------------------------------------------------------------------
CSS = """
<style>
.hero {
    background: #1E2A35;
    border-radius: 8px;
    padding: 1.5rem 2rem;
    margin-bottom: 1.2rem;
    border-bottom: 6px solid #F2B705;
}
.hero h1 {
    color: #FFFFFF;
    font-size: 2.2rem;
    margin: 0;
}
.hero p {
    color: #CBD5E1;
    margin: 0.4rem 0 0 0;
    font-size: 1rem;
}
.verdict {
    border-radius: 6px;
    padding: 1rem 1.25rem;
    margin: 0.25rem 0 1rem 0;
    border: 1px solid #D8DDE2;
    background: #FFFFFF;
    color: #1E2A35;
}
.verdict.safe { border-left: 8px solid #1F8A5B; }
.verdict.unsafe { border-left: 8px solid #D9381E; }
.verdict.idle { border-left: 8px solid #94A3B8; }
.verdict .title { font-weight: 700; font-size: 1.25rem; }
.verdict .sub { color: #5B6875; font-size: 0.95rem; margin-top: 0.2rem; }
.creator-label {
    margin-top: 2.5rem;
    padding: 0.85rem 1rem;
    border-top: 1px solid #D8DDE2;
    color: #5B6875;
    font-size: 0.92rem;
    text-align: center;
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# RENDER HEADER IMMEDIATELY SO PAGE IS NEVER BLANK
# ----------------------------------------------------------------------------
st.markdown(
    f"""
    <div class="hero">
        <h1>🦺 {PAGE_TITLE}</h1>
        <p>Upload a site photo, video, or camera snapshot to check for helmet, safety belt, and smoking violations.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------------
# HELPERS
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading YOLO model…")
def load_model(path: str) -> YOLO:
    return YOLO(path)


def is_violation(name: str, violation_set: set) -> bool:
    n = name.lower()
    return name in violation_set or n.startswith("no-") or n.startswith("no_")


def prepare_rgb_image(pil_img: Image.Image) -> np.ndarray:
    rgb = pil_img.convert("RGB")
    if rgb.width <= 0 or rgb.height <= 0:
        raise ValueError("Uploaded image has invalid dimensions.")
    arr = np.asarray(rgb, dtype=np.uint8)
    if arr.ndim != 3 or arr.shape[2] != 3:
        arr = np.stack([arr[:, :, 0]] * 3, axis=-1).astype(np.uint8)
    return arr


def run_detection_rgb(model: YOLO, rgb_arr: np.ndarray, conf: float, iou: float, imgsz: int = 640):
    results = model.predict(
        source=rgb_arr,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        verbose=False,
    )
    result = results[0]
    annotated_bgr = result.plot()
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
    detections = []
    if result.boxes is not None and len(result.boxes) > 0:
        names = getattr(result, "names", None) or getattr(model, "names", {})
        for c, p in zip(result.boxes.cls.tolist(), result.boxes.conf.tolist()):
            cls_idx = int(c)
            label = names.get(cls_idx, str(cls_idx)) if isinstance(names, dict) else names[cls_idx]
            detections.append((label, float(p)))
    return annotated_rgb, detections


def summarize(detections, violation_set):
    counts = Counter(name for name, _ in detections)
    violations = {k: v for k, v in counts.items() if is_violation(k, violation_set)}
    return counts, violations


def render_verdict(detections, violation_set):
    if not detections:
        st.markdown(
            '<div class="verdict idle"><div class="title">Ready / Nothing detected</div>'
            '<div class="sub">Upload a photo or video, or lower the confidence threshold if objects were missed.</div></div>',
            unsafe_allow_html=True,
        )
        return
    _, violations = summarize(detections, violation_set)
    if violations:
        total = sum(violations.values())
        items = ", ".join(f"{v} × {k}" for k, v in violations.items())
        st.markdown(
            f'<div class="verdict unsafe"><div class="title">⚠️ {total} safety violation{"s" if total != 1 else ""} found</div>'
            f'<div class="sub">{items}</div></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="verdict safe"><div class="title">✅ No violations found</div>'
            '<div class="sub">All detected workers comply with required safety rules.</div></div>',
            unsafe_allow_html=True,
        )


def render_metrics(detections, violation_set):
    counts, violations = summarize(detections, violation_set)
    c1, c2, c3 = st.columns(3)
    c1.metric("Objects detected", sum(counts.values()))
    c2.metric("Violations", sum(violations.values()))
    avg = np.mean([p for _, p in detections]) if detections else 0.0
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


def to_png_bytes(rgb: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    return buf.getvalue()


def reencode_for_browser(src: str, dst: str) -> bool:
    if shutil.which("ffmpeg") is None:
        return False
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", src, "-vcodec", "libx264", "-pix_fmt", "yuv420p", dst],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return os.path.exists(dst) and os.path.getsize(dst) > 0
    except Exception:
        return False


def render_creator_footer():
    st.markdown(
        f'<div class="creator-label">Created by <strong>{CREATOR_NAME}</strong></div>',
        unsafe_allow_html=True,
    )


# ----------------------------------------------------------------------------
# SIDEBAR & MODEL LOADING
# ----------------------------------------------------------------------------
with st.sidebar:
    st.header("Settings")

    uploaded_weights = st.file_uploader(
        "Custom model weights (.pt)",
        type=["pt"],
        help="Leave empty to use best.pt bundled with the app.",
    )
    model_path = DEFAULT_MODEL_PATH
    if uploaded_weights is not None:
        tmp_dir = tempfile.mkdtemp()
        model_path = os.path.join(tmp_dir, uploaded_weights.name)
        with open(model_path, "wb") as f:
            f.write(uploaded_weights.getbuffer())

    conf = st.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.05)
    iou = st.slider("Overlap threshold (IoU)", 0.10, 0.95, 0.45, 0.05)
    imgsz = st.select_slider("Inference image size (px)", options=[640, 960, 1280], value=640)

all_names = []
model = None
if os.path.exists(model_path):
    try:
        model = load_model(model_path)
        if isinstance(model.names, dict):
            all_names = [model.names[k] for k in sorted(model.names.keys())]
        else:
            all_names = list(model.names)
    except Exception as e:
        st.error(f"Could not load model `{model_path}`: {e}")

with st.sidebar:
    st.subheader("Violation classes")
    default_selected = (
        [c for c in all_names if is_violation(c, set(DEFAULT_VIOLATION_CLASSES))]
        if all_names
        else DEFAULT_VIOLATION_CLASSES
    )
    violation_choice = st.multiselect(
        "Classes counted as unsafe",
        options=all_names or DEFAULT_VIOLATION_CLASSES,
        default=default_selected,
    )
    violation_set = set(violation_choice)

    if model is not None:
        with st.expander("Model classes"):
            st.write(", ".join(all_names))

if model is None:
    st.warning(
        f"Model not found at `{DEFAULT_MODEL_PATH}`. Upload `best.pt` in the sidebar to begin."
    )
    render_creator_footer()
    st.stop()

# ----------------------------------------------------------------------------
# MAIN TABS
# ----------------------------------------------------------------------------
tab_img, tab_vid, tab_cam = st.tabs(["📷 Image", "🎥 Video", "📸 Camera"])


def image_flow(pil_img: Image.Image, key: str):
    try:
        rgb_arr = prepare_rgb_image(pil_img)
    except Exception as e:
        st.error(f"Invalid image: {e}")
        return

    with st.spinner("Checking image…"):
        try:
            annotated, dets = run_detection_rgb(model, rgb_arr, conf, iou, imgsz=imgsz)
        except Exception as e:
            st.error(f"Detection failed: {e}")
            return

    render_verdict(dets, violation_set)
    left, right = st.columns(2)
    left.image(rgb_arr, caption="Original", use_container_width=True)
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
        "Drop a site photo here",
        type=["jpg", "jpeg", "png", "webp"],
        key="img_up",
    )
    if files:
        with Image.open(files) as opened_img:
            image_flow(opened_img.copy(), "img")
    else:
        render_verdict([], violation_set)


with tab_vid:
    vid = st.file_uploader(
        "Drop a site video here",
        type=["mp4", "mov", "avi", "mkv"],
        key="vid_up",
    )
    frame_skip = st.select_slider(
        "Speed vs. detail",
        options=[1, 2, 3, 5],
        value=3,
        format_func=lambda x: "Every frame" if x == 1 else f"Every {x} frames",
    )

    if vid is not None:
        st.video(vid)

    if vid is not None and st.button("Process video", key="run_vid"):
        suffix = os.path.splitext(vid.name)[1] or ".mp4"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as src:
            src.write(vid.getvalue())
            src_path = src.name

        cap = cv2.VideoCapture(src_path)
        if not cap.isOpened():
            st.error("Unable to open the uploaded video file.")
            os.remove(src_path)
            st.stop()

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        if fps <= 0:
            fps = 25.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        raw_out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
        writer = cv2.VideoWriter(raw_out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        verdict_slot = st.empty()
        preview = st.empty()
        bar = st.progress(0.0, text="Starting…")

        all_counts = Counter()
        frames_with_violation = 0
        processed = 0
        last_rgb, last_dets = None, []
        idx = 0
        t0 = time.time()

        try:
            while True:
                ok, frame_bgr = cap.read()
                if not ok:
                    break
                if idx % frame_skip == 0 or last_rgb is None:
                    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    last_rgb, last_dets = run_detection_rgb(model, frame_rgb, conf, iou, imgsz=imgsz)
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
        except Exception as e:
            st.error(f"Video processing failed: {e}")
        finally:
            cap.release()
            writer.release()
            if os.path.exists(src_path):
                os.remove(src_path)

        elapsed = time.time() - t0
        st.success(f"Done. Checked {processed} frames in {elapsed:.1f}s.")

        m1, m2, m3 = st.columns(3)
        m1.metric("Frames checked", processed)
        m2.metric("Frames with violations", frames_with_violation)
        m3.metric("Violation rate", f"{(frames_with_violation / processed):.0%}" if processed else "0%")

        if all_counts:
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Class": k,
                            "Total detections": v,
                            "Status": "Violation" if is_violation(k, violation_set) else "OK",
                        }
                        for k, v in all_counts.most_common()
                    ]
                ),
                use_container_width=True,
                hide_index=True,
            )

        final_path = raw_out.replace(".mp4", "_h264.mp4")
        playable = reencode_for_browser(raw_out, final_path)
        out_path = final_path if playable else raw_out
        if playable:
            preview.empty()
            st.video(out_path)

        if os.path.exists(out_path):
            with open(out_path, "rb") as f:
                st.download_button(
                    "Download annotated video",
                    f.read(),
                    file_name="safety_check.mp4",
                    mime="video/mp4",
                )


with tab_cam:
    st.caption("Take a snapshot with your device camera and check it instantly.")
    snap = st.camera_input("Camera", label_visibility="collapsed")
    if snap:
        with Image.open(snap) as opened_snap:
            image_flow(opened_snap.copy(), "cam")

render_creator_footer()