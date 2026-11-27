import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO

st.set_page_config(
    page_title="Smoking Detection — Public Safety",
    page_icon="🚭",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = str(BASE_DIR / "best.pt")


@st.cache_resource
def load_model(path: str) -> YOLO:
    return YOLO(path)


# Main Header & Subheading
st.title("🚭 Smoking Detection App")
st.subheader("🏙️ Smoking Detection on Public Areas")
st.markdown(
    "<div style='text-align: right; font-size: 0.9rem; color: #ff4b4b; font-weight: 600; margin-top: -10px; margin-bottom: 16px;'>"
    "🚫 Smoking is prohibited in publis areas"
    "</div>",
    unsafe_allow_html=True,
)

# Caution Board
st.markdown(
    """
    <div style="
        background-color: #fff3cd;
        border: 3px solid #f59e0b;
        border-left: 12px solid #dc2626;
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 22px;
        color: #1f2937;
    ">
        <div style="font-size: 1.15rem; font-weight: 800; color: #b91c1c; letter-spacing: 0.5px;">
            ⚠️ CAUTION BOARD 🚭
        </div>
        <div style="font-size: 1.05rem; font-weight: 700; margin-top: 6px; color: #111827;">
            🛑 "Smoking is injuries to health" 🫁
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Sidebar controls
st.sidebar.header("⚙️ Inference Settings")
conf_threshold = st.sidebar.slider(
    "🎯 Confidence Threshold", min_value=0.05, max_value=1.0, value=0.25, step=0.05
)
iou_threshold = st.sidebar.slider(
    "📐 IoU Threshold", min_value=0.1, max_value=1.0, value=0.45, step=0.05
)

try:
    model = load_model(MODEL_PATH)
except Exception as e:
    st.error(f"❌ Failed to load model from `{MODEL_PATH}`: {e}")
    st.stop()

# Tabs for Image and Video Detection
tab_img, tab_vid = st.tabs(["🖼️ Image Detection", "🎥 Video Detection"])

# ==================== IMAGE DETECTION TAB ====================
with tab_img:
    uploaded_file = st.file_uploader(
        "📤 Upload a public area image for inspection",
        type=["jpg", "jpeg", "png", "webp"],
        key="img_uploader",
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        img_array = np.array(image)

        with st.spinner("🔍 Scanning image for smoking activity..."):
            results = model.predict(
                source=img_array,
                conf=conf_threshold,
                iou=iou_threshold,
                verbose=False,
            )

        result = results[0]
        annotated_bgr = result.plot()
        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🖼️ Original Image")
            st.image(image, use_container_width=True)
        with col2:
            st.subheader("🎯 Detection Result")
            st.image(annotated_rgb, use_container_width=True)

        # Alert Banner
        boxes = result.boxes
        names = result.names
        detected_classes = (
            [names[int(cls_id)] for cls_id in boxes.cls] if boxes is not None else []
        )

        if "cigarette" in detected_classes:
            st.error("🚨 ALERT: Cigarette detected! Smoking is strictly prohibited here. 🚭")
        else:
            st.success("✅ Clear: No cigarette detected in this area.")

        if boxes is not None and len(boxes) > 0:
            st.subheader("📋 Detected Objects")
            detections_data = [
                {
                    "Label": f"{'🚬' if names[int(box.cls[0])] == 'cigarette' else '🧑'} {names[int(box.cls[0])].title()}",
                    "Confidence": f"{float(box.conf[0]):.2%}",
                }
                for box in boxes
            ]
            st.table(detections_data)

# ==================== VIDEO DETECTION TAB ====================
with tab_vid:
    uploaded_video = st.file_uploader(
        "🎬 Upload a public area CCTV / surveillance video",
        type=["mp4", "avi", "mov", "mkv"],
        key="vid_uploader",
    )

    if uploaded_video is not None:
        st.video(uploaded_video)

        if st.button("▶️ Start Smoking Detection on Video", type="primary"):
            suffix = os.path.splitext(uploaded_video.name)[1] or ".mp4"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tfile:
                tfile.write(uploaded_video.read())
                input_video_path = tfile.name

            output_video_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name

            cap = cv2.VideoCapture(input_video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

            progress_bar = st.progress(0)
            status_text = st.empty()
            live_frame_placeholder = st.empty()

            frame_idx = 0
            smoking_frames = 0
            cigarette_detections = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                results = model.predict(
                    source=frame,
                    conf=conf_threshold,
                    iou=iou_threshold,
                    verbose=False,
                )
                result = results[0]
                annotated_frame = result.plot()

                boxes = result.boxes
                names = result.names
                if boxes is not None and len(boxes) > 0:
                    cls_names = [names[int(cls_id)] for cls_id in boxes.cls]
                    cig_count = cls_names.count("cigarette")
                    if cig_count > 0:
                        smoking_frames += 1
                        cigarette_detections += cig_count

                out.write(annotated_frame)
                frame_idx += 1

                if frame_idx % 5 == 0 or frame_idx == total_frames:
                    if total_frames > 0:
                        progress_bar.progress(min(frame_idx / total_frames, 1.0))
                        status_text.text(f"🔍 Processing frame {frame_idx} / {total_frames}...")
                    live_frame_placeholder.image(
                        cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB),
                        caption="📡 Live Detection Feed",
                        use_container_width=True,
                    )

            cap.release()
            out.release()
            progress_bar.empty()
            status_text.empty()

            if smoking_frames > 0:
                st.error(
                    f"🚨 SMOKING DETECTED IN VIDEO! Cigarette spotted in **{smoking_frames}** frames "
                    f"({cigarette_detections} total cigarette detections). 🚭"
                )
            else:
                st.success("✅ Clear: No smoking activity detected in the video.")

            with open(output_video_path, "rb") as f:
                st.download_button(
                    label="⬇️ Download Annotated Video",
                    data=f,
                    file_name="smoking_detected_output.mp4",
                    mime="video/mp4",
                )

            os.remove(input_video_path)
            os.remove(output_video_path)

# Footer Credit
st.markdown("---")
st.markdown(
    "<div style='text-align: center; font-size: 0.95rem; color: #9ca3af; padding: 8px 0;'>"
    "✨ Created by <strong>NAGURU DARBAR</strong> 🚀"
    "</div>",
    unsafe_allow_html=True,
)