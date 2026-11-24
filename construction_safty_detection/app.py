import os
import tempfile

import numpy as np
from PIL import Image
import streamlit as st
from ultralytics import YOLO

st.set_page_config(
    page_title="Construction Safety Detection",
    page_icon="🦺",
    layout="wide",
)

st.title("🦺 Construction Safety Detection")
st.write("Upload an image or video to run safety gear and hazard detection using `best.pt`.")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best.pt")


@st.cache_resource
def load_model(model_path: str) -> YOLO:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    return YOLO(model_path)


st.sidebar.header("Settings")
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.05, 1.0, 0.25, 0.05)

try:
    model = load_model(MODEL_PATH)
except Exception as e:
    st.error(f"Failed to load `best.pt`: {e}")
    st.stop()

mode = st.radio("Select Input Type", ["Image", "Video"], horizontal=True)

if mode == "Image":
    uploaded_file = st.file_uploader(
        "Upload a construction site image",
        type=["jpg", "jpeg", "png", "webp"],
    )

    if uploaded_file is not None:
        try:
            pil_image = Image.open(uploaded_file).convert("RGB")
        except Exception as e:
            st.error(f"Could not read the uploaded image: {e}")
            st.stop()

        rgb_image = np.array(pil_image)

        with st.spinner("Running detection..."):
            results = model.predict(source=rgb_image, conf=conf_threshold, verbose=False)
            annotated_rgb = results[0].plot()

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original Image")
            st.image(rgb_image, use_container_width=True)
        with col2:
            st.subheader("Detected Output")
            st.image(annotated_rgb, use_container_width=True)

        boxes = results[0].boxes
        if boxes is not None and len(boxes) > 0:
            class_ids = boxes.cls.cpu().numpy().astype(int)
            names = results[0].names
            counts = {}
            for cid in class_ids:
                label = names.get(cid, str(cid))
                counts[label] = counts.get(label, 0) + 1

            st.subheader("Detection Summary")
            st.json(counts)
        else:
            st.info("No objects detected above the selected confidence threshold.")

elif mode == "Video":
    try:
        import cv2
    except Exception as e:
        st.error(f"OpenCV (`cv2`) failed to load for video processing: {e}")
        st.stop()

    uploaded_video = st.file_uploader(
        "Upload a construction site video",
        type=["mp4", "avi", "mov", "mkv"],
    )

    if uploaded_video is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as input_tmp:
            input_tmp.write(uploaded_video.read())
            input_path = input_tmp.name

        output_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        output_path = output_tmp.name
        output_tmp.close()

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            st.error("Could not open the uploaded video.")
            os.remove(input_path)
            st.stop()

        fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        writer = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        progress_bar = st.progress(0)
        status_text = st.empty()
        frame_idx = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_idx += 1
                results = model.predict(source=frame, conf=conf_threshold, verbose=False)
                writer.write(results[0].plot())
                if total_frames > 0:
                    progress_bar.progress(min(frame_idx / total_frames, 1.0))
                status_text.text(
                    f"Processing frame {frame_idx}"
                    + (f" / {total_frames}" if total_frames > 0 else "")
                )
        finally:
            cap.release()
            writer.release()
            if os.path.exists(input_path):
                os.remove(input_path)

        st.success("Video processing complete!")
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            st.video(output_path)
            with open(output_path, "rb") as f:
                st.download_button(
                    label="Download Processed Video",
                    data=f.read(),
                    file_name="safety_detection_output.mp4",
                    mime="video/mp4",
                )
            os.remove(output_path)