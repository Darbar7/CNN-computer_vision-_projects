import os
import numpy as np
from PIL import Image, ImageDraw
import streamlit as st
import torch

st.set_page_config(
    page_title="Construction Safety Detection",
    page_icon="🦺",
    layout="wide",
)

st.title("🦺 Construction Safety Detection")
st.write("Upload a construction site image to run detection using `best.pt`.")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best.pt")


@st.cache_resource
def load_model(model_path: str):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    model = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    if hasattr(model, "float"):
        model = model.float()
    model.eval()
    return model


st.sidebar.header("Settings")
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.05, 1.0, 0.25, 0.05)

try:
    model = load_model(MODEL_PATH)
except Exception as e:
    st.error(f"Failed to load `best.pt`: {e}")
    st.stop()

uploaded_file = st.file_uploader(
    "Upload a construction site image",
    type=["jpg", "jpeg", "png", "webp"],
)

if uploaded_file is not None:
    pil_image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(pil_image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)

    with st.spinner("Running inference..."):
        with torch.no_grad():
            outputs = model(tensor)

    annotated = pil_image.copy()
    draw = ImageDraw.Draw(annotated)
    names = getattr(model, "names", {}) or {}
    counts = {}

    # Handle standard YOLO / detection tensor output if present
    pred = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
    if torch.is_tensor(pred) and pred.ndim == 3 and pred.shape[-1] >= 6:
        rows = pred[0]
        for row in rows:
            conf = float(row[4].item())
            if conf >= conf_threshold:
                x1, y1, x2, y2 = [float(v.item()) for v in row[:4]]
                cls_id = int(torch.argmax(row[5:]).item())
                label = names.get(cls_id, str(cls_id))
                counts[label] = counts.get(label, 0) + 1
                draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
                draw.text((x1, max(0.0, y1 - 14.0)), f"{label} {conf:.2f}", fill="red")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Image")
        st.image(pil_image, use_container_width=True)
    with col2:
        st.subheader("Output Image")
        st.image(annotated, use_container_width=True)

    if counts:
        st.subheader("Detection Summary")
        st.json(counts)
    else:
        st.info("Inference completed. If boxes aren't drawn, your `best.pt` uses a custom output format and `ultralytics` + `packages.txt` (Option A) is the cleanest path.")