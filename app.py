"""
Simple YOLO Object Detection App
--------------------------------
Uses Ultralytics YOLO26 (latest) with a Streamlit UI.

Modes:
  1. Webcam  – take a photo from your camera
  2. Images  – upload one or more images from your laptop
"""

import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np


# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="YOLO Object Detection",
    page_icon="🔍",
    layout="centered",
)

st.title("YOLO Object Detection")
st.caption("Latest Ultralytics YOLO26 — webcam or image upload")


# ---------------------------------------------------------------------------
# Load the model once and cache it (so it is not re-downloaded every rerun)
# "yolo26n.pt" = YOLO26 nano — smallest & fastest pretrained model
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model():
    """Download (first run) and load the YOLO26 nano model."""
    return YOLO("yolo26n.pt")


model = load_model()


# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------
st.sidebar.header("Settings")

# Confidence threshold: how sure the model must be to keep a detection
confidence = st.sidebar.slider(
    "Confidence threshold",
    min_value=0.1,
    max_value=1.0,
    value=0.25,
    step=0.05,
)

# Choose input source
mode = st.sidebar.radio(
    "Input source",
    options=["Webcam", "Upload images"],
)


# ---------------------------------------------------------------------------
# Helper: run detection on a PIL image and return the annotated image
# ---------------------------------------------------------------------------
def detect(image: Image.Image) -> Image.Image:
    """
    Run YOLO on one image.

    Args:
        image: PIL Image (RGB)

    Returns:
        Annotated PIL Image with boxes and labels drawn on it
    """
    # YOLO expects a numpy array (BGR or RGB both work; we pass RGB)
    results = model.predict(np.array(image), conf=confidence, verbose=False)

    # results[0].plot() returns a BGR numpy array with drawings
    annotated_bgr = results[0].plot()

    # Convert BGR → RGB for correct colors in Streamlit / PIL
    annotated_rgb = annotated_bgr[:, :, ::-1]
    return Image.fromarray(annotated_rgb)


# ---------------------------------------------------------------------------
# Mode 1: Webcam
# st.camera_input opens the browser camera and returns a single snapshot
# ---------------------------------------------------------------------------
if mode == "Webcam":
    st.subheader("Webcam")
    photo = st.camera_input("Take a photo")

    if photo is not None:
        image = Image.open(photo).convert("RGB")
        with st.spinner("Detecting objects..."):
            result = detect(image)

        # Show original and result side by side
        col1, col2 = st.columns(2)
        col1.image(image, caption="Original", use_container_width=True)
        col2.image(result, caption="Detections", use_container_width=True)


# ---------------------------------------------------------------------------
# Mode 2: Upload images from laptop
# accept_multiple_files=True lets you pick a whole set of images
# ---------------------------------------------------------------------------
else:
    st.subheader("Upload images")
    files = st.file_uploader(
        "Choose one or more images",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        accept_multiple_files=True,
    )

    if files:
        for file in files:
            image = Image.open(file).convert("RGB")
            with st.spinner(f"Detecting objects in {file.name}..."):
                result = detect(image)

            st.markdown(f"**{file.name}**")
            col1, col2 = st.columns(2)
            col1.image(image, caption="Original", use_container_width=True)
            col2.image(result, caption="Detections", use_container_width=True)
            st.divider()
