"""
Simple YOLO Object Detection App
--------------------------------
Uses Ultralytics YOLO26 (latest) with a Streamlit UI.

Modes:
  1. Live webcam – continuous video with detections drawn in real time
  2. Images      – upload one or more images from your laptop
"""

import av
import numpy as np
import streamlit as st
from PIL import Image
from streamlit_webrtc import RTCConfiguration, WebRtcMode, webrtc_streamer
from ultralytics import YOLO


# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="YOLO Object Detection",
    page_icon="🔍",
    layout="centered",
)

st.title("YOLO Object Detection")
st.caption("Latest Ultralytics YOLO26 — live webcam or image upload")


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
# Shared settings (used by both live video and image upload)
# Stored in a dict so the live-video callback can read the latest value
# even while the WebRTC thread is running.
# ---------------------------------------------------------------------------
st.sidebar.header("Settings")

confidence = st.sidebar.slider(
    "Confidence threshold",
    min_value=0.1,
    max_value=1.0,
    value=0.25,
    step=0.05,
)

# Mutable holder so the live callback always sees the current slider value
settings = st.session_state.setdefault("settings", {})
settings["confidence"] = confidence

mode = st.sidebar.radio(
    "Input source",
    options=["Live webcam", "Upload images"],
)


# ---------------------------------------------------------------------------
# Helper: run detection on a BGR numpy frame (OpenCV / WebRTC format)
# ---------------------------------------------------------------------------
def detect_bgr(frame_bgr: np.ndarray, conf: float) -> np.ndarray:
    """
    Run YOLO on one BGR frame and return an annotated BGR frame.

    Args:
        frame_bgr: H×W×3 uint8 image in BGR order
        conf:      minimum confidence to keep a detection

    Returns:
        Annotated BGR frame with boxes and labels drawn on it
    """
    results = model.predict(frame_bgr, conf=conf, verbose=False)
    return results[0].plot()  # already BGR with drawings


# ---------------------------------------------------------------------------
# Helper: run detection on a PIL image (for the upload mode)
# ---------------------------------------------------------------------------
def detect_pil(image: Image.Image) -> Image.Image:
    """Run YOLO on a PIL RGB image; return an annotated PIL RGB image."""
    # Convert RGB → BGR for the shared helper, then back to RGB for display
    rgb = np.array(image)
    bgr = rgb[:, :, ::-1]
    annotated_bgr = detect_bgr(bgr, settings["confidence"])
    annotated_rgb = annotated_bgr[:, :, ::-1]
    return Image.fromarray(annotated_rgb)


# ---------------------------------------------------------------------------
# Mode 1: Live webcam (continuous video in + detections out)
# streamlit-webrtc streams the camera over WebRTC and calls our callback
# on every frame so YOLO can annotate and send the result back live.
# ---------------------------------------------------------------------------
if mode == "Live webcam":
    st.subheader("Live webcam")
    st.write(
        "Click **START**, allow camera access, then watch detections "
        "update on the live stream."
    )

    # Public STUN server helps the browser connect (needed on cloud deploys)
    rtc_config = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

    def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
        """
        Called for every webcam frame.
        Input  → raw camera frame
        Output → same frame with YOLO boxes/labels drawn on it
        """
        # Convert the WebRTC frame to a numpy BGR image
        img = frame.to_ndarray(format="bgr24")

        # Run YOLO and draw boxes
        annotated = detect_bgr(img, settings["confidence"])

        # Send the annotated frame back to the browser
        return av.VideoFrame.from_ndarray(annotated, format="bgr24")

    webrtc_streamer(
        key="yolo-live",
        mode=WebRtcMode.SENDRECV,  # send camera up, receive annotated video
        rtc_configuration=rtc_config,
        video_frame_callback=video_frame_callback,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,  # process frames without blocking the UI
    )


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
                result = detect_pil(image)

            st.markdown(f"**{file.name}**")
            col1, col2 = st.columns(2)
            col1.image(image, caption="Original", use_container_width=True)
            col2.image(result, caption="Detections", use_container_width=True)
            st.divider()
