# YOLO Object Detection

A **simple** object detection app using the latest **Ultralytics YOLO26** model.

- **Live webcam** — continuous video with detections drawn in real time
- **Upload images** — pick one or more images from your laptop

---

## Quick start (local)

### 1. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the app

```bash
streamlit run app.py
```

Open the URL shown in the terminal (usually `http://localhost:8501`).

The first run downloads `yolo26n.pt` (YOLO26 nano) automatically.

---

## How to use

1. In the sidebar, pick **Live webcam** or **Upload images**.
2. Adjust the **confidence threshold** if you want fewer/more detections.
3. **Live webcam:** click **START**, allow camera access — boxes update on the live stream.
4. **Upload images:** select files — originals and detections appear side by side.

---

## Deploy with Docker

```bash
# Build
docker build -t yolo-detect .

# Run (open http://localhost:8501)
docker run -p 8501:8501 yolo-detect
```

---

## Deploy on Streamlit Community Cloud

1. Push this repo to GitHub (already done if you cloned it).
2. Go to [share.streamlit.io](https://share.streamlit.io).
3. Click **New app** → select this repo → set main file to `app.py`.
4. Deploy. `packages.txt` installs the system libs OpenCV needs.

---

## Project layout

```
app.py              # Entire app (Streamlit + YOLO)
requirements.txt    # Python packages
packages.txt        # Linux apt packages (Streamlit Cloud)
Dockerfile          # Container deployment
.streamlit/         # Streamlit config
```

---

## Notes

- Model: `yolo26n.pt` (nano) — fast and small. Swap to `yolo26s.pt` / `yolo26m.pt` in `app.py` for higher accuracy.
- Live webcam uses `streamlit-webrtc` (browser camera in, annotated video out).
- Weights are downloaded on first use and are git-ignored (`*.pt`).

