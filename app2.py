"""
app2.py — Streamlit Face Recognition App (Local Webcam Edition)

Uses OpenCV VideoCapture directly — no webrtc dependency needed.
Simple, reliable, works locally out of the box.

Run with:  streamlit run app2.py
"""

import json
import os
import time

import cv2
import numpy as np
import streamlit as st
import face_recognition

# ── Configuration ──────────────────────────────────────────────────────────
DB_PATH = os.environ.get(
    "DB_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "database.json"),
)
TOLERANCE = 0.5
RESIZE_SCALE = 0.25


# ── Load database ─────────────────────────────────────────────────────────
@st.cache_data
def load_database(path: str):
    """Load enrolled faces from database.json (cached)."""
    if os.path.exists(path):
        with open(path, "r") as f:
            db = json.load(f)
    else:
        db = {"persons": []}

    names = [p["name"] for p in db["persons"]]
    encodings = [np.array(p["embedding"]) for p in db["persons"]]
    return names, encodings


def recognize_and_annotate(frame, known_names, known_encodings):
    """Run face recognition on a BGR frame and return the annotated frame."""
    scale = int(1 / RESIZE_SCALE)

    # Resize for speed
    small = cv2.resize(frame, (0, 0), fx=RESIZE_SCALE, fy=RESIZE_SCALE)
    rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

    face_locs = face_recognition.face_locations(rgb_small)
    face_encs = face_recognition.face_encodings(rgb_small, face_locs)

    results = []

    for (top, right, bottom, left), enc in zip(face_locs, face_encs):
        name = "Unknown"
        confidence = 0.0
        color = (0, 0, 255)  # Red

        if known_encodings:
            matches = face_recognition.compare_faces(
                known_encodings, enc, tolerance=TOLERANCE
            )
            distances = face_recognition.face_distance(known_encodings, enc)
            best_idx = int(np.argmin(distances))
            best_distance = distances[best_idx]

            if matches[best_idx]:
                name = known_names[best_idx]
                confidence = (1 - best_distance) * 100
                color = (0, 255, 0)  # Green

        # Scale back to full resolution
        top *= scale
        right *= scale
        bottom *= scale
        left *= scale

        results.append({"name": name, "confidence": confidence})

        # Draw bounding box
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

        # Draw label
        if name != "Unknown":
            label = f"{name} - {confidence:.0f}%"
        else:
            label = "Unknown"

        label_y = top - 10 if top - 10 > 10 else top + 20
        cv2.rectangle(
            frame, (left, label_y - 18), (right, label_y + 4),
            color, cv2.FILLED,
        )
        cv2.putText(
            frame, label, (left + 4, label_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
        )

    return frame, results


# ── Streamlit Page Config ──────────────────────────────────────────────────
st.set_page_config(
    page_title="Face Recognition",
    page_icon="👤",
    layout="wide",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        color: white;
        text-align: center;
    }
    .main-header h1 { margin: 0; font-size: 2rem; }
    .main-header p  { margin: 0.3rem 0 0; opacity: 0.9; }
    .status-card {
        background: #1e1e2e;
        border-radius: 10px;
        padding: 1rem;
        border: 1px solid #333;
        text-align: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #667eea;
    }
    .face-detected { color: #4ade80 !important; }
    .face-unknown  { color: #f87171 !important; }
</style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>👤 Real-Time Face Recognition</h1>
    <p>Detects and identifies enrolled faces via webcam</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────
known_names, known_encodings = load_database(DB_PATH)

st.sidebar.header("📋 Enrolled Faces")
if known_names:
    for i, n in enumerate(known_names, 1):
        st.sidebar.write(f"**{i}.** {n}")
else:
    st.sidebar.warning("No faces enrolled yet.")

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Settings")
st.sidebar.info(
    f"**Tolerance:** {TOLERANCE}\n\n"
    f"**Resize:** {int(RESIZE_SCALE * 100)}%\n\n"
    f"**Model:** dlib ResNet-29\n\n"
    f"**Metric:** L2 Euclidean Distance"
)

# ── Main Layout ────────────────────────────────────────────────────────────
col_video, col_info = st.columns([3, 1])

with col_info:
    st.markdown("### 📊 Live Stats")
    fps_placeholder = st.empty()
    faces_placeholder = st.empty()
    details_placeholder = st.empty()

with col_video:
    run = st.toggle("🎥 Start Camera", value=False)
    frame_placeholder = st.empty()

# ── Camera Loop ────────────────────────────────────────────────────────────
if run:
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("❌ Cannot open webcam. Check your camera connection.")
    else:
        st.toast("📷 Camera started!", icon="✅")

        while run:
            ret, frame = cap.read()
            if not ret:
                st.warning("Failed to read from webcam.")
                break

            t0 = time.time()

            # Run recognition
            annotated, results = recognize_and_annotate(
                frame, known_names, known_encodings
            )

            fps = 1.0 / max(time.time() - t0, 0.001)

            # Convert BGR → RGB for Streamlit display
            rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(rgb, channels="RGB", use_container_width=True)

            # Update live stats
            fps_placeholder.metric("⚡ FPS", f"{fps:.1f}")
            faces_placeholder.metric("🔍 Faces Detected", len(results))

            if results:
                details_md = ""
                for r in results:
                    if r["name"] != "Unknown":
                        details_md += f"- 🟢 **{r['name']}** — {r['confidence']:.0f}%\n"
                    else:
                        details_md += f"- 🔴 **Unknown**\n"
                details_placeholder.markdown(details_md)
            else:
                details_placeholder.markdown("*No faces in frame*")

        cap.release()
else:
    frame_placeholder.info("👆 Toggle **Start Camera** above to begin.")
