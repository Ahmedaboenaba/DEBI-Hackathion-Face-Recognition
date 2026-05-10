"""
app.py — Streamlit Face Recognition App

Uses streamlit-webrtc to access the webcam in the browser,
runs the same recognition logic as main.py on each frame,
and displays the annotated video feed.

Run with:  python -m streamlit run app.py 
"""

import json
import os

import av
import cv2
import numpy as np
import streamlit as st
import face_recognition
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

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


# ── Video Processor ────────────────────────────────────────────────────────
class FaceRecognitionProcessor(VideoProcessorBase):
    """Processes each webcam frame through the face recognition pipeline."""

    def __init__(self):
        self.known_names, self.known_encodings = load_database(DB_PATH)

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")

        scale = int(1 / RESIZE_SCALE)

        # ── Resize, detect, encode ─────────────────────────────────
        small = cv2.resize(img, (0, 0), fx=RESIZE_SCALE, fy=RESIZE_SCALE)
        rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        face_locs = face_recognition.face_locations(rgb_small)
        face_encs = face_recognition.face_encodings(rgb_small, face_locs)

        # ── Match & annotate each face ─────────────────────────────
        for (top, right, bottom, left), enc in zip(face_locs, face_encs):
            name = "Unknown"
            confidence = 0.0
            color = (0, 0, 255)

            if self.known_encodings:
                matches = face_recognition.compare_faces(
                    self.known_encodings, enc, tolerance=TOLERANCE
                )
                distances = face_recognition.face_distance(
                    self.known_encodings, enc
                )
                best_idx = int(np.argmin(distances))
                best_distance = distances[best_idx]

                if matches[best_idx]:
                    name = self.known_names[best_idx]
                    confidence = (1 - best_distance) * 100
                    color = (0, 255, 0)

            # Scale back to full resolution
            top *= scale
            right *= scale
            bottom *= scale
            left *= scale

            # Bounding box
            cv2.rectangle(img, (left, top), (right, bottom), color, 2)

            # Label
            if name != "Unknown":
                label = f"{name} — {confidence:.0f}%"
            else:
                label = "Unknown"

            label_y = top - 10 if top - 10 > 10 else top + 20
            cv2.rectangle(
                img, (left, label_y - 18), (right, label_y + 4),
                color, cv2.FILLED,
            )
            cv2.putText(
                img, label, (left + 4, label_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
            )

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# ── Streamlit UI ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Face Recognition",
    page_icon="👤",
    layout="centered",
)

st.title("👤 Real-Time Face Recognition")
st.markdown(
    "This app detects faces via your webcam and matches them against "
    "the enrolled database. Green = recognized, Red = unknown."
)

# Show enrolled count
names, _ = load_database(DB_PATH)
st.sidebar.header("📋 Enrolled Faces")
if names:
    for i, n in enumerate(names, 1):
        st.sidebar.write(f"{i}. {n}")
else:
    st.sidebar.warning("No faces enrolled yet.")

st.sidebar.markdown("---")
st.sidebar.info(f"**Tolerance:** {TOLERANCE}\n\n**Resize:** {int(RESIZE_SCALE*100)}%")

# WebRTC streamer
webrtc_streamer(
    key="face-recognition",
    video_processor_factory=FaceRecognitionProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)
