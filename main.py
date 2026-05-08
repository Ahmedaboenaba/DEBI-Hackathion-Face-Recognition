"""
main.py — Jupyter-Compatible Face Recognition Engine (Kaggle Edition)
Works inside Kaggle notebooks where getUserMedia is blocked by the iframe sandbox.

Two capture modes:
  1. Upload a photo via ipywidgets FileUpload
  2. One-shot camera capture using IPython.display.Javascript (prompts browser permission)

Both feed into process_frame(b64) → annotated b64 for display.
"""

import base64
import io
import json
import os

import cv2
import numpy as np
import face_recognition
from PIL import Image

# ── Configuration ──────────────────────────────────────────────────────────
DB_PATH = os.environ.get("DB_PATH", "/kaggle/working/database.json")
TOLERANCE = 0.5
RESIZE_SCALE = 0.25  # Process frames at 25% resolution for speed

# ── Load enrolled faces at startup ─────────────────────────────────────────
print("=" * 50)
print("  FACE RECOGNITION ENGINE — JUPYTER MODE")
print("=" * 50)
print(f"[INFO] DB_PATH = {DB_PATH}")


def _load_database(path: str) -> dict:
    """Load the face database from disk, or return an empty one."""
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {"persons": []}


_db = _load_database(DB_PATH)
known_names = [p["name"] for p in _db["persons"]]
known_encodings = [np.array(p["embedding"]) for p in _db["persons"]]

print(f"[INFO] Loaded {len(known_names)} enrolled face(s).")
if not known_names:
    print("[WARNING] Database is empty. All faces will be labelled 'Unknown'.")
print("[INFO] Ready to process frames.\n")


# ── Helper: encode a BGR numpy frame → base64 JPEG ────────────────────────
def _encode_frame(frame: np.ndarray) -> str:
    """Convert a BGR numpy frame to a base64-encoded JPEG string."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_out = Image.fromarray(rgb)
    buf = io.BytesIO()
    pil_out.save(buf, format="JPEG", quality=80)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ── Core: process a single frame ──────────────────────────────────────────
def process_frame(b64_image: str) -> str:
    """
    Receive a base64 JPEG frame, run face recognition,
    and return the annotated frame as a base64 JPEG string.

    If no face is detected the original frame is returned unmodified.
    """
    # ── Decode base64 → numpy BGR array ──
    img_bytes = base64.b64decode(b64_image)
    pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    # ── Resize to 25 % for faster detection ──
    small = cv2.resize(frame, (0, 0), fx=RESIZE_SCALE, fy=RESIZE_SCALE)
    rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

    face_locs = face_recognition.face_locations(rgb_small)
    face_encs = face_recognition.face_encodings(rgb_small, face_locs)

    # No faces → return original frame unmodified
    if not face_locs:
        return _encode_frame(frame)

    # ── Match each face & draw annotations ──
    scale = int(1 / RESIZE_SCALE)

    for (top, right, bottom, left), enc in zip(face_locs, face_encs):
        name = "Unknown"
        color = (0, 0, 255)  # Red for unknown

        if known_encodings:
            matches = face_recognition.compare_faces(
                known_encodings, enc, tolerance=TOLERANCE
            )
            distances = face_recognition.face_distance(known_encodings, enc)
            best_idx = int(np.argmin(distances))
            if matches[best_idx]:
                name = known_names[best_idx]
                color = (0, 255, 0)  # Green for recognized

        # Scale bounding box back to full resolution
        top *= scale
        right *= scale
        bottom *= scale
        left *= scale

        # Bounding box
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

        # Label background + text
        label_y = top - 10 if top - 10 > 10 else top + 20
        cv2.rectangle(
            frame, (left, label_y - 18), (right, label_y + 4), color, cv2.FILLED
        )
        cv2.putText(
            frame, name, (left + 4, label_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
        )

    return _encode_frame(frame)
