"""
main.py — Real-Time Face Recognition Engine (Local Webcam Edition)

Loads enrolled faces from database.json, opens the webcam,
and performs live face recognition with annotated bounding boxes.

Press Q to quit.
"""

import json
import os
import sys

import cv2
import numpy as np
import face_recognition

# ── Configuration ──────────────────────────────────────────────────────────
DB_PATH = os.environ.get(
    "DB_PATH",
    r"D:\Self_Study\DEBI\DEBI-Hackathion-Face-Recognition\database.json",
)
TOLERANCE = 0.5
RESIZE_SCALE = 0.25  # Process frames at 25% resolution for speed


# ── Load enrolled faces at startup ─────────────────────────────────────────
def _load_database(path: str) -> dict:
    """Load the face database from disk, or return an empty one."""
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {"persons": []}


def recognize_faces(frame, known_names, known_encodings):
    """
    Run face detection + recognition on a single BGR frame.

    Returns the annotated frame with bounding boxes and labels drawn.
    Handles zero faces and multiple faces gracefully.
    """
    scale = int(1 / RESIZE_SCALE)  # = 4

    # ── Resize to 25%, detect faces, extract embeddings ────────────
    small = cv2.resize(frame, (0, 0), fx=RESIZE_SCALE, fy=RESIZE_SCALE)
    rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

    face_locs = face_recognition.face_locations(rgb_small)
    face_encs = face_recognition.face_encodings(rgb_small, face_locs)

    # No faces detected — skip annotation
    if not face_locs:
        return frame

    # ── Match each detected face against the database ──────────────
    for (top, right, bottom, left), enc in zip(face_locs, face_encs):
        name = "Unknown"
        confidence = 0.0
        color = (0, 0, 255)  # Red for unknown

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
                color = (0, 255, 0)  # Green for recognized

        # ── Scale bounding box back to full resolution ─────────────
        top *= scale
        right *= scale
        bottom *= scale
        left *= scale

        # ── Draw bounding box (green=known, red=unknown) ───────────
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

        # ── Build label with confidence percentage ─────────────────
        if name != "Unknown":
            label = f"{name} — {confidence:.0f}%"
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

    return frame


def main():
    print("=" * 50)
    print("  FACE RECOGNITION ENGINE — LOCAL WEBCAM")
    print("=" * 50)
    print(f"[INFO] DB_PATH = {DB_PATH}")

    # ── 1. Load all names and embeddings from database.json ────────────
    db = _load_database(DB_PATH)
    known_names = [p["name"] for p in db["persons"]]
    known_encodings = [np.array(p["embedding"]) for p in db["persons"]]

    print(f"[INFO] Loaded {len(known_names)} enrolled face(s).")
    if not known_names:
        print("[WARNING] Database is empty. All faces will be labelled 'Unknown'.")

    # ── 2. Open the webcam ─────────────────────────────────────────────
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam. Check your camera connection.")
        return

    print("[INFO] Webcam opened. Press Q to quit.\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Failed to read from webcam.")
                break

            # ── 3-6. Recognize faces and annotate frame ────────────
            frame = recognize_faces(frame, known_names, known_encodings)

            # ── 7. Show live feed and exit on Q ────────────────────
            cv2.imshow("Face Recognition", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user (Ctrl+C).")
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        sys.exit(1)
    finally:
        # ── Clean up (always runs) ─────────────────────────────────
        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Webcam released. Goodbye!")


if __name__ == "__main__":
    main()
