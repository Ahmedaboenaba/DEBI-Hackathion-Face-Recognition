"""
enroll.py — Face Enrollment Script
Captures a photo from the webcam, extracts the face embedding,
and saves the person's name + embedding into database.json.
"""

import json
import os
import cv2
import face_recognition
import numpy as np

DB_PATH = os.environ.get("DB_PATH", "database.json") # you have to check if change the path based on you status


def load_database(path: str) -> dict:
    """Load the existing face database from disk, or create a fresh one."""
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {"persons": []}


def save_database(path: str, db: dict) -> None:
    """Persist the face database to disk."""
    with open(path, "w") as f:
        json.dump(db, f, indent=4)


def capture_photo() -> np.ndarray:
    """Open the webcam and capture a single frame when the user presses SPACE.

    Returns
    -------
    np.ndarray
        The captured BGR frame.
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Check your camera connection.")

    print("[INFO] Webcam opened. Press SPACE to capture, or Q to quit.")

    frame = None
    while True:
        ret, current_frame = cap.read()
        if not ret:
            print("[ERROR] Failed to read from webcam.")
            break

        cv2.imshow("Enrollment - Press SPACE to capture", current_frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(" "):
            frame = current_frame
            print("[INFO] Photo captured!")
            break
        elif key == ord("q"):
            print("[INFO] Enrollment cancelled by user.")
            break

    cap.release()
    cv2.destroyAllWindows()
    return frame


def extract_embedding(frame: np.ndarray):
    """Detect a face in the frame and return its 128-d embedding.

    Parameters
    ----------
    frame : np.ndarray
        A BGR image (as returned by OpenCV).

    Returns
    -------
    list[float] | None
        The 128-dimensional face embedding, or None if no face was found.
    """
    # face_recognition expects RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect face locations
    face_locations = face_recognition.face_locations(rgb_frame)

    if len(face_locations) == 0:
        return None

    if len(face_locations) > 1:
        print(f"[WARNING] {len(face_locations)} faces detected. Using the first one.")

    # Compute the 128-d face encoding
    encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    return encodings[0].tolist()


def enroll():
    """Main enrollment loop."""
    print("=" * 50)
    print("  FACE ENROLLMENT SYSTEM")
    print("=" * 50)
    print(f"[INFO] DB_PATH = {DB_PATH}")

    while True:
        frame = capture_photo()

        if frame is None:
            print("[INFO] No photo was captured. Exiting.")
            return

        embedding = extract_embedding(frame)

        if embedding is None:
            print("[WARNING] No face detected in the photo. Please try again.\n")
            retry = input("Try again? (y/n): ").strip().lower()
            if retry != "y":
                print("[INFO] Enrollment cancelled.")
                return
            continue

        # Face detected — ask for the person's name
        name = input("Enter the person's name: ").strip()
        if not name:
            print("[WARNING] Name cannot be empty. Enrollment cancelled.")
            return

        # Save to database
        db = load_database(DB_PATH)
        db["persons"].append({
            "name": name,
            "embedding": embedding
        })
        save_database(DB_PATH, db)

        print(f"[SUCCESS] '{name}' enrolled successfully! "
              f"(Total enrolled: {len(db['persons'])})\n")

        another = input("Enroll another person? (y/n): ").strip().lower()
        if another != "y":
            break

    print("[INFO] Enrollment complete. Goodbye!")


if __name__ == "__main__":
    enroll()
