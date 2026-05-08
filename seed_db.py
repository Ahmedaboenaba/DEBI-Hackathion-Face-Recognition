"""
seed_db.py — Batch Face Enrollment from Static Photos
Scans the known_faces/ folder, extracts face embeddings from each image,
and saves all {name, embedding} entries into database.json.

Usage:
    1. Place one clear photo per person in known_faces/
       (e.g. ahmed.jpg, sara.png)
    2. Run:  python seed_db.py
"""

import json
import os
import face_recognition

DB_PATH = os.environ.get("DB_PATH", "/kaggle/working/database.json")
KNOWN_FACES_DIR = "/kaggle/input/datasets/ahmedaboenaba/known-faces/known_faces"
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}


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


def seed():
    """Scan known_faces/ and enroll every detected face into the database."""
    print("=" * 50)
    print("  DATABASE SEEDING FROM STATIC PHOTOS")
    print("=" * 50)
    print(f"[INFO] DB_PATH = {DB_PATH}")

    if not os.path.isdir(KNOWN_FACES_DIR):
        print(f"[ERROR] Directory '{KNOWN_FACES_DIR}/' not found. "
              "Please create it and add photos.")
        return

    # Collect image files
    image_files = [
        f for f in sorted(os.listdir(KNOWN_FACES_DIR))
        if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS
    ]

    if not image_files:
        print(f"[WARNING] No image files found in '{KNOWN_FACES_DIR}/'. "
              "Add .jpg, .jpeg, or .png files and try again.")
        return

    print(f"[INFO] Found {len(image_files)} image(s) in '{KNOWN_FACES_DIR}/'.\n")

    db = load_database(DB_PATH)
    enrolled = 0
    skipped = 0

    for filename in image_files:
        name = os.path.splitext(filename)[0]
        filepath = os.path.join(KNOWN_FACES_DIR, filename)

        print(f"  Processing: {filename} → \"{name}\"")

        # Load image and extract face encodings
        image = face_recognition.load_image_file(filepath)
        encodings = face_recognition.face_encodings(image)

        if len(encodings) == 0:
            print(f"  [WARNING] No face detected in '{filename}'. Skipping.\n")
            skipped += 1
            continue

        if len(encodings) > 1:
            print(f"  [WARNING] Multiple faces in '{filename}'. "
                  "Using the first detected face.")

        embedding = encodings[0].tolist()

        db["persons"].append({
            "name": name,
            "embedding": embedding
        })

        enrolled += 1
        print(f"  [SUCCESS] '{name}' enrolled successfully!\n")

    save_database(DB_PATH, db)

    print("-" * 50)
    print(f"  Done!  Enrolled: {enrolled}  |  Skipped: {skipped}")
    print(f"  Total persons in database: {len(db['persons'])}")
    print("-" * 50)


if __name__ == "__main__":
    seed()
