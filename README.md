# 🧑‍💻 Face Recognition System — DEBI Hackathon

A real-time face recognition application that enrolls individuals via webcam and identifies them against a stored database of face embeddings.

## 📦 Tech Stack

| Component        | Technology                        |
| ---------------- | --------------------------------- |
| Face Detection   | `face_recognition` (dlib-based)   |
| Image Capture    | `opencv-python`                   |
| Data Processing  | `numpy`                           |
| Storage          | JSON file (`database.json`)       |

## 🚀 Getting Started

### 1. Prerequisites

- **Python 3.8+**
- **CMake** — required to build `dlib` (the backend for `face_recognition`)
  - Windows: `choco install cmake` or download from [cmake.org](https://cmake.org/download/)
  - Linux: `sudo apt install cmake`
- A working **webcam** (only needed for `enroll.py`)

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Enroll Faces

#### Option A — Batch enroll from static photos (recommended)

1. Place one clear photo per person in `known_faces/`
2. Name each file after the person (e.g. `ahmed.jpg`, `sara.png`)
3. Run the seeding script:

```bash
python seed_db.py
```

- Supported formats: `.jpg`, `.jpeg`, `.png`
- The filename (without extension) becomes the person's name
- Photos with no detectable face are skipped with a warning
- Then run the app normally

#### Option B — Live webcam enrollment

```bash
python enroll.py
```

- The webcam feed will open — press **SPACE** to capture a photo.
- If a face is detected, you'll be prompted to enter the person's name.
- If no face is detected, you can retry.
- Enrolled data is saved to `database.json`.

## 📁 Project Structure

```
├── README.md            # Project documentation
├── requirements.txt     # Python dependencies
├── database.json        # Face embeddings database
├── enroll.py            # Live webcam enrollment
├── seed_db.py           # Batch enrollment from static photos
├── known_faces/         # Drop photos here for batch enrollment
│   ├── ahmed.jpg
│   └── sara.png
└── project.ipynb        # Experimentation notebook
```

## 🧠 How It Works

1. **Capture** — The webcam captures a photo of the person.
2. **Detect** — `face_recognition` locates faces in the image.
3. **Encode** — A 128-dimensional embedding is extracted from the detected face.
4. **Store** — The name and embedding are saved to `database.json`.

## 👥 Team

DEBI Hackathon Team

## 📄 License

This project is for educational / hackathon purposes.
