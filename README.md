# 👤 Face Recognition System — DEBI Hackathon

[![CI / Deploy](https://github.com/Ahmedaboenaba/DEBI-Hackathion-Face-Recognition/actions/workflows/deploy.yml/badge.svg)](https://github.com/Ahmedaboenaba/DEBI-Hackathion-Face-Recognition/actions)

A real-time face recognition application that identifies enrolled individuals through a webcam feed. Built for the DEBI Hackathon — detects faces, matches them against a stored database of 128-dimensional embeddings, and displays annotated bounding boxes with name and confidence percentage.

## 🌐 Live Demo

> **Streamlit Cloud:** [https://debi-face-recognition.streamlit.app](https://debi-face-recognition.streamlit.app)
>
> *(Update this URL once Streamlit Cloud assigns the final link)*

## 📦 Tech Stack

| Component         | Technology                                     |
| ----------------- | ---------------------------------------------- |
| Face Detection    | `face_recognition` (dlib HOG / CNN)            |
| Image Processing  | `opencv-python`                                |
| Data Processing   | `numpy`                                        |
| Web App           | `streamlit` + `streamlit-webrtc`               |
| Containerization  | `Docker` (Python 3.10-slim)                    |
| CI/CD             | GitHub Actions → Streamlit Cloud               |
| Storage           | JSON file (`database.json`)                    |
| Testing           | `pytest`                                       |

## 🚀 Getting Started

### 1. Prerequisites

- **Python 3.8+**
- **CMake** — required to build `dlib`
  - Windows: `choco install cmake` or download from [cmake.org](https://cmake.org/download/)
  - Linux: `sudo apt install cmake libopenblas-dev liblapack-dev`
- A working **webcam**

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Enroll Faces

#### Option A — Batch enroll from static photos (recommended)

1. Place one clear photo per person in `known_faces/`
2. Name each file after the person (e.g. `Ahmad Imad.jpg`, `Ahmed Jaber.jpg`)
3. Run the seeding script:

```bash
python seed_db.py
```

- Supported formats: `.jpg`, `.jpeg`, `.png`
- The filename (without extension) becomes the person's name
- Photos with no detectable face are skipped with a warning

#### Option B — Live webcam enrollment

```bash
python enroll.py
```

- The webcam feed will open — press **SPACE** to capture a photo.
- If a face is detected, you'll be prompted to enter the person's name.
- Enrolled data is saved to `database.json`.

### 4. Run Locally

**Streamlit app (browser-based):**

```bash
streamlit run app.py
```

**OpenCV desktop version:**

```bash
python main.py
```

Press **Q** to quit the webcam feed.

### 5. Run with Docker

```bash
docker build -t face-recognition .
docker run -p 8501:8501 face-recognition
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

## 📁 Project Structure

```
├── app.py                          # Streamlit web app (streamlit-webrtc)
├── main.py                         # Local webcam recognition engine
├── enroll.py                       # Live webcam enrollment
├── seed_db.py                      # Batch enrollment from photos
├── database.json                   # Face embeddings database
├── known_faces/                    # Drop photos here for enrollment
├── requirements.txt                # Python dependencies
├── packages.txt                    # System deps for Streamlit Cloud
├── Dockerfile                      # Docker container config
├── .streamlit/config.toml          # Streamlit server config
├── .github/workflows/deploy.yml    # CI/CD pipeline
├── tests/test_db.py                # Database validation tests
└── README.md                       # This file
```

## 🧠 How It Works

1. **Enroll** — Capture or upload a face photo; extract a 128-d embedding and save to `database.json`
2. **Detect** — On each webcam frame, resize to 25% and detect face locations
3. **Encode** — Extract 128-d embeddings for each detected face
4. **Match** — Compare against the database using Euclidean distance (tolerance = 0.5)
5. **Annotate** — Draw green box + `"Name — 91%"` for matches, red box + `"Unknown"` otherwise

## 🧪 Testing

```bash
python -m pytest tests/ -v
```

## 👥 Team

DEBI Hackathon Team

## 📄 License

This project is for educational / hackathon purposes.
