<<<<<<< HEAD
# Face Recognition Based Attendance System

This project is a face recognition-based attendance system that uses OpenCV and Python. The system uses a camera to capture images of individuals and then compares them with the images in the database to mark attendance.

## Installation

1. Clone the repository to your local machine. ``` git clone https://github.com/Arijit1080/Face-Recognition-Based-Attendance-System ```
2. Install the required packages using ```pip install -r requirements.txt```.
3. Download the dlib models from https://drive.google.com/drive/folders/12It2jeNQOxwStBxtagL1vvIJokoz-DL4?usp=sharing and place the data folder inside the repo

## Usage

1. Collect the Faces Dataset by running ``` python get_faces_from_camera_tkinter.py``` .
2. Convert the dataset into ```python features_extraction_to_csv.py```.
3. To take the attendance run ```python attendance_taker.py``` .
4. Check the Database by ```python app.py```.


## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue if you find any bugs or have any suggestions.


=======
# ERP Face Recognition System

IoT-based web application using DeepFace ArcFace embeddings + FAISS for instant face registration and recognition.

## Architecture

- **No training required** — uses pretrained ArcFace model from DeepFace
- **Instant registration** — ~1 second per new face (augmentation + embedding extraction)
- **FAISS IndexFlatIP** — cosine similarity search on unit-norm embeddings
- **Flask web app** — webcam capture, live recognition, ERP profile pages

## File Structure

```
src/
  app.py                 - Flask routes (webcam, registration, ERP)
  embedding_store.py     - FAISS index + DeepFace ArcFace embeddings
  register.py            - Face registration (augment + embed + store)
  preprocess.py          - Face detection (Haar cascade) + cropping
  imageaugmentation.py   - Keras ImageDataGenerator (15 augmented images)
  test.py                - Quick augmentation test script

templates/
  index.html             - Webcam scanner + live recognition
  add_face.html          - Registration form (upload or webcam)
  erp.html               - Student profile page

models/
  face_index.faiss       - FAISS index (512-d embeddings)
  face_meta.json         - Roll number → index mapping

faces/
  <ROLL_NUMBER>/         - Augmented images per student
    original.jpg
    aug_1.jpg ... aug_15.jpg
```

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Generate SSL cert for webcam access over LAN
openssl req -x509 -newkey rsa:2048 -keyout key.pem -out cert.pem -days 365 -nodes -subj '/CN=localhost'

# Run
cd src && python app.py
```

Access at `https://localhost:5000` or `https://<your-lan-ip>:5000`

## How It Works

### Registration Flow
1. User uploads face photo (or captures via webcam)
2. Face detection (OpenCV Haar cascade) validates the image
3. Image augmentation generates 15 variations (rotation, shift, brightness, zoom)
4. DeepFace ArcFace extracts 512-d embedding for each augmented image
5. Mean embedding computed and L2-normalized
6. Stored in FAISS index with roll number mapping

### Recognition Flow
1. Webcam captures frame every 1.5s
2. Face detection crops the face region
3. DeepFace ArcFace extracts embedding
4. FAISS searches for nearest neighbor (cosine similarity)
5. If score ≥ 0.40 threshold, student identified
6. ERP profile displayed with name, branch, year, attendance

## Key Features

- **Instant registration** — no GPU, no epochs, no retraining
- **High accuracy** — ArcFace embeddings trained on millions of faces
- **Scalable** — FAISS handles thousands of faces efficiently
- **Robust** — mean embedding over 15 augmented images reduces variance
- **Real-time** — <100ms recognition latency on CPU

## Configuration

Edit `src/embedding_store.py`:
- `MODEL_NAME` — 'ArcFace' (default), 'Facenet512', 'VGG-Face'
- `DETECTOR` — 'opencv' (fast), 'retinaface' (accurate)
- `threshold` in `FaceStore.search()` — 0.40 (default), lower = more permissive

## Dependencies

- **deepface** — pretrained face recognition models
- **faiss-cpu** — fast similarity search
- **flask** — web framework
- **opencv-python** — face detection + image processing
- **tf-keras** — backend for DeepFace
>>>>>>> 6dcab3f13385d181d17318e60890c2961500da32
