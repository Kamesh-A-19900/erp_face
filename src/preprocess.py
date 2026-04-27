import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import base64

# Load Haar Cascade once
FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

def detect_and_crop_face(image_input, target_size=(224, 224)):
    """Detect and crop the largest face from an image."""
    if isinstance(image_input, str):
        img = cv2.imread(image_input)
        if img is None:
            return None
    elif isinstance(image_input, np.ndarray):
        img = image_input.copy()
        if img.shape[-1] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    elif isinstance(image_input, Image.Image):
        img = cv2.cvtColor(np.array(image_input), cv2.COLOR_RGB2BGR)
    else:
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
    )

    if len(faces) == 0:
        return None

    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    face = img[y:y+h, x:x+w]
    face = cv2.resize(face, target_size)
    face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    return face_rgb


def preprocess_for_model(face_array):
    arr = face_array.astype(np.float32)
    arr = np.expand_dims(arr, axis=0)
    return tf.keras.applications.resnet50.preprocess_input(arr)


def decode_base64_image(data_url):
    try:
        header, encoded = data_url.split(',', 1)
        img_bytes = base64.b64decode(encoded)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    except Exception:
        return None