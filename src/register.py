"""
register.py
Registers a new face:
  1. Augment source image → faces/<roll>/  (already done by imageaugmentation.py)
  2. Extract ArcFace embedding for each augmented image
  3. Average embeddings → unit-norm mean vector
  4. Store in FAISS via FaceStore.add()

No GPU, no epochs, no retraining of any weights.
~1 second per registration.
"""

import os
import shutil
import numpy as np

from preprocess import detect_and_crop_face
from embedding_store import get_embedding, get_store
from imageaugmentation import dataGen

FACES_DIR = os.path.join(os.path.dirname(__file__), '..', 'faces')


def _mean_embedding(student_dir: str) -> np.ndarray | None:
    """Compute mean L2-normalised ArcFace embedding over all images in a directory."""
    vecs = []
    for fname in sorted(os.listdir(student_dir)):
        fpath = os.path.join(student_dir, fname)
        face  = detect_and_crop_face(fpath)       # → (224,224,3) RGB or None
        if face is not None:
            vec = get_embedding(face)             # unit-norm (512,)
            vecs.append(vec)

    if not vecs:
        return None

    mean = np.mean(vecs, axis=0)
    return mean / (np.linalg.norm(mean) + 1e-8)


def add_new_face(roll_number: str, image_path: str):
    """
    Public API — called from app.py in a background thread.

    Steps:
      1. Augment source image into faces/<roll>/
      2. Compute mean embedding over all augmented images
      3. Store in FAISS
    """
    student_dir = os.path.join(FACES_DIR, roll_number)
    os.makedirs(student_dir, exist_ok=True)

    # 1. Augment
    dataGen(image_path, save_dir=student_dir, n_images=15)
    shutil.copy(image_path, os.path.join(student_dir, 'original.jpg'))
    print(f"[register] Augmented images saved for {roll_number}.")

    # 2. Mean embedding
    emb = _mean_embedding(student_dir)
    if emb is None:
        raise ValueError(f"No detectable faces in augmented images for {roll_number}.")

    # 3. Store
    get_store().add(roll_number, emb)
    print(f"[register] {roll_number} registered successfully.")