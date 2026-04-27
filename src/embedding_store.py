"""
embedding_store.py
DeepFace ArcFace embeddings + FAISS IndexFlatIP for cosine search.
No training. Registration takes ~1 second.
"""

import os
import json
import numpy as np
import faiss
from deepface import DeepFace

MODEL_DIR    = os.path.join(os.path.dirname(__file__), '..', 'models')
FAISS_PATH   = os.path.join(MODEL_DIR, 'face_index.faiss')
META_PATH    = os.path.join(MODEL_DIR, 'face_meta.json')

os.makedirs(MODEL_DIR, exist_ok=True)

EMBED_DIM  = 512           # ArcFace output dimension
MODEL_NAME = 'ArcFace'     # alternatives: 'Facenet512', 'VGG-Face'
DETECTOR   = 'opencv'      # fast; swap 'retinaface' for better detection accuracy


# ── Embedding extraction ───────────────────────────────────────────────────────

def get_embedding(face_rgb_array: np.ndarray) -> np.ndarray:
    """
    face_rgb_array: (H, W, 3) uint8 RGB numpy array (already cropped by preprocess.py).
    Returns unit-norm (512,) float32 vector.
    DeepFace.represent accepts numpy arrays directly.
    """
    result = DeepFace.represent(
        img_path=face_rgb_array,
        model_name=MODEL_NAME,
        enforce_detection=False,   # face already cropped upstream
        detector_backend='skip',   # skip re-detection, already cropped
    )
    vec = np.array(result[0]['embedding'], dtype=np.float32)
    vec /= (np.linalg.norm(vec) + 1e-8)
    return vec


# ── FAISS store ────────────────────────────────────────────────────────────────

class FaceStore:
    """
    Thin wrapper around a FAISS IndexFlatIP (inner product = cosine on unit vectors).
    roll_numbers[i] maps FAISS index position i → roll number string.
    """

    def __init__(self):
        self.index        = faiss.IndexFlatIP(EMBED_DIM)
        self.roll_numbers: list[str] = []
        self._load()

    # ── persistence ──────────────────────────────────────────────────────────

    def _load(self):
        if os.path.exists(FAISS_PATH) and os.path.exists(META_PATH):
            self.index        = faiss.read_index(FAISS_PATH)
            with open(META_PATH) as f:
                self.roll_numbers = json.load(f)
            print(f"[store] Loaded {self.index.ntotal} embeddings.")
        else:
            print("[store] Fresh FAISS index created.")

    def _save(self):
        faiss.write_index(self.index, FAISS_PATH)
        with open(META_PATH, 'w') as f:
            json.dump(self.roll_numbers, f)

    # ── CRUD ─────────────────────────────────────────────────────────────────

    def add(self, roll_number: str, embedding: np.ndarray):
        """Add or overwrite a student's mean embedding."""
        if roll_number in self.roll_numbers:
            self._remove(roll_number)

        vec = embedding.reshape(1, -1).astype(np.float32)
        self.index.add(vec)
        self.roll_numbers.append(roll_number)
        self._save()
        print(f"[store] {roll_number} added ({self.index.ntotal} total).")

    def _remove(self, roll_number: str):
        """Rebuild the index without the given roll number."""
        keep = [i for i, r in enumerate(self.roll_numbers) if r != roll_number]
        if not keep:
            self.index        = faiss.IndexFlatIP(EMBED_DIM)
            self.roll_numbers = []
            return

        vecs = np.zeros((self.index.ntotal, EMBED_DIM), dtype=np.float32)
        for i in range(self.index.ntotal):
            self.index.reconstruct(i, vecs[i])

        kept_vecs          = vecs[keep]
        self.roll_numbers  = [self.roll_numbers[i] for i in keep]
        self.index         = faiss.IndexFlatIP(EMBED_DIM)
        self.index.add(kept_vecs)

    # ── search ───────────────────────────────────────────────────────────────

    def search(self, embedding: np.ndarray, threshold: float = 0.40):
        """
        Returns (roll_number, score) if best score >= threshold, else (None, score).
        Score is cosine similarity in [-1, 1].
        """
        if self.index.ntotal == 0:
            return None, 0.0

        vec              = embedding.reshape(1, -1).astype(np.float32)
        scores, indices  = self.index.search(vec, k=1)
        score            = float(scores[0][0])
        idx              = int(indices[0][0])

        if score >= threshold and 0 <= idx < len(self.roll_numbers):
            return self.roll_numbers[idx], score
        return None, score


# ── Module-level singleton ────────────────────────────────────────────────────

_store: FaceStore | None = None

def get_store() -> FaceStore:
    global _store
    if _store is None:
        _store = FaceStore()
    return _store