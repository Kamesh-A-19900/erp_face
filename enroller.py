"""
Headless face enrollment script for Raspberry Pi.
Captures face images from USB webcam and saves them for feature extraction.

Usage:
    python enroller.py --roll 21CS001
"""

import argparse
import configparser
import logging
import os
import re
import sys

import cv2
import dlib

logger = logging.getLogger(__name__)

_SHAPE_PREDICTOR = "data/data_dlib/shape_predictor_68_face_landmarks.dat"
_FACES_DIR = "data/data_faces_from_camera"

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def load_enroller_config(path: str = "config.ini") -> dict:
    """Read camera.device_index and enrollment.num_images from config.ini."""
    config = {"camera_index": 0, "num_images": 10}
    if not os.path.exists(path):
        logger.warning("config.ini not found at '%s'; using defaults.", path)
        return config

    parser = configparser.ConfigParser()
    parser.read(path)
    config["camera_index"] = parser.getint("camera", "device_index", fallback=0)
    config["num_images"] = parser.getint("enrollment", "num_images", fallback=10)
    return config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def get_next_person_index(base_dir: str) -> int:
    """Scan base_dir for person_<N>_* dirs and return max(N)+1 (or 1 if empty)."""
    if not os.path.exists(base_dir):
        return 1

    max_idx = 0
    pattern = re.compile(r"^person_(\d+)_")
    for entry in os.listdir(base_dir):
        m = pattern.match(entry)
        if m:
            idx = int(m.group(1))
            if idx > max_idx:
                max_idx = idx
    return max_idx + 1


# ---------------------------------------------------------------------------
# Enroller class
# ---------------------------------------------------------------------------


class Enroller:
    def __init__(self, roll_number: str, config: dict) -> None:
        self.roll_number = roll_number
        self.config = config

    def run(self) -> None:
        person_idx = get_next_person_index(_FACES_DIR)
        save_dir = os.path.join(_FACES_DIR, f"person_{person_idx}_{self.roll_number}")
        os.makedirs(save_dir, exist_ok=True)
        logger.info("Saving images to: %s", save_dir)

        detector = dlib.get_frontal_face_detector()

        cap = cv2.VideoCapture(self.config["camera_index"])
        if not cap.isOpened():
            logger.error("Cannot open webcam at index %d", self.config["camera_index"])
            sys.exit(1)

        num_images = self.config["num_images"]
        saved = 0
        img_idx = 0

        logger.info("Capturing %d face images for roll number: %s", num_images, self.roll_number)

        try:
            while saved < num_images:
                ret, frame = cap.read()
                if not ret:
                    logger.error("Webcam read failed.")
                    break

                faces = detector(frame, 0)

                if len(faces) == 0:
                    continue
                if len(faces) > 1:
                    logger.warning("Multiple faces detected (%d); skipping frame.", len(faces))
                    continue

                # Exactly one face — crop and save
                face = faces[0]
                top = max(0, face.top())
                bottom = min(frame.shape[0], face.bottom())
                left = max(0, face.left())
                right = min(frame.shape[1], face.right())
                cropped = frame[top:bottom, left:right]

                img_path = os.path.join(save_dir, f"img_face_{img_idx:04d}.jpg")
                cv2.imwrite(img_path, cropped)
                saved += 1
                img_idx += 1
                logger.info("Saved image %d/%d: %s", saved, num_images, img_path)

        finally:
            cap.release()

        logger.info(
            "Enrollment complete: %d images saved for %s", saved, self.roll_number
        )
        sys.exit(0)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    parser = argparse.ArgumentParser(description="Headless face enrollment")
    parser.add_argument("--roll", required=True, help="Roll number to enroll (e.g. 21CS001)")
    args = parser.parse_args()

    config = load_enroller_config()
    enroller = Enroller(roll_number=args.roll, config=config)
    enroller.run()


if __name__ == "__main__":
    main()
