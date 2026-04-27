"""
Headless face recognition script for Raspberry Pi.
Reads frames from USB webcam, recognizes faces using dlib ResNet model,
and sends roll numbers over USB serial to Arduino Mega for TFT display.
"""

import configparser
import logging
import os
import sys
import time

import cv2
import dlib
import numpy as np
import pandas as pd
import serial

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_DEFAULTS = {
    "camera_index": 0,
    "serial_port": "/dev/ttyUSB0",
    "baud_rate": 9600,
    "confidence_threshold": 0.4,
}


def load_config(path: str = "config.ini") -> dict:
    """Read config.ini and return a dict of typed values.

    Falls back to hardcoded defaults and logs a WARNING if the file is absent.
    """
    config = dict(_DEFAULTS)
    if not os.path.exists(path):
        logger.warning("config.ini not found at '%s'; using hardcoded defaults.", path)
        return config

    parser = configparser.ConfigParser()
    parser.read(path)

    try:
        config["camera_index"] = parser.getint("camera", "device_index", fallback=_DEFAULTS["camera_index"])
    except (configparser.Error, ValueError):
        pass

    try:
        config["serial_port"] = parser.get("serial", "port", fallback=_DEFAULTS["serial_port"])
    except configparser.Error:
        pass

    try:
        config["baud_rate"] = parser.getint("serial", "baud_rate", fallback=_DEFAULTS["baud_rate"])
    except (configparser.Error, ValueError):
        pass

    try:
        config["confidence_threshold"] = parser.getfloat(
            "recognition", "confidence_threshold", fallback=_DEFAULTS["confidence_threshold"]
        )
    except (configparser.Error, ValueError):
        pass

    return config


# ---------------------------------------------------------------------------
# Face database
# ---------------------------------------------------------------------------


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Compute L2 distance between two 128-D vectors."""
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    return float(np.sqrt(np.sum(np.square(a - b))))


def load_face_database(csv_path: str) -> tuple:
    """Load face descriptors from CSV.

    Returns:
        (names, descriptors) where names is list[str] and
        descriptors is list[np.ndarray] of 128-D float arrays.

    Raises:
        FileNotFoundError if csv_path does not exist.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Face database not found: {csv_path}")

    csv_rd = pd.read_csv(csv_path, header=None)
    names = []
    descriptors = []

    for i in range(csv_rd.shape[0]):
        names.append(str(csv_rd.iloc[i][0]))
        features = []
        for j in range(1, 129):
            val = csv_rd.iloc[i][j]
            features.append(0.0 if val == "" else float(val))
        descriptors.append(np.array(features, dtype=float))

    logger.info("Loaded %d faces from database: %s", len(names), csv_path)
    return names, descriptors


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------


def find_best_match(
    descriptor: np.ndarray,
    names: list,
    descriptors: list,
    threshold: float,
) -> tuple:
    """Compare descriptor against all known descriptors.

    Returns:
        (roll_number, min_distance) if min_distance < threshold,
        ("unknown", min_distance) otherwise.
    """
    if not descriptors:
        return "unknown", float("inf")

    distances = [euclidean_distance(descriptor, d) for d in descriptors]
    min_dist = min(distances)
    best_idx = distances.index(min_dist)

    if min_dist < threshold:
        return names[best_idx], min_dist
    return "unknown", min_dist


# ---------------------------------------------------------------------------
# Cooldown tracker
# ---------------------------------------------------------------------------


class CooldownTracker:
    """Tracks per-roll-number 30-second cooldown to avoid duplicate sends."""

    COOLDOWN_SECONDS: int = 30  # hardcoded, not configurable

    def __init__(self) -> None:
        self._last_sent: dict = {}  # roll_number -> unix timestamp

    def should_send(self, roll_number: str, now: float = None) -> bool:
        """Return True if roll_number has not been sent within COOLDOWN_SECONDS."""
        if now is None:
            now = time.time()
        last = self._last_sent.get(roll_number)
        if last is None:
            return True
        return (now - last) >= self.COOLDOWN_SECONDS

    def record_sent(self, roll_number: str, now: float = None) -> None:
        """Record that roll_number was just sent."""
        if now is None:
            now = time.time()
        self._last_sent[roll_number] = now


# ---------------------------------------------------------------------------
# Serial sender
# ---------------------------------------------------------------------------


class SerialSender:
    """Wraps pyserial for sending person data to Arduino Mega.

    Sends CSV line: NAME,REG,ROLE\\n
    The name field in features_all.csv may encode all three as "Name|Reg|Role".
    If only a plain roll number is stored, it is used as REG and defaults apply.
    """

    def __init__(self, port: str, baud_rate: int) -> None:
        """Open serial port. Raises serial.SerialException on failure."""
        self._serial = serial.Serial(port, baud_rate, timeout=1)
        logger.debug("Serial port opened: %s @ %d baud", port, baud_rate)

    def send(self, name_field: str) -> None:
        """Build NAME,REG,ROLE string from name_field and write over serial.

        name_field formats supported:
          "Name|Reg|Role"  -> sent as-is split by |
          "21CS001"        -> sent as "Student,21CS001,Student"
        """
        if "|" in name_field:
            parts = name_field.split("|", 2)
            display_name = parts[0].strip()
            reg = parts[1].strip() if len(parts) > 1 else name_field
            role = parts[2].strip() if len(parts) > 2 else "Student"
        else:
            display_name = "Student"
            reg = name_field.strip()
            role = "Student"

        payload = f"{display_name},{reg},{role}\n"
        self._serial.write(payload.encode("utf-8"))
        logger.debug("Serial TX: %r", payload.strip())

    def close(self) -> None:
        """Close the serial port."""
        if self._serial and self._serial.is_open:
            self._serial.close()
            logger.debug("Serial port closed.")


# ---------------------------------------------------------------------------
# Recognizer — main orchestration class
# ---------------------------------------------------------------------------

_SHAPE_PREDICTOR = "data/data_dlib/shape_predictor_68_face_landmarks.dat"
_FACE_RECO_MODEL = "data/data_dlib/dlib_face_recognition_resnet_model_v1.dat"
_FEATURES_CSV = "data/features_all.csv"


class Recognizer:
    """Headless face recognition loop."""

    def __init__(self, config: dict) -> None:
        self.config = config

    def run(self) -> None:
        """Main recognition loop. Exits with sys.exit(1) on startup failure."""
        # Load face database
        try:
            names, descriptors = load_face_database(_FEATURES_CSV)
        except FileNotFoundError as exc:
            logger.error("Cannot load face database: %s", exc)
            sys.exit(1)

        # Open serial port
        try:
            sender = SerialSender(
                self.config["serial_port"], self.config["baud_rate"]
            )
        except serial.SerialException as exc:
            logger.error("Cannot open serial port '%s': %s", self.config["serial_port"], exc)
            sys.exit(1)

        # Open webcam
        cap = cv2.VideoCapture(self.config["camera_index"])
        if not cap.isOpened():
            logger.error("Cannot open webcam at index %d", self.config["camera_index"])
            sender.close()
            sys.exit(1)

        # Load dlib models
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(_SHAPE_PREDICTOR)
        face_reco_model = dlib.face_recognition_model_v1(_FACE_RECO_MODEL)

        cooldown = CooldownTracker()
        threshold = self.config["confidence_threshold"]

        logger.info("Recognizer started. Press Ctrl+C to stop.")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.error("Webcam read failed; stopping.")
                    break

                faces = detector(frame, 0)

                for face in faces:
                    shape = predictor(frame, face)
                    descriptor = np.array(
                        face_reco_model.compute_face_descriptor(frame, shape),
                        dtype=float,
                    )
                    roll, dist = find_best_match(descriptor, names, descriptors, threshold)

                    if roll == "unknown":
                        logger.debug("Unknown face detected (dist=%.4f)", dist)
                        continue

                    # Log recognition event
                    logger.info(
                        "Recognized: %s | distance=%.4f | time=%s",
                        roll,
                        dist,
                        time.strftime("%Y-%m-%d %H:%M:%S"),
                    )

                    if cooldown.should_send(roll):
                        try:
                            sender.send(roll)
                            cooldown.record_sent(roll)
                            logger.debug("Sent to Arduino: %s", roll)
                        except serial.SerialException as exc:
                            logger.error("Serial write failed: %s", exc)
                            raise

        except KeyboardInterrupt:
            logger.info("Interrupted by user.")
        finally:
            cap.release()
            sender.close()
            logger.info("Recognizer stopped.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    config = load_config()
    recognizer = Recognizer(config)
    recognizer.run()


if __name__ == "__main__":
    main()
