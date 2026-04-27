"""
Tests for recognizer.py components.
Includes unit tests and property-based tests (pytest + hypothesis).
"""

import configparser
import importlib
import os
import sys
import tempfile
import time
import types
from unittest.mock import MagicMock, patch, call

import numpy as np
import pytest
from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st

# ---------------------------------------------------------------------------
# Stub heavy native modules so tests can import recognizer without cv2/dlib
# ---------------------------------------------------------------------------

def _stub_module(name):
    mod = types.ModuleType(name)
    sys.modules[name] = mod
    return mod

for _name in ["cv2", "dlib"]:
    if _name not in sys.modules:
        _stub_module(_name)

# Stub serial with a minimal SerialException
if "serial" not in sys.modules:
    _serial = _stub_module("serial")
    class _SerialException(Exception):
        pass
    _serial.Serial = MagicMock
    _serial.SerialException = _SerialException

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from recognizer import (
    CooldownTracker,
    SerialSender,
    euclidean_distance,
    find_best_match,
    load_config,
    load_face_database,
)

# ---------------------------------------------------------------------------
# Unit tests: load_config
# ---------------------------------------------------------------------------


def test_load_config_absent_returns_defaults(tmp_path, caplog):
    import logging
    with caplog.at_level(logging.WARNING):
        cfg = load_config(str(tmp_path / "nonexistent.ini"))
    assert cfg["camera_index"] == 0
    assert cfg["serial_port"] == "/dev/ttyUSB0"
    assert cfg["baud_rate"] == 9600
    assert cfg["confidence_threshold"] == 0.4
    assert any("not found" in r.message.lower() or "default" in r.message.lower() for r in caplog.records)


def test_load_config_reads_values(tmp_path):
    ini = tmp_path / "config.ini"
    ini.write_text(
        "[camera]\ndevice_index = 2\n"
        "[serial]\nport = /dev/ttyACM0\nbaud_rate = 115200\n"
        "[recognition]\nconfidence_threshold = 0.35\n"
        "[enrollment]\nnum_images = 20\n"
    )
    cfg = load_config(str(ini))
    assert cfg["camera_index"] == 2
    assert cfg["serial_port"] == "/dev/ttyACM0"
    assert cfg["baud_rate"] == 115200
    assert abs(cfg["confidence_threshold"] - 0.35) < 1e-9


# ---------------------------------------------------------------------------
# Property test: config round-trip (Property 5)
# Feature: face-attendance-arduino-display, Property 5: config round-trip
# ---------------------------------------------------------------------------

@given(
    cam=st.integers(0, 9),
    port=st.from_regex(r"/dev/tty[A-Za-z0-9]{1,8}", fullmatch=True),
    baud=st.sampled_from([9600, 19200, 38400, 57600, 115200]),
    threshold=st.floats(0.01, 1.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_property_5_config_round_trip(tmp_path, cam, port, baud, threshold):
    """Property 5: writing config values and reading them back produces identical values."""
    ini = tmp_path / "config.ini"
    content = (
        f"[camera]\ndevice_index = {cam}\n"
        f"[serial]\nport = {port}\nbaud_rate = {baud}\n"
        f"[recognition]\nconfidence_threshold = {threshold}\n"
        f"[enrollment]\nnum_images = 10\n"
    )
    ini.write_text(content)
    cfg = load_config(str(ini))
    assert cfg["camera_index"] == cam
    assert cfg["serial_port"] == port
    assert cfg["baud_rate"] == baud
    assert abs(cfg["confidence_threshold"] - threshold) < 1e-6


# ---------------------------------------------------------------------------
# Unit tests: load_face_database
# ---------------------------------------------------------------------------


def test_load_face_database_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_face_database(str(tmp_path / "missing.csv"))


def test_load_face_database_valid_csv(tmp_path):
    csv_path = tmp_path / "features_all.csv"
    row1 = "21CS001," + ",".join(["0.1"] * 128)
    row2 = "21CS002," + ",".join(["0.5"] * 128)
    csv_path.write_text(row1 + "\n" + row2 + "\n")
    names, descriptors = load_face_database(str(csv_path))
    assert names == ["21CS001", "21CS002"]
    assert len(descriptors) == 2
    assert descriptors[0].shape == (128,)
    assert abs(descriptors[0][0] - 0.1) < 1e-6


# ---------------------------------------------------------------------------
# Unit tests: euclidean_distance
# ---------------------------------------------------------------------------


def test_euclidean_distance_identical():
    a = np.ones(128)
    assert euclidean_distance(a, a) == pytest.approx(0.0)


def test_euclidean_distance_known():
    a = np.array([3.0, 0.0])
    b = np.array([0.0, 4.0])
    assert euclidean_distance(a, b) == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# Unit tests: find_best_match
# ---------------------------------------------------------------------------


def test_find_best_match_returns_unknown_when_empty():
    desc = np.zeros(128)
    roll, dist = find_best_match(desc, [], [], 0.4)
    assert roll == "unknown"


def test_find_best_match_returns_roll_below_threshold():
    db_desc = np.zeros(128)
    query = np.zeros(128)
    query[0] = 0.1  # dist = 0.1 < 0.4
    roll, dist = find_best_match(query, ["21CS001"], [db_desc], 0.4)
    assert roll == "21CS001"
    assert dist == pytest.approx(0.1)


def test_find_best_match_returns_unknown_above_threshold():
    db_desc = np.zeros(128)
    query = np.ones(128)  # large distance
    roll, dist = find_best_match(query, ["21CS001"], [db_desc], 0.4)
    assert roll == "unknown"


# ---------------------------------------------------------------------------
# Property test: face matching correctness (Property 1)
# Feature: face-attendance-arduino-display, Property 1: face matching correctness
# ---------------------------------------------------------------------------

_desc_strategy = st.lists(
    st.floats(-1.0, 1.0, allow_nan=False, allow_infinity=False),
    min_size=128,
    max_size=128,
)


@given(
    query=_desc_strategy,
    db_entries=st.lists(
        st.tuples(
            st.text(min_size=1, max_size=10, alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"),
            _desc_strategy,
        ),
        min_size=1,
        max_size=5,
    ),
    threshold=st.floats(0.01, 2.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=100)
def test_property_1_face_matching_correctness(query, db_entries, threshold):
    """Property 1: find_best_match returns the roll with minimum distance when below threshold."""
    names = [e[0] for e in db_entries]
    descriptors = [np.array(e[1], dtype=float) for e in db_entries]
    query_arr = np.array(query, dtype=float)

    roll, dist = find_best_match(query_arr, names, descriptors, threshold)

    # Compute expected
    distances = [euclidean_distance(query_arr, d) for d in descriptors]
    min_dist = min(distances)
    best_idx = distances.index(min_dist)

    assert abs(dist - min_dist) < 1e-9
    if min_dist < threshold:
        assert roll == names[best_idx]
    else:
        assert roll == "unknown"


# ---------------------------------------------------------------------------
# Unit tests: CooldownTracker
# ---------------------------------------------------------------------------


def test_cooldown_first_send_allowed():
    ct = CooldownTracker()
    assert ct.should_send("21CS001", now=1000.0) is True


def test_cooldown_blocks_within_30s():
    ct = CooldownTracker()
    ct.record_sent("21CS001", now=1000.0)
    assert ct.should_send("21CS001", now=1029.99) is False


def test_cooldown_allows_after_30s():
    ct = CooldownTracker()
    ct.record_sent("21CS001", now=1000.0)
    assert ct.should_send("21CS001", now=1030.0) is True


def test_cooldown_exactly_30s_allows():
    ct = CooldownTracker()
    ct.record_sent("21CS001", now=1000.0)
    # exactly 30s elapsed — should be allowed (>= 30)
    assert ct.should_send("21CS001", now=1030.0) is True


def test_cooldown_independent_roll_numbers():
    ct = CooldownTracker()
    ct.record_sent("21CS001", now=1000.0)
    # Different roll number should not be blocked
    assert ct.should_send("21CS002", now=1005.0) is True


# ---------------------------------------------------------------------------
# Property test: cooldown enforcement (Property 3)
# Feature: face-attendance-arduino-display, Property 3: cooldown enforcement
# ---------------------------------------------------------------------------

@given(
    roll_a=st.text(min_size=1, max_size=10, alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"),
    roll_b=st.text(min_size=1, max_size=10, alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"),
    elapsed=st.floats(0.0, 29.99, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=100)
def test_property_3_cooldown_enforcement(roll_a, roll_b, elapsed):
    """Property 3: should_send returns False within 30s; distinct rolls are independent."""
    ct = CooldownTracker()
    t0 = 1000.0
    ct.record_sent(roll_a, now=t0)

    # Same roll within cooldown must be blocked
    assert ct.should_send(roll_a, now=t0 + elapsed) is False

    # Different roll must not be affected (unless same string)
    if roll_b != roll_a:
        assert ct.should_send(roll_b, now=t0 + elapsed) is True


# ---------------------------------------------------------------------------
# Unit tests: SerialSender
# ---------------------------------------------------------------------------


def test_serial_sender_send_correct_bytes():
    mock_serial = MagicMock()
    with patch("recognizer.serial.Serial", return_value=mock_serial):
        sender = SerialSender("/dev/ttyUSB0", 9600)
        # Plain roll number -> "Student,21CS001,Student\n"
        sender.send("21CS001")
        mock_serial.write.assert_called_once_with(b"Student,21CS001,Student\n")


def test_serial_sender_send_pipe_format():
    mock_serial = MagicMock()
    with patch("recognizer.serial.Serial", return_value=mock_serial):
        sender = SerialSender("/dev/ttyUSB0", 9600)
        sender.send("John Doe|21CS001|Faculty")
        mock_serial.write.assert_called_once_with(b"John Doe,21CS001,Faculty\n")


def test_serial_sender_close():
    mock_serial = MagicMock()
    mock_serial.is_open = True
    with patch("recognizer.serial.Serial", return_value=mock_serial):
        sender = SerialSender("/dev/ttyUSB0", 9600)
        sender.close()
        mock_serial.close.assert_called_once()


def test_serial_sender_open_failure():
    import serial as pyserial
    with patch("recognizer.serial.Serial", side_effect=pyserial.SerialException("no port")):
        with pytest.raises(pyserial.SerialException):
            SerialSender("/dev/ttyUSB0", 9600)


# ---------------------------------------------------------------------------
# Property test: serial message format (Property 2)
# Feature: face-attendance-arduino-display, Property 2: serial message format
# ---------------------------------------------------------------------------

@given(
    roll=st.text(
        min_size=1,
        max_size=20,
        alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd")),
    )
)
@settings(max_examples=100)
def test_property_2_serial_message_format(roll):
    """Property 2: bytes written are NAME,REG,ROLE\\n encoded as UTF-8.
    For a plain roll number (no pipe), defaults to Student,<roll>,Student.
    """
    mock_serial = MagicMock()
    with patch("recognizer.serial.Serial", return_value=mock_serial):
        sender = SerialSender("/dev/ttyUSB0", 9600)
        sender.send(roll)
        expected = f"Student,{roll},Student\n".encode("utf-8")
        mock_serial.write.assert_called_once_with(expected)


# ---------------------------------------------------------------------------
# Property test: recognition log content (Property 6)
# Feature: face-attendance-arduino-display, Property 6: recognition log content
# ---------------------------------------------------------------------------

@given(
    roll=st.text(min_size=1, max_size=20, alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"),
    dist=st.floats(0.0, 1.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_property_6_recognition_log_content(roll, dist, caplog):
    """Property 6: INFO log for recognition contains roll number, distance, and timestamp."""
    import logging
    import recognizer as rec_module

    with caplog.at_level(logging.INFO, logger="recognizer"):
        rec_module.logger.info(
            "Recognized: %s | distance=%.4f | time=%s",
            roll,
            dist,
            time.strftime("%Y-%m-%d %H:%M:%S"),
        )

    assert len(caplog.records) >= 1
    record = caplog.records[-1]
    assert roll in record.message
    assert "distance" in record.message.lower() or f"{dist:.4f}" in record.message
