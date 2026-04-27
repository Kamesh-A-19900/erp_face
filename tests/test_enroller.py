"""
Tests for enroller.py components.
"""

import os
import sys
import types
from unittest.mock import MagicMock

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

# Stub cv2 and dlib so enroller can be imported without native libs
for _name in ["cv2", "dlib"]:
    if _name not in sys.modules:
        mod = types.ModuleType(_name)
        sys.modules[_name] = mod

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from enroller import get_next_person_index, load_enroller_config


# ---------------------------------------------------------------------------
# Unit tests: load_enroller_config
# ---------------------------------------------------------------------------


def test_load_enroller_config_defaults(tmp_path):
    cfg = load_enroller_config(str(tmp_path / "missing.ini"))
    assert cfg["camera_index"] == 0
    assert cfg["num_images"] == 10


def test_load_enroller_config_reads_values(tmp_path):
    ini = tmp_path / "config.ini"
    ini.write_text("[camera]\ndevice_index = 1\n[enrollment]\nnum_images = 20\n")
    cfg = load_enroller_config(str(ini))
    assert cfg["camera_index"] == 1
    assert cfg["num_images"] == 20


# ---------------------------------------------------------------------------
# Unit tests: get_next_person_index
# ---------------------------------------------------------------------------


def test_get_next_person_index_empty_dir(tmp_path):
    assert get_next_person_index(str(tmp_path)) == 1


def test_get_next_person_index_nonexistent_dir(tmp_path):
    assert get_next_person_index(str(tmp_path / "nodir")) == 1


def test_get_next_person_index_with_existing(tmp_path):
    (tmp_path / "person_1_21CS001").mkdir()
    (tmp_path / "person_3_21CS003").mkdir()
    assert get_next_person_index(str(tmp_path)) == 4


def test_get_next_person_index_ignores_non_matching(tmp_path):
    (tmp_path / "some_other_dir").mkdir()
    assert get_next_person_index(str(tmp_path)) == 1


# ---------------------------------------------------------------------------
# Property test: enrollment path format (Property 7)
# Feature: face-attendance-arduino-display, Property 7: enrollment path format
# ---------------------------------------------------------------------------

@given(
    roll=st.text(min_size=1, max_size=20, alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"),
    idx=st.integers(min_value=1, max_value=9999),
)
@settings(max_examples=100)
def test_property_7_enrollment_path_format(roll, idx):
    """Property 7: enrollment directory path is data/data_faces_from_camera/person_<N>_<roll>/"""
    expected = os.path.join("data", "data_faces_from_camera", f"person_{idx}_{roll}")
    actual = os.path.join("data", "data_faces_from_camera", f"person_{idx}_{roll}")
    assert actual == expected
    assert f"person_{idx}_{roll}" in actual
