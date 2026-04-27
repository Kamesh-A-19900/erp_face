# Implementation Plan: Face Attendance Arduino Display

## Overview

Implement the headless face recognition pipeline by creating `recognizer.py`, `enroller.py`, `config.ini`, and the Arduino sketch. The existing dlib recognition logic is preserved; all GUI, SQLite, and tkinter code is excluded from the new scripts.

## Tasks

- [ ] 1. Create `config.ini` and config loader
  - [x] 1.1 Create `config.ini` in the project root with sections `[camera]`, `[serial]`, `[recognition]`, `[enrollment]` and all default values from the design
    - _Requirements: 5.1, 5.2, 5.3_
  - [x] 1.2 Implement `load_config(path)` in `recognizer.py` using `configparser`; return a dict with typed defaults; log WARNING if file is absent
    - _Requirements: 5.1, 5.2, 5.4_
  - [ ]* 1.3 Write property test for config round-trip (Property 5)
    - **Property 5: Config round-trip**
    - **Validates: Requirements 5.2**
    - Write to a temp `config.ini`, call `load_config`, assert values match originals
    - `# Feature: face-attendance-arduino-display, Property 5: config round-trip`
  - [ ]* 1.4 Write unit tests for `load_config` with absent file (defaults returned, warning logged) and with present file
    - _Requirements: 5.4_

- [ ] 2. Implement core recognition utilities in `recognizer.py`
  - [x] 2.1 Implement `euclidean_distance(a, b)` and `load_face_database(csv_path)` — port the CSV loading logic from `attendance_taker.py`, returning `(names, descriptors)`; raise `FileNotFoundError` if CSV is absent
    - _Requirements: 1.1, 1.2, 1.6_
  - [x] 2.2 Implement `find_best_match(descriptor, names, descriptors, threshold)` — compute distances, return `(roll_number, min_dist)` or `("unknown", min_dist)`
    - _Requirements: 1.7, 1.8_
  - [ ]* 2.3 Write property test for face matching correctness (Property 1)
    - **Property 1: Face matching correctness**
    - **Validates: Requirements 1.7, 1.8**
    - Use `hypothesis` to generate random 128-D float lists as descriptors and databases; assert result is correct roll number when min distance < threshold, else "unknown"
    - `# Feature: face-attendance-arduino-display, Property 1: face matching correctness`
  - [ ]* 2.4 Write unit test for `load_face_database` with missing CSV (FileNotFoundError) and with a valid two-row CSV
    - _Requirements: 1.1, 1.2_

- [ ] 3. Implement `CooldownTracker` in `recognizer.py`
  - [x] 3.1 Implement `CooldownTracker` class with `COOLDOWN_SECONDS = 30`, `should_send(roll, now)`, and `record_sent(roll, now)` — use injectable `now` parameter (defaults to `time.time()`)
    - _Requirements: 2.4, 2.5, 2.6_
  - [ ]* 3.2 Write property test for cooldown enforcement (Property 3)
    - **Property 3: Cooldown enforcement**
    - **Validates: Requirements 2.5, 2.6**
    - Generate random roll numbers and elapsed times < 30s; assert `should_send` returns False after `record_sent`; assert distinct roll numbers are independent
    - `# Feature: face-attendance-arduino-display, Property 3: cooldown enforcement`
  - [ ]* 3.3 Write unit tests for `CooldownTracker` at boundary conditions (exactly 30s, 29.99s, 30.01s)
    - _Requirements: 2.5_

- [ ] 4. Implement `SerialSender` in `recognizer.py`
  - [x] 4.1 Implement `SerialSender.__init__` opening `serial.Serial(port, baud_rate)`; implement `send(roll_number)` writing `(roll_number + "\n").encode("utf-8")`; implement `close()`
    - _Requirements: 2.1, 2.2, 2.7_
  - [ ]* 4.2 Write property test for serial message format (Property 2)
    - **Property 2: Serial message format**
    - **Validates: Requirements 2.1**
    - Mock `serial.Serial`; generate random roll number strings; assert bytes written equal `(roll + "\n").encode("utf-8")`
    - `# Feature: face-attendance-arduino-display, Property 2: serial message format`
  - [ ]* 4.3 Write unit tests for `SerialSender` with mocked serial: successful send, `SerialException` on open, `close` called on exit
    - _Requirements: 2.2, 2.3, 2.7_

- [x] 5. Checkpoint — Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 6. Implement `Recognizer` class and `main()` in `recognizer.py`
  - [x] 6.1 Implement `Recognizer.__init__` storing config; implement `run()` with the full loop: load DB (exit 1 on failure), open serial (exit 1 on failure), open webcam (exit 1 on failure), frame loop with dlib detection → `find_best_match` → `CooldownTracker` → `SerialSender.send`; handle `KeyboardInterrupt` for clean shutdown
    - Use dlib model paths `data/data_dlib/shape_predictor_68_face_landmarks.dat` and `data/data_dlib/dlib_face_recognition_resnet_model_v1.dat`
    - No `cv2.imshow`, no `cv2.namedWindow`, no tkinter
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 1.10, 2.1, 2.5, 2.6, 2.7_
  - [x] 6.2 Add structured logging: INFO for recognition events (roll number, timestamp, distance), DEBUG for unknown faces, DEBUG for serial transmissions, ERROR for runtime failures
    - _Requirements: 6.1, 6.2, 6.3, 6.4_
  - [ ]* 6.3 Write property test for recognition log content (Property 6)
    - **Property 6: Recognition log content**
    - **Validates: Requirements 6.1**
    - Use `caplog` fixture; generate random roll numbers and distances; call the logging statement directly; assert log record contains roll number, distance, and a timestamp
    - `# Feature: face-attendance-arduino-display, Property 6: recognition log content`
  - [ ]* 6.4 Write unit tests for `Recognizer.run` startup failure paths using mocks: missing CSV exits 1, serial open failure exits 1, webcam open failure exits 1
    - _Requirements: 1.2, 1.4, 2.3_

- [ ] 7. Implement `enroller.py`
  - [x] 7.1 Implement `load_enroller_config(path)` reading `camera.device_index` and `enrollment.num_images` from `config.ini` with defaults
    - _Requirements: 5.5_
  - [x] 7.2 Implement `get_next_person_index(base_dir)` scanning `data/data_faces_from_camera/` for `person_<N>_*` directories and returning `max(N) + 1` (or 1 if empty)
    - _Requirements: 4.3_
  - [ ]* 7.3 Write property test for enrollment path format (Property 7)
    - **Property 7: Enrollment path format**
    - **Validates: Requirements 4.3**
    - Generate random roll number strings and integer indices ≥ 1; assert constructed path equals `data/data_faces_from_camera/person_<N>_<roll>/`
    - `# Feature: face-attendance-arduino-display, Property 7: enrollment path format`
  - [x] 7.4 Implement `Enroller` class: `__init__` stores roll number and config; `run()` determines next index, creates directory, opens webcam, loops until `num_images` saved — skip frames with 0 or >1 faces, save cropped image for exactly 1 face, log completion and exit 0
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7_
  - [x] 7.5 Implement `main()` with `argparse` requiring `--roll`; call `Enroller.run()`
    - _Requirements: 4.1_
  - [ ]* 7.6 Write unit tests for `Enroller`: missing `--roll` argument exits with error; frame with 0 faces is skipped; frame with 2 faces logs warning and is skipped; correct number of images saved
    - _Requirements: 4.1, 4.5, 4.6, 4.4_

- [ ] 8. Implement Arduino sketch `arduino/attendance_display/attendance_display.ino`
  - [x] 8.1 Create `arduino/attendance_display/` directory and `attendance_display.ino` with pin definitions (CS=10, DC=9, RST=8), `Adafruit_ILI9341` and `Adafruit_GFX` includes, global `tft` instance, and `setup()` initializing Serial at 9600 baud, TFT, black fill, and "Ready" centered text at size 3
    - _Requirements: 3.4, 3.5, 3.6, 3.8_
  - [x] 8.2 Implement `loop()` polling `Serial.available()`, reading until `\n` via `Serial.readStringUntil('\n')`, trimming whitespace, truncating to 20 characters, and calling `displayRoll(roll)`
    - _Requirements: 3.1, 3.7_
  - [x] 8.3 Implement `displayRoll(String roll)` clearing the screen (black fill), computing centered cursor position for font size 3 (char width 18px, height 24px, screen 240×320), setting text color white, and printing the roll number
    - _Requirements: 3.2, 3.3, 3.8_

- [x] 9. Final checkpoint — Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.
