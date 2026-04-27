# Requirements Document

## Introduction

This feature converts an existing Flask/tkinter-based face recognition attendance system into a fully headless pipeline running on a Raspberry Pi (16 GB RAM). A USB webcam captures live video, the existing dlib ResNet face recognition model identifies registered persons, and upon recognition the system sends the person's roll number over USB serial to an Arduino Mega. The Arduino Mega displays the roll number on a 2.4" ILI9341-based TFT LCD shield (Xcluma brand with Touch Panel and Micro SD Reader). No web UI, no tkinter window, and no ERP integration are involved.

## Hardware Setup

- **Host**: Raspberry Pi (16 GB RAM), running Python 3
- **Camera**: USB webcam (Xcluma) connected to Raspberry Pi via USB
- **Microcontroller**: Arduino Mega 2560
- **Display**: Xcluma 2.4" TFT LCD Shield with Touch Panel and Micro SD Reader (ILI9341 driver)
- **Connection**: Arduino Mega connected to Raspberry Pi via USB (serial over USB)

## Glossary

- **Recognizer**: The headless Python script on the Raspberry Pi that captures frames, runs face detection and recognition, and dispatches serial messages.
- **Face_Database**: The CSV file (`data/features_all.csv`) containing 128-D dlib face descriptors keyed by roll number.
- **Roll_Number**: A string identifier stored as the person's name in the Face_Database (e.g., `"21CS001"`).
- **Serial_Link**: The USB serial connection between the Raspberry Pi and the Arduino Mega.
- **Arduino_Display**: The Arduino Mega microcontroller connected to the 2.4" ILI9341 TFT LCD shield that renders roll numbers.
- **Enroller**: The headless Python script that captures face images for a new person and saves them to disk.
- **Feature_Extractor**: The existing `features_extraction_to_csv.py` script that computes 128-D descriptors and writes `features_all.csv`.
- **Confidence_Threshold**: The maximum Euclidean distance (default 0.4) below which a face match is accepted.
- **Cooldown_Period**: 30 seconds — the minimum time that must elapse before the same Roll_Number is sent again over the Serial_Link.

---

## Requirements

### Requirement 1: Headless Face Recognition Loop

**User Story:** As a system operator, I want a headless Python script that continuously reads from the USB webcam and recognizes faces, so that attendance can be tracked without any graphical interface.

#### Acceptance Criteria

1. THE Recognizer SHALL load the Face_Database from `data/features_all.csv` at startup.
2. IF `data/features_all.csv` does not exist at startup, THEN THE Recognizer SHALL log an error message and exit with a non-zero status code.
3. WHEN the Recognizer starts, THE Recognizer SHALL open the USB webcam using its device index (configurable, default 0).
4. IF the webcam cannot be opened, THEN THE Recognizer SHALL log an error message and exit with a non-zero status code.
5. WHILE the webcam is open, THE Recognizer SHALL read frames continuously and run dlib frontal face detection on each frame.
6. WHEN a face is detected in a frame, THE Recognizer SHALL compute its 128-D descriptor using the dlib ResNet model and compare it against all entries in the Face_Database.
7. WHEN the minimum Euclidean distance to any known face is below the Confidence_Threshold, THE Recognizer SHALL identify the face as the corresponding Roll_Number.
8. WHEN a face cannot be matched within the Confidence_Threshold, THE Recognizer SHALL classify it as "unknown" and take no further action.
9. THE Recognizer SHALL use the existing dlib shape predictor and ResNet model files located at `data/data_dlib/` without modification.
10. THE Recognizer SHALL NOT open any GUI window (no cv2.imshow, no tkinter).

---

### Requirement 2: Serial Notification to Arduino Mega

**User Story:** As a system operator, I want the recognized roll number sent over USB serial to the Arduino Mega, so that the TFT display can show who has been identified.

#### Acceptance Criteria

1. WHEN a Roll_Number is recognized, THE Recognizer SHALL transmit the Roll_Number string followed by a newline character (`\n`) over the Serial_Link.
2. THE Recognizer SHALL open the Serial_Link at startup using a configurable port name (default `/dev/ttyUSB0`) and baud rate (default 9600).
3. IF the Serial_Link cannot be opened at startup, THEN THE Recognizer SHALL log an error and exit with a non-zero status code.
4. THE Cooldown_Period SHALL be fixed at 30 seconds per Roll_Number.
5. WHILE a Roll_Number has been sent within the last 30 seconds, THE Recognizer SHALL NOT retransmit that Roll_Number.
6. THE Recognizer SHALL apply the Cooldown_Period independently per Roll_Number, so that different persons can be recognized without delay.
7. WHEN the Recognizer exits, THE Recognizer SHALL close the Serial_Link cleanly.

---

### Requirement 3: Arduino Mega TFT Display

**User Story:** As a system operator, I want the Arduino Mega to display the received roll number on the TFT LCD, so that a person can visually confirm their attendance was recorded.

#### Acceptance Criteria

1. WHEN the Arduino_Display receives a newline-terminated string over serial (Serial at 9600 baud), THE Arduino_Display SHALL parse the string as a Roll_Number.
2. WHEN a Roll_Number is parsed, THE Arduino_Display SHALL clear the TFT screen and render the Roll_Number in large, readable text centered on the display.
3. WHILE no new Roll_Number has been received, THE Arduino_Display SHALL keep the last Roll_Number visible on the TFT screen.
4. WHEN the Arduino_Display is powered on, THE Arduino_Display SHALL show a "Ready" message on the TFT screen until the first Roll_Number is received.
5. THE Arduino_Display SHALL use the Adafruit ILI9341 library and Adafruit GFX library.
6. THE Arduino_Display SHALL use the following pin mapping for the Arduino Mega with the 2.4" TFT shield:
   - CS  = 10
   - DC  = 9
   - RST = 8
   - MOSI = 51 (hardware SPI)
   - MISO = 50 (hardware SPI)
   - SCK  = 52 (hardware SPI)
7. IF a received string exceeds 20 characters, THE Arduino_Display SHALL truncate it to 20 characters before rendering.
8. THE Arduino_Display SHALL display the roll number in font size 3 or larger so it is readable from a distance.

---

### Requirement 4: Headless Face Enrollment

**User Story:** As a system operator, I want a headless command-line script to register new persons by capturing their face images, so that enrollment does not require a graphical interface.

#### Acceptance Criteria

1. THE Enroller SHALL accept a roll number as a required command-line argument (e.g., `--roll 21CS001`).
2. WHEN launched, THE Enroller SHALL open the USB webcam and capture face images automatically without requiring GUI interaction.
3. WHEN a single face is detected in a frame, THE Enroller SHALL save the cropped face image to `data/data_faces_from_camera/person_<N>_<roll_number>/`.
4. THE Enroller SHALL capture a configurable number of face images per person (default 10) and then exit automatically.
5. IF no face is detected in a frame, THE Enroller SHALL skip that frame and continue capturing.
6. IF more than one face is detected in a frame, THE Enroller SHALL skip that frame and log a warning.
7. WHEN the required number of images has been saved, THE Enroller SHALL log a completion message and exit with status code 0.

---

### Requirement 5: Configuration

**User Story:** As a system operator, I want all runtime parameters in a single configuration file, so that I can adjust settings without modifying source code.

#### Acceptance Criteria

1. THE Recognizer SHALL read configuration from a file named `config.ini` located in the project root directory.
2. THE Recognizer SHALL support the following configurable parameters: webcam device index, serial port name, baud rate, and Confidence_Threshold.
3. THE Cooldown_Period SHALL be hardcoded to 30 seconds and is NOT configurable.
4. IF `config.ini` is absent, THE Recognizer SHALL use hardcoded default values and log a warning that defaults are in use.
5. THE Enroller SHALL read the webcam device index and number of images to capture from `config.ini`.

---

### Requirement 6: Logging

**User Story:** As a system operator, I want structured log output to stdout, so that I can monitor the system and diagnose issues without a UI.

#### Acceptance Criteria

1. THE Recognizer SHALL log each recognition event with the Roll_Number, timestamp, and Euclidean distance at INFO level.
2. THE Recognizer SHALL log each "unknown face" event at DEBUG level.
3. THE Recognizer SHALL log each serial transmission at DEBUG level.
4. IF any runtime error occurs (e.g., webcam lost, serial write failure), THEN THE Recognizer SHALL log the error at ERROR level before exiting.
