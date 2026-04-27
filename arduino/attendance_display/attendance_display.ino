/*
 * attendance_display.ino
 *
 * Arduino Mega 2560 + MCUFRIEND_kbv 2.4" TFT Shield
 * Receives "NAME,REG,ROLE\n" over USB serial (9600 baud) from Raspberry Pi.
 * Blinks each field in colour, then holds the final screen for 5 seconds.
 *
 * Required libraries (install via Arduino Library Manager):
 *   - MCUFRIEND_kbv
 *   - Adafruit GFX Library
 */

#include <MCUFRIEND_kbv.h>
#include <Adafruit_GFX.h>

MCUFRIEND_kbv tft;

String input = "";
String personName = "";
String reg        = "";
String role       = "";

// -----------------------------------------------------------------------
// setup()
// -----------------------------------------------------------------------
void setup() {
  Serial.begin(9600);

  uint16_t ID = tft.readID();
  tft.begin(ID);
  tft.setRotation(1);          // landscape
  tft.fillScreen(0x0000);      // black

  tft.setTextSize(2);
  tft.setTextColor(0xFFFF);    // white
  tft.setCursor(20, 20);
  tft.println("ERP READY");
}

// -----------------------------------------------------------------------
// loop() — read serial until newline, then parse and display
// -----------------------------------------------------------------------
void loop() {
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n') {
      parseData(input);
      showERP();
      input = "";
    } else {
      input += c;
    }
  }
}

// -----------------------------------------------------------------------
// parseData() — split "NAME,REG,ROLE" into globals
// -----------------------------------------------------------------------
void parseData(String data) {
  data.trim();
  int i1 = data.indexOf(',');
  int i2 = data.lastIndexOf(',');

  if (i1 < 0 || i2 <= i1) {
    // Fallback: treat whole string as REG
    personName = "Student";
    reg        = data;
    role       = "Student";
  } else {
    personName = data.substring(0, i1);
    reg        = data.substring(i1 + 1, i2);
    role       = data.substring(i2 + 1);
  }
}

// -----------------------------------------------------------------------
// blinkText() — flash a label+value 3 times in the given colour
// -----------------------------------------------------------------------
void blinkText(String label, String value, uint16_t color) {
  for (int i = 0; i < 3; i++) {
    tft.fillScreen(0x0000);
    tft.setTextSize(2);
    tft.setCursor(10, 60);
    tft.setTextColor(color);
    tft.print(label);
    tft.print(": ");
    tft.println(value);
    delay(400);
    tft.fillScreen(0x0000);
    delay(200);
  }
}

// -----------------------------------------------------------------------
// showERP() — blink each field then hold final screen 5 s
// -----------------------------------------------------------------------
void showERP() {
  blinkText("NAME", personName, 0x07E0);   // green
  blinkText("REG",  reg,        0xFFFF);   // white
  blinkText("ROLE", role,       0xF800);   // red

  // Hold final screen
  tft.fillScreen(0x0000);
  tft.setTextSize(2);
  tft.setTextColor(0xFFFF);
  tft.setCursor(10, 60);
  tft.println(personName);
  tft.println(reg);
  tft.println(role);
  delay(5000);
}
