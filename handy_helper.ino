#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

Adafruit_PWMServoDriver pwm(0x40);

const int SERVO_MIN = 100;
const int SERVO_MAX = 500;

// Current state of each finger
bool fingerOpen[5] = {
    false, false, false, false, false
};

// Convert finger number + state into servo angle
int getAngle(int finger, bool open) {

    // Thumb, index, middle
    if (finger < 3) {
        return open ? 450 : 100;
    }

    // Ring, pinky
    return open ? 100 : 450;
}

void setFinger(int finger, bool open) {

    if (finger < 0 || finger > 4)
        return;

    int angle = getAngle(finger, open);

    pwm.setPWM(finger, 0, angle);

    fingerOpen[finger] = open;
}

void openAll() {
    for (int i = 0; i < 5; i++) {
        setFinger(i, true);
    }
}

void closeAll() {
    for (int i = 4; i >= 0; i--) {
        setFinger(i, false);
    }
}

void processCommand(String command) {

    command.trim();

    if (command.length() == 0)
        return;

    // The command is the complete hand position, one character per finger:
    // "10100" -> thumb and middle open, the rest closed.

    if (command.length() != 5) {
        Serial.println("ERR");
        return;
    }

    for (int i = 0; i < 5; i++) {
        if (command[i] != '0' && command[i] != '1') {
            Serial.println("ERR");
            return;
        }
    }

    for (int i = 0; i < 5; i++) {
        setFinger(i, command[i] == '1');
    }

    Serial.println("OK");
}

void setup() {

    Serial.begin(115200);

    pwm.begin();
    pwm.setPWMFreq(50);

    // Start from a known position so it matches what Python expects.
    closeAll();

    delay(500);

    Serial.println("HAND_READY");
}

void loop() {

    if (Serial.available()) {

        String command = Serial.readStringUntil('\n');
        processCommand(command);
    }
}