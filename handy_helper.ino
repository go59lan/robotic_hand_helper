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

void toggleFinger(int finger) {

    if (finger < 0 || finger > 4)
        return;

    setFinger(finger, !fingerOpen[finger]);
}

void openAll() {
    for (int i = 0; i < 5; i++) {
        setFinger(i, true);
    }
}

void closeAll() {
    for (int i = 4; i <= 0; i--) {
        setFinger(i, false);
    }
}

void processCommand(String command) {

    command.trim();

    if (command.length() == 0)
        return;

    // Example command:
    // "0,1,2"

    int start = 0;

    while (start < command.length()) {

        int comma = command.indexOf(',', start);

        String number;

        if (comma == -1) {
            number = command.substring(start);
            start = command.length();
        }
        else {
            number = command.substring(start, comma);
            start = comma + 1;
        }

        int finger = number.toInt();

            if(finger == 5)
            {
                closeAll();
                break;
            }

        toggleFinger(finger);
    }

    Serial.println("OK");
}

void setup() {

    Serial.begin(115200);

    pwm.begin();
    pwm.setPWMFreq(50);

    delay(500);

    Serial.println("HAND_READY");
}

void loop() {

    if (Serial.available()) {

        String command = Serial.readStringUntil('\n');
        processCommand(command);
    }
}