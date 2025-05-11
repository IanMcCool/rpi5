#include "mbed.h"

// USB serial interface
Serial pc(USBTX, USBRX);

// Servo PWM outputs on D3 (pan) and D5 (tilt)
PwmOut pan_servo(D3);
PwmOut tilt_servo(D5);

// Frame-center constants and pixel-to-degree conversion
const int   FRAME_CENTER_X = 640;
const int   FRAME_CENTER_Y = 360;
const float M_PAN          = 20.28f;   // pixels per degree (horizontal)
const float M_TILT         = 18.90f;   // pixels per degree (vertical)

// Servo pulse parameters for 20 ms period (50 Hz)
const float PULSE_MIN    = 0.05f;  // 5% duty cycle -> 0°
const float PULSE_CENTER = 0.075f; // 7.5% duty cycle -> 90°
const float PULSE_MAX    = 0.10f;  // 10% duty cycle -> 180°

const float GAIN        = 0.35f;   // Proportional gain for smoothing
const float DEAD_BAND   = 1.0f;    // Degrees; ignore small corrections

const float GAIN_TILT        = 0.35f;   // Proportional gain for smoothing
const float DEAD_BAND_TILT   = 1.0f;    // Degrees; ignore small corrections


// Current servo angles (degrees)
float current_pan_angle  = 90.0f;
float current_tilt_angle = 90.0f;

// Convert angle (0–180°) to duty cycle fraction
float angle_to_duty(float angle) {
    return PULSE_MIN + (angle / 180.0f) * (PULSE_MAX - PULSE_MIN);
}

// Move one servo to the target angle (clamped between 0° and 180°)
void move_servo(PwmOut &servo, float &current_angle, float target_angle) {
    if (target_angle < 0.0f)   target_angle = 0.0f;
    if (target_angle > 180.0f) target_angle = 180.0f;
    current_angle = target_angle;
    servo.write(angle_to_duty(current_angle));
}

int main() {
    // Initialize serial at 9600 baud
    pc.baud(9600);

    // Set servo PWM period to 20 ms (50 Hz)
    pan_servo.period_ms(20);
    tilt_servo.period_ms(20);

    // Center both servos
    move_servo(pan_servo, current_pan_angle,  90.0f);
    move_servo(tilt_servo, current_tilt_angle, 90.0f);

    // Buffer for incoming "x,y\n" data
    char buf[32];
    int  idx = 0;

    while (true) {
        // If data is available over serial
        if (pc.readable()) {
            char c = pc.getc();

            // On newline or carriage return, process buffer
            if (c == '\n' || c == '\r') {
                buf[idx] = '\0';  // null-terminate string
                idx = 0;

                int x = 0, y = 0;
                // Parse "x,y"
                if (sscanf(buf, "%d,%d", &x, &y) == 2) {
                    // Calculate angular deltas
                    float delta_pan  = (x - FRAME_CENTER_X) / M_PAN;
                    float delta_tilt = (y - FRAME_CENTER_Y) / M_TILT;

                    // Apply dead-band and proportional gain
                    if (fabsf(delta_pan) > DEAD_BAND)
                        delta_pan *= GAIN;
                    else
                        delta_pan = 0.0f;

                    if (fabsf(delta_tilt) > DEAD_BAND_TILT)
                        delta_tilt *= GAIN_TILT;
                    else
                        delta_tilt = 0.0f;

                    // Compute new target angles
                    float new_pan  = current_pan_angle  + delta_pan;
                    float new_tilt = current_tilt_angle + delta_tilt;

                    // Move servos
                    move_servo(pan_servo, current_pan_angle,  new_pan);
                    move_servo(tilt_servo, current_tilt_angle, new_tilt);

                    // Echo back only the new angles
                    pc.printf("%.2f,%.2f\n", new_pan, new_tilt);
                }
            }
            // Otherwise, accumulate characters in buffer
            else if (idx < (int)sizeof(buf) - 1) {
                buf[idx++] = c;
            }
        }
    }
}
