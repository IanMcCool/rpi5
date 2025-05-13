"""
detect_with_servo.py
Run person-detection on Pi with Hailo-8 and directly drive pan/tilt servos via PCA9685,
while printing the current servo angles to console.
"""
import gi
# GStreamer setup
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

import hailo
import numpy as np
import cv2
from adafruit_servokit import ServoKit

from hailo_apps_infra.hailo_rpi_common import (
    get_caps_from_pad,
    get_numpy_from_buffer,
    app_callback_class,
)
from hailo_apps_infra.detection_pipeline import GStreamerDetectionApp

# === Servo Configuration ===
PAN_CHANNEL        = 0
TILT_CHANNEL       = 1
SERVO_MIN_PULSE_US = 500
SERVO_MAX_PULSE_US = 2000
kit = ServoKit(channels=16)
kit.servo[PAN_CHANNEL].set_pulse_width_range(SERVO_MIN_PULSE_US, SERVO_MAX_PULSE_US)
kit.servo[TILT_CHANNEL].set_pulse_width_range(SERVO_MIN_PULSE_US, SERVO_MAX_PULSE_US)

# === Control Parameters ===
FRAME_CENTER_X = 640
FRAME_CENTER_Y = 360
# Pixel-per-degree conversion based on Arducam V1 720p FOV: 54 H x 41 V
default_horizontal_fov = 54.0  # degrees
default_vertical_fov   = 41.0  # degrees
M_PAN  = 2*(1280.0 / default_horizontal_fov)  # ~23.70 px/deg
M_TILT =  2*(720.0 / default_vertical_fov)    # ~17.56 px/deg
DEAD_ZONE_PAN   = 2.0    # degrees
DEAD_ZONE_TILT  = 1.0    # degrees
SMOOTH_PAN        = 0.15   # smoothing gain
SMOOTH_TILT       = 0.15   # smoothing gain

# === State ===
current_pan_angle  = 90.0  # degrees
current_tilt_angle = 90.0  # degrees
# Center servos at start
kit.servo[PAN_CHANNEL].angle  = current_pan_angle
kit.servo[TILT_CHANNEL].angle = current_tilt_angle

class UserAppCallback(app_callback_class):
    """Extend the Hailo callback class for state if needed."""
    pass


def app_callback(pad, info, user_data):
    """
    Callback for each frame: detect first person, compute pan/tilt adjustments,
    drive servos accordingly, and print current angles.
    """
    global current_pan_angle, current_tilt_angle

    buffer = info.get_buffer()
    if buffer is None:
        return Gst.PadProbeReturn.OK

    # Get video format and dimensions
    video_format, width, height = get_caps_from_pad(pad)

    # Optionally convert to numpy frame for overlay
    frame = None
    if user_data.use_frame and video_format and width and height:
        frame = get_numpy_from_buffer(buffer, video_format, width, height)

    # Run Hailo detections using hailo SDK
    roi = hailo.get_roi_from_buffer(buffer)
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)

    for det in detections:
        if det.get_label() != "remote":
            continue

        # Compute bounding box center in pixels
        x1, y1 = int(det.get_bbox().xmin() * width), int(det.get_bbox().ymin() * height)
        x2, y2 = int(det.get_bbox().xmax() * width), int(det.get_bbox().ymax() * height)
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0

        # Compute angular errors
        delta_pan  = (cx - FRAME_CENTER_X) / M_PAN
        delta_tilt = (cy - FRAME_CENTER_Y) / M_TILT

        # Apply dead-band and smoothing gain
        if abs(delta_pan) > DEAD_ZONE_PAN:
            delta_pan *= SMOOTH_PAN
        else:
            delta_pan = 0.0

        if abs(delta_tilt) > DEAD_ZONE_TILT:
            delta_tilt *= SMOOTH_TILT
        else:
            delta_tilt = 0.0

        # Compute new angles and clamp between 0-180
        new_pan  = max(0.0, min(180.0, current_pan_angle  + delta_pan))
        new_tilt = max(0.0, min(180.0, current_tilt_angle + delta_tilt))

        # Drive servos
        kit.servo[PAN_CHANNEL].angle  = new_pan
        kit.servo[TILT_CHANNEL].angle = new_tilt

        # Update state
        current_pan_angle, current_tilt_angle = new_pan, new_tilt

        # Print current servo angles
        print(f"Current servo angles -> Pan: {current_pan_angle:.2f}, Tilt: {current_tilt_angle:.2f}")

        # Overlay on frame if enabled
        if frame is not None:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.circle(frame, (int(cx), int(cy)), 5, (255,0,0), -1)
            cv2.putText(
                frame,
                f"Pan:{current_pan_angle:.1f}, Tilt:{current_tilt_angle:.1f}",
                (10,50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0,255,0),
                2
            )
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            user_data.set_frame(frame_bgr)

        # Only handle first person per frame
        break

    return Gst.PadProbeReturn.OK


if __name__ == "__main__":
    # Initialize and run the detection application
    user_data = UserAppCallback()
    app = GStreamerDetectionApp(app_callback, user_data)
    app.run()
