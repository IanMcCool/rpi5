import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import os
import numpy as np
import cv2
import hailo
import serial

# ——— SERIAL COMMUNICATION SETUP ———
try:
    serial_port = serial.Serial(port='/dev/ttyACM0', baudrate=9600, timeout=1)
    print("Serial port opened successfully.")
except Exception as e:
    print("Failed to open serial port:", e)
    serial_port = None

# ——— CAMERA & SERVO CONSTANTS ———
H_FOV = 54.0            # degrees horizontal FOV
V_FOV = 41.0            # degrees vertical FOV

# Updated resolution
FRAME_X_PIXELS = 1920
FRAME_Y_PIXELS = 1080
CENTER_X_PIXELS = FRAME_X_PIXELS // 2   # 960
CENTER_Y_PIXELS = FRAME_Y_PIXELS // 2   # 540

# Servo center and dynamic limits based on FOV
CENTER_DEGREE      = 90.0
RIGHT_FROM_CENTER  = CENTER_DEGREE + H_FOV / 2.0   # 90 + 27  = 117°
LEFT_FROM_CENTER   = CENTER_DEGREE - H_FOV / 2.0   # 90 - 27  = 63°
TOP_FROM_CENTER    = CENTER_DEGREE + V_FOV / 2.0   # 90 + 20.5= 110.5°
BOTTOM_FROM_CENTER = CENTER_DEGREE - V_FOV / 2.0   # 90 - 20.5= 69.5°

# Dead‑zone in pixels around center (unchanged)
DEADZONE_PX_X = 20
DEADZONE_PX_Y = 20

# Pixel→degree conversion factors (will change with resolution)
deg_per_px_x = H_FOV / FRAME_X_PIXELS   # = 54° / 1920 ≈ 0.0281°/px
deg_per_px_y = V_FOV / FRAME_Y_PIXELS   # = 41° / 1080 ≈ 0.0380°/px

# Proportional gain
Kp = 0.5

from hailo_apps_infra.hailo_rpi_common import (
    get_caps_from_pad,
    get_numpy_from_buffer,
    app_callback_class,
)
from hailo_apps_infra.pose_estimation_pipeline import GStreamerPoseEstimationApp

# ——— USER DATA & STATE ———
class user_app_callback_class(app_callback_class):
    def __init__(self):
        super().__init__()
        # start servos centered
        self.current_pan_angle  = CENTER_DEGREE
        self.current_tilt_angle = CENTER_DEGREE

# ——— CALLBACK FUNCTION ———
def app_callback(pad, info, user_data):
    buffer = info.get_buffer()
    if buffer is None:
        return Gst.PadProbeReturn.OK

    user_data.increment()
    string_to_print = f"Frame: {user_data.get_count()}\n"

    # get frame data if requested
    fmt, width, height = get_caps_from_pad(pad)
    frame = None
    if user_data.use_frame and fmt and width and height:
        frame = get_numpy_from_buffer(buffer, fmt, width, height)

    # detection + landmarks
    roi        = hailo.get_roi_from_buffer(buffer)
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)
    keypoints  = get_keypoints()

    for det in detections:
        if det.get_label() != "person":
            continue

        # detection info
        bbox       = det.get_bbox()
        confidence = det.get_confidence()
        track      = det.get_objects_typed(hailo.HAILO_UNIQUE_ID)
        track_id   = track[0].get_id() if len(track) == 1 else 0
        string_to_print += f"ID={track_id}  Conf={confidence:.2f}\n"

        landmarks = det.get_objects_typed(hailo.HAILO_LANDMARKS)
        if not landmarks:
            continue

        pts = landmarks[0].get_points()
        idx = keypoints['nose']
        pt  = pts[idx]

        # pixel coords (float)
        x = (pt.x() * bbox.width() + bbox.xmin()) * width
        y = (pt.y() * bbox.height() + bbox.ymin()) * height
        string_to_print += f"nose: x={x:.2f}, y={y:.2f}\n"

        if user_data.use_frame:
            cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)

        # compute error from center
        error_x = x - CENTER_X_PIXELS
        error_y = CENTER_Y_PIXELS - y

        # only adjust if outside dead‑zone
        if abs(error_x) > DEADZONE_PX_X or abs(error_y) > DEADZONE_PX_Y:
            # map pixel error → angle error
            delta_pan  = error_x * deg_per_px_x
            delta_tilt = error_y * deg_per_px_y

            # proportional update
            pan_angle  = user_data.current_pan_angle  + Kp * delta_pan
            tilt_angle = user_data.current_tilt_angle + Kp * delta_tilt

            # clamp to mechanical limits
            pan_angle  = max(LEFT_FROM_CENTER,   min(RIGHT_FROM_CENTER,  pan_angle))
            tilt_angle = max(BOTTOM_FROM_CENTER, min(TOP_FROM_CENTER,    tilt_angle))

            # update state
            user_data.current_pan_angle  = pan_angle
            user_data.current_tilt_angle = tilt_angle

            # send to MCU
            if serial_port:
                msg = f"{pan_angle:.1f},{tilt_angle:.1f}\n"
                serial_port.write(msg.encode('utf-8'))
                serial_port.flush()

    # display frame if needed
    if user_data.use_frame and frame is not None:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        user_data.set_frame(frame)

    print(string_to_print)
    return Gst.PadProbeReturn.OK

# COCO keypoint mapping
def get_keypoints():
    return {
        'nose': 0, 'left_eye':1, 'right_eye':2,
        'left_ear':3, 'right_ear':4,
        'left_shoulder':5, 'right_shoulder':6,
        'left_elbow':7, 'right_elbow':8,
        'left_wrist':9, 'right_wrist':10,
        'left_hip':11, 'right_hip':12,
        'left_knee':13, 'right_knee':14,
        'left_ankle':15, 'right_ankle':16,
    }

if __name__ == "__main__":
    user_data = user_app_callback_class()
    app       = GStreamerPoseEstimationApp(app_callback, user_data)
    app.run()
