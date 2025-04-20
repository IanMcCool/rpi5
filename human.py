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
H_FOV = 54               # degrees horizontal FOV
V_FOV = 41               # degrees vertical FOV
FRAME_X_PIXELS = 640
FRAME_Y_PIXELS = 480
CENTER_X_PIXELS = FRAME_X_PIXELS // 2
CENTER_Y_PIXELS = FRAME_Y_PIXELS // 2

# servo angle limits (degrees)
CENTER_DEGREE     = 90.0
RIGHT_FROM_CENTER = 117.0  # 90 + (54/2)
LEFT_FROM_CENTER  =  63.0  # 90 - (54/2)
TOP_FROM_CENTER   = 110.5  # 90 + (41/2)
BOTTOM_FROM_CENTER=  69.5  # 90 - (41/2)

# precompute pixel→degree conversions
deg_per_px_x = H_FOV / FRAME_X_PIXELS
deg_per_px_y = V_FOV / FRAME_Y_PIXELS

# proportional gain for control
Kp = 0.5

from hailo_apps_infra.hailo_rpi_common import (
    get_caps_from_pad,
    get_numpy_from_buffer,
    app_callback_class,
)
from hailo_apps_infra.pose_estimation_pipeline import GStreamerPoseEstimationApp

# ——— USER DATA CLASS FOR STATE ———
class user_app_callback_class(app_callback_class):
    def __init__(self):
        super().__init__()
        # initialize pan/tilt at center
        self.current_pan_angle  = CENTER_DEGREE
        self.current_tilt_angle = CENTER_DEGREE

# ——— CALLBACK ———
def app_callback(pad, info, user_data):
    buffer = info.get_buffer()
    if buffer is None:
        return Gst.PadProbeReturn.OK

    # frame count
    user_data.increment()
    string_to_print = f"Frame count: {user_data.get_count()}\n"

    # get video metadata
    fmt, width, height = get_caps_from_pad(pad)
    frame = None
    if user_data.use_frame and fmt and width and height:
        frame = get_numpy_from_buffer(buffer, fmt, width, height)

    # run detection + landmarks
    roi        = hailo.get_roi_from_buffer(buffer)
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)
    keypoints  = get_keypoints()

    for det in detections:
        if det.get_label() != "person":
            continue

        # bounding box + confidence
        bbox       = det.get_bbox()
        confidence = det.get_confidence()
        track      = det.get_objects_typed(hailo.HAILO_UNIQUE_ID)
        track_id   = track[0].get_id() if len(track)==1 else 0
        string_to_print += (
            f"Detection: ID={track_id}  Conf={confidence:.2f}\n"
        )

        # landmarks
        landmarks = det.get_objects_typed(hailo.HAILO_LANDMARKS)
        if not landmarks:
            continue

        points = landmarks[0].get_points()
        # only nose
        idx   = keypoints['nose']
        point = points[idx]

        # compute nose coords in pixels
        x = (point.x() * bbox.width() + bbox.xmin()) * width
        y = (point.y() * bbox.height() + bbox.ymin()) * height
        string_to_print += f"nose: x={x:.2f}, y={y:.2f}\n"

        # draw on frame
        if user_data.use_frame:
            cv2.circle(frame, (int(x),int(y)), 5, (0,255,0), -1)

        # ——— PROPORTIONAL CONTROL ———
        error_x = x - CENTER_X_PIXELS
        error_y = CENTER_Y_PIXELS - y

        delta_pan  = error_x * deg_per_px_x
        delta_tilt = error_y * deg_per_px_y

        pan_angle  = user_data.current_pan_angle  + Kp * delta_pan
        tilt_angle = user_data.current_tilt_angle + Kp * delta_tilt

        # clamp to servo limits
        pan_angle  = max(LEFT_FROM_CENTER,   min(RIGHT_FROM_CENTER,  pan_angle))
        tilt_angle = max(BOTTOM_FROM_CENTER, min(TOP_FROM_CENTER,    tilt_angle))

        # update state
        user_data.current_pan_angle  = pan_angle
        user_data.current_tilt_angle = tilt_angle

        # send over UART
        if serial_port:
            msg = f"{pan_angle:.1f},{tilt_angle:.1f}\n"
            serial_port.write(msg.encode('utf-8'))
            serial_port.flush()

    # display frame if requested
    if user_data.use_frame and frame is not None:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        user_data.set_frame(frame)

    print(string_to_print)
    return Gst.PadProbeReturn.OK

# COCO keypoint map
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
    # initialize GStreamer app with our callback & state
    user_data = user_app_callback_class()
    app       = GStreamerPoseEstimationApp(app_callback, user_data)
    app.run()
