#!/usr/bin/env python3
"""
detect_stm32.py

Tracks one chosen object class from a Hailo GStreamer pipeline,
draws a bounding box around the first detection per frame, sends its
center over UART, and displays live video.

Keyboard controls (in the OpenCV window)
----------------------------------------
  1  person
  2  bottle
  3  cup
  4  toothbrush
  5  book
  q  quit
  Esc quit
All GUI work is done in the main thread, so there are no Qt timer
warnings.
"""

import gi
gi.require_version("Gst", "1.0")

from gi.repository import Gst, GLib
import cv2
import hailo
import serial
import threading

# -------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------
CLASS_MAP = {
    ord("1"): "person",
    ord("2"): "bottle",
    ord("3"): "cup",
    ord("4"): "toothbrush",
    ord("5"): "book",
}

SERIAL_PATH     = "/dev/ttyACM0"
SERIAL_BAUDRATE = 9600
SERIAL_TIMEOUT  = 0.1

WINDOW_NAME     = "Hailo Detect"
FONT            = cv2.FONT_HERSHEY_SIMPLEX
TEXT_SCALE      = 1.0
TEXT_COLOR      = (0, 255, 0)
TEXT_THICKNESS  = 2

# -------------------------------------------------------------------------
# Globals (shared between pad-probe thread and GUI thread)
# -------------------------------------------------------------------------
app           = None          # will hold GStreamerDetectionApp
target_label  = "person"      # current class to track
serial_port   = None          # UART handle
latest_frame  = None          # most recent BGR frame
frame_lock    = threading.Lock()

# -------------------------------------------------------------------------
# Serial port setup
# -------------------------------------------------------------------------
try:
    serial_port = serial.Serial(
        port=SERIAL_PATH,
        baudrate=SERIAL_BAUDRATE,
        timeout=SERIAL_TIMEOUT,
    )
    print(f"Opened serial port {SERIAL_PATH} at {SERIAL_BAUDRATE} baud")
except Exception as e:
    print(f"WARNING: could not open serial port: {e}")

# -------------------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------------------
def send_center(cx: int, cy: int) -> None:
    """Send 'cx,cy' newline terminated over UART."""
    if serial_port:
        msg = f"{cx},{cy}\n"
        serial_port.write(msg.encode("utf-8"))
        serial_port.flush()

def draw_bbox_and_center(frame, bbox, cx: int, cy: int, color) -> None:
    """Draw rectangle and center point in the frame."""
    h, w = frame.shape[:2]
    x1 = int(bbox.xmin() * w)
    y1 = int(bbox.ymin() * h)
    x2 = int(bbox.xmax() * w)
    y2 = int(bbox.ymax() * h)

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.circle(frame, (cx, cy), 5, color, -1)

# -------------------------------------------------------------------------
# Hailo helper imports (after gi import)
# -------------------------------------------------------------------------
from hailo_apps_infra.hailo_rpi_common import (
    get_caps_from_pad,
    get_numpy_from_buffer,
    app_callback_class,
)
from hailo_apps_infra.detection_pipeline import GStreamerDetectionApp

# -------------------------------------------------------------------------
# UserData class for frame counter
# -------------------------------------------------------------------------
class UserData(app_callback_class):
    def __init__(self):
        super().__init__()
        self._count = 0

    def increment(self) -> None:
        self._count += 1

    def count(self) -> int:
        return self._count

# -------------------------------------------------------------------------
# Pad-probe callback (runs in GStreamer worker thread)
# -------------------------------------------------------------------------
def pad_callback(pad, info, user_data):
    global latest_frame, target_label

    buf = info.get_buffer()
    if buf is None:
        return Gst.PadProbeReturn.OK

    user_data.increment()

    fmt, width, height = get_caps_from_pad(pad)
    frame_rgb = get_numpy_from_buffer(buf, fmt, width, height)
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    roi = hailo.get_roi_from_buffer(buf)
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)

    sent = False
    for det in detections:
        if det.get_label() != target_label:
            continue

        bbox = det.get_bbox()
        cx = int((bbox.xmin() + bbox.xmax()) * 0.5 * width)
        cy = int((bbox.ymin() + bbox.ymax()) * 0.5 * height)

        draw_bbox_and_center(frame_bgr, bbox, cx, cy, TEXT_COLOR)

        if not sent:
            send_center(cx, cy)
            sent = True

    # Overlay info
    cv2.putText(
        frame_bgr,
        f"Tracking: {target_label}",
        (10, 30),
        FONT,
        TEXT_SCALE,
        TEXT_COLOR,
        TEXT_THICKNESS,
    )
    cv2.putText(
        frame_bgr,
        f"Frames: {user_data.count()}",
        (10, 60),
        FONT,
        TEXT_SCALE,
        TEXT_COLOR,
        TEXT_THICKNESS,
    )

    # Share frame with GUI thread
    with frame_lock:
        latest_frame = frame_bgr.copy()

    return Gst.PadProbeReturn.OK

# -------------------------------------------------------------------------
# GUI loop (runs in main thread via GLib timeout)
# -------------------------------------------------------------------------
def gui_loop() -> bool:
    """Called every few ms in the main thread to show frame and read keys."""
    global latest_frame, target_label, app

    with frame_lock:
        frame = None if latest_frame is None else latest_frame.copy()

    if frame is not None:
        cv2.imshow(WINDOW_NAME, frame)
        key = cv2.waitKey(1) & 0xFF

        if key in CLASS_MAP:
            target_label = CLASS_MAP[key]
            print(f"*** Switched to '{target_label}' ***")
        elif key in (ord("q"), 27):
            GLib.idle_add(app.quit)  # schedule quit in GLib main loop
            return False             # stop this timeout

    return True                      # keep the timeout active

# -------------------------------------------------------------------------
# Main entry point
# -------------------------------------------------------------------------
if __name__ == "__main__":
    # Initialize GStreamer
    Gst.init(None)

    # Build and start the detection app
    app = GStreamerDetectionApp(pad_callback, UserData())

    # Run gui_loop every 10 ms in the main GLib thread
    GLib.timeout_add(10, gui_loop)

    try:
        app.run()
    finally:
        cv2.destroyAllWindows()
        if serial_port:
            serial_port.close()
