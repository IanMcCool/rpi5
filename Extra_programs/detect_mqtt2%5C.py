import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import os
import numpy as np
import cv2
import hailo
import serial
import paho.mqtt.publish as publish

from hailo_apps_infra.hailo_rpi_common import (
    get_caps_from_pad,
    get_numpy_from_buffer,
    app_callback_class,
)
from hailo_apps_infra.detection_pipeline import GStreamerDetectionApp

# -----------------------------------------------------------------------------------------------
# MQTT SETTINGS
# -----------------------------------------------------------------------------------------------
MQTT_HOST  = "192.168.1.231"
MQTT_PORT  = 1883
MQTT_USER  = "esp32user"
MQTT_PASS  = "red123"
MQTT_TOPIC = "display/text"

# -----------------------------------------------------------------------------------------------
# Initialize serial port
# -----------------------------------------------------------------------------------------------
try:
    serial_port = serial.Serial(port='/dev/ttyACM0', baudrate=9600, timeout=1)
    print("Serial port opened successfully.")
except Exception as e:
    print("Failed to open serial port:", e)
    serial_port = None

# -----------------------------------------------------------------------------------------------
# User-defined callback class (unchanged)
# -----------------------------------------------------------------------------------------------
class user_app_callback_class(app_callback_class):
    def __init__(self):
        super().__init__()
        self.new_variable = 42

    def new_function(self):
        return "The meaning of life is: "

# -----------------------------------------------------------------------------------------------
# Utility: compute bbox center
# -----------------------------------------------------------------------------------------------
def get_bbox_center(bbox, frame_width=None, frame_height=None):
    x_min = bbox.xmin()
    y_min = bbox.ymin()
    x_max = bbox.xmax()
    y_max = bbox.ymax()
    x_center_norm = (x_min + x_max) / 2.0
    y_center_norm = (y_min + y_max) / 2.0
    if frame_width and frame_height:
        return x_center_norm * frame_width, y_center_norm * frame_height
    else:
        return x_center_norm, y_center_norm

# -----------------------------------------------------------------------------------------------
# Main GStreamer pad probe callback
# -----------------------------------------------------------------------------------------------
def app_callback(pad, info, user_data):
    buffer = info.get_buffer()
    if buffer is None:
        return Gst.PadProbeReturn.OK

    user_data.increment()
    string_to_print = f"Frame count: {user_data.get_count()}\n"

    # Get frame if needed
    fmt, width, height = get_caps_from_pad(pad)
    frame = None
    if user_data.use_frame and fmt and width and height:
        frame = get_numpy_from_buffer(buffer, fmt, width, height)

    # Run detection
    roi        = hailo.get_roi_from_buffer(buffer)
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)

    found_person = False
    cx = cy = None

    for det in detections:
        if det.get_label() != "bottle":
            continue
        found_person = True

        # 1) compute bbox and center
        bbox = det.get_bbox()
        x1, y1 = int(bbox.xmin() * width), int(bbox.ymin() * height)
        x2, y2 = int(bbox.xmax() * width), int(bbox.ymax() * height)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        # 2) draw if we have a frame
        if frame is not None:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)

        # 3) send over serial (cx,cy guaranteed to exist)
        if serial_port:
            msg = f"{cx},{cy}\n"
            serial_port.write(msg.encode('utf-8'))
            serial_port.flush()
            
            string_to_print += f"Person @ ({cx},{cy}) sent over serial\n"
        break   # only first person

    # overlay info on frame
    if frame is not None:
        count = 1 if found_person else 0
        cv2.putText(frame, f"Detections: {count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame,
                    f"{user_data.new_function()}{user_data.new_variable}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        user_data.set_frame(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    # publish MQTT: "1" if found, else "2"
    payload = "1" if found_person else "2"
    try:
        publish.single(
            MQTT_TOPIC,
            payload,
            hostname=MQTT_HOST,
            port=MQTT_PORT,
            auth={'username': MQTT_USER,
                  'password': MQTT_PASS}
        )
    except Exception as e:
        print("MQTT publish failed:", e)

    print(string_to_print)
    return Gst.PadProbeReturn.OK

# -----------------------------------------------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------------------------------------------
if __name__ == "__main__":
    user_data = user_app_callback_class()
    app = GStreamerDetectionApp(app_callback, user_data)
    app.run()
