import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import os
import numpy as np
import cv2
import hailo
import serial  # Import pyserial
import threading  # For running a serial reader thread

from hailo_apps_infra.hailo_rpi_common import (
    get_caps_from_pad,
    get_numpy_from_buffer,
    app_callback_class,
)
from hailo_apps_infra.pose_estimation_pipeline import GStreamerPoseEstimationApp

# -----------------------------------------------------------------------------------------------
# Setup serial communication with NucleoF401RE
# -----------------------------------------------------------------------------------------------
# Change 'port' and 'baudrate' to match your configuration.
# On Linux, this might be something like '/dev/ttyACM0'; on Windows, 'COM3' or similar.
try:
    ser = serial.Serial(port='/dev/ttyACM0', baudrate=9600, timeout=1)
    print("Serial port opened successfully.")
except Exception as e:
    print("Failed to open serial port:", e)
    ser = None

# -----------------------------------------------------------------------------------------------
# Function to continuously read echoed messages from the serial port.
# -----------------------------------------------------------------------------------------------
def read_serial():
    """Continuously reads lines from the serial port and prints them."""
    while True:
        if ser is not None and ser.is_open:
            try:
                # Read a line of response from the Nucleo board.
                echo = ser.readline().decode('utf-8', errors='replace').strip()
                if echo:
                    print("Echo received:", echo)
            except Exception as e:
                print("Error reading from serial port:", e)
        else:
            # If serial port is not available, just sleep for a bit.
            pass

# Start the serial reading thread (if the serial port was successfully opened)
if ser is not None:
    read_thread = threading.Thread(target=read_serial, daemon=True)
    read_thread.start()

# -----------------------------------------------------------------------------------------------
# User-defined class to be used in the callback function
# -----------------------------------------------------------------------------------------------
class user_app_callback_class(app_callback_class):
    def __init__(self):
        super().__init__()

# -----------------------------------------------------------------------------------------------
# Function to send coordinates to the NucleoF401RE via UART.
# -----------------------------------------------------------------------------------------------
def send_eye_coordinates(eye_label, x, y):
    """
    Construct and send a formatted string containing the eye coordinate over UART.
    The format here is simple, e.g., 'LE:123.45,67.89\n' for the left eye,
    and 'RE:123.45,67.89\n' for the right eye. Adapt the message format as needed.
    """
    if eye_label == 'left_eye':
        message = f"LE:{x:.2f},{y:.2f}\n"
    elif eye_label == 'right_eye':
        message = f"RE:{x:.2f},{y:.2f}\n"
    else:
        return  # Skip if label is unknown
   
    if ser is not None and ser.is_open:
        try:
            ser.write(message.encode('utf-8'))
            print(f"Sent: {message.strip()}")
        except Exception as e:
            print(f"Error writing to serial port: {e}")
    else:
        print("Serial port not available for sending:", message)

# -----------------------------------------------------------------------------------------------
# User-defined callback function from the pose estimation pipeline
# -----------------------------------------------------------------------------------------------
def app_callback(pad, info, user_data):
    # Get the GstBuffer from the probe info
    buffer = info.get_buffer()
    if buffer is None:
        return Gst.PadProbeReturn.OK

    # Increment the frame count in user_data
    user_data.increment()
    string_to_print = f"Frame count: {user_data.get_count()}\n"

    # Get the format and dimensions from the pad's capabilities
    format, width, height = get_caps_from_pad(pad)

    # Obtain the video frame if needed
    frame = None
    if user_data.use_frame and format is not None and width is not None and height is not None:
        frame = get_numpy_from_buffer(buffer, format, width, height)

    # Get the detections from the buffer
    roi = hailo.get_roi_from_buffer(buffer)
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)
    keypoints = get_keypoints()

    # Process each detection
    for detection in detections:
        label = detection.get_label()
        bbox = detection.get_bbox()
        confidence = detection.get_confidence()
        if label == "person":
            # Get the track ID if available
            track_id = 0
            track = detection.get_objects_typed(hailo.HAILO_UNIQUE_ID)
            if len(track) == 1:
                track_id = track[0].get_id()
            string_to_print += (f"Detection: ID: {track_id} Label: {label} Confidence: {confidence:.2f}\n")

            # If landmarks are available, process eye keypoints
            landmarks = detection.get_objects_typed(hailo.HAILO_LANDMARKS)
            if len(landmarks) != 0:
                points = landmarks[0].get_points()
                for nose in ['nose']:
                    keypoint_index = keypoints[nose]
                    point = points[keypoint_index]
                    # Compute absolute pixel coordinates
                    x = int((point.x() * bbox.width() + bbox.xmin()) * width)
                    y = int((point.y() * bbox.height() + bbox.ymin()) * height)
                    string_to_print += f"{nose}: x: {x:.2f} y: {y:.2f}\n"
                   
                    if user_data.use_frame:
                        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
                   
                    # Send the computed coordinates via UART to the Nucleo board.
                    send_eye_coordinates(nose, x, y)

    if user_data.use_frame:
        # Convert the frame color format from RGB (used in the pipeline) to BGR (used by OpenCV)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        user_data.set_frame(frame)

    print(string_to_print)
    return Gst.PadProbeReturn.OK

# -----------------------------------------------------------------------------------------------
# Utility function to get the COCO keypoints correspondence map
# -----------------------------------------------------------------------------------------------
def get_keypoints():
    """Return the COCO keypoints with their indices."""
    keypoints = {
        'nose': 0,
        'left_eye': 1,
        'right_eye': 2,
        'left_ear': 3,
        'right_ear': 4,
        'left_shoulder': 5,
        'right_shoulder': 6,
        'left_elbow': 7,
        'right_elbow': 8,
        'left_wrist': 9,
        'right_wrist': 10,
        'left_hip': 11,
        'right_hip': 12,
        'left_knee': 13,
        'right_knee': 14,
        'left_ankle': 15,
        'right_ankle': 16,
    }
    return keypoints

# -----------------------------------------------------------------------------------------------
# Main execution: set up and run the pose estimation app
# -----------------------------------------------------------------------------------------------
if __name__ == "__main__":
    user_data = user_app_callback_class()
    app = GStreamerPoseEstimationApp(app_callback, user_data)
    app.run()

    # Close the serial port when the application finishes
    if ser is not None and ser.is_open:
        ser.close()
