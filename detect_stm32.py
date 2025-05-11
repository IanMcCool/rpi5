import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import os
import numpy as np
import cv2
import hailo
import serial

from hailo_apps_infra.hailo_rpi_common import (
    get_caps_from_pad,
    get_numpy_from_buffer,
    app_callback_class,
)
from hailo_apps_infra.detection_pipeline import GStreamerDetectionApp
#open serial port to Nucleo
try:
    serial_port = serial.Serial(port='/dev/ttyACM0', baudrate=9600, timeout=1)
    print("Serial port opened successfully.")
except Exception as e:
    print("Failed to open serial port:", e)
    serial_port = None

#call the user_app_callback_class
class user_app_callback_class(app_callback_class):
    def __init__(self):
        super().__init__()
        self.new_variable = 42  # New variable example

    def new_function(self):  # New function example
        return "The meaning of life is: "


#The callback function below runs the inference loop for the object detection.
#This function is invoked from the gstreamer_helper pipeline
# -----------------------------------------------------------------------------------------------
# Callback function invoked by the detection pipeline
# -----------------------------------------------------------------------------------------------
def app_callback(pad, info, user_data):
    # Retrieve GstBuffer
    buffer = info.get_buffer()
    if buffer is None:
        return Gst.PadProbeReturn.OK

    #counts the frames and displays to the terminal
    user_data.increment()
    string_to_print = f"Frame count: {user_data.get_count()}\n"

    #get video format and dimensions
    format, width, height = get_caps_from_pad(pad)

    # Pull frame if needed
    frame = None
    if user_data.use_frame and format and width and height:
        frame = get_numpy_from_buffer(buffer, format, width, height)

    #run the detections
    roi = hailo.get_roi_from_buffer(buffer)
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)

    #This will only detect the first object that the user sets the program to
    found_first = False
    for detection in detections:
        if found_first:
            break

        if detection.get_label() != "person": #class that is used for tracking
            continue

        # extracts the coordinates from the bounding box of detected object
        bbox = detection.get_bbox()
        x1 = int(bbox.xmin() * width)
        y1 = int(bbox.ymin() * height)
        x2 = int(bbox.xmax() * width)
        y2 = int(bbox.ymax() * height)

        #computes the center of the bounding box relative to the frame
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
       

        #sends the coordinates over serial and reads the echo for debugging
        if serial_port:
            msg = f"{cx},{cy}\n"
            serial_port.write(msg.encode('utf-8'))
            serial_port.flush()

            #read every character from echo
            echo = ""
            while True:
                if serial_port.in_waiting > 0:
                    c = serial_port.read(1).decode('utf-8')
                    if c == '\n' or c == '\r':
                        break
                    echo += c
           
            #print the coordinates and echo to the terminal      
            string_to_print += f"First person @ ({cx},{cy}) sent over serial\n"
            if echo:
                string_to_print += f"Echo from Nucleo: {echo}\n"

        found_first = True #set to true after first object found

    # Overlay detection count
    if user_data.use_frame and frame is not None:
        count = 1 if found_first else 0
        cv2.putText(frame, f"Detections: {count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        # Example of using new_variable and new_function
        cv2.putText(frame,
                    f"{user_data.new_function()}{user_data.new_variable}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        # Convert RGB->BGR for display
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        user_data.set_frame(frame_bgr)

    print(string_to_print)
    return Gst.PadProbeReturn.OK

# -----------------------------------------------------------------------------------------------
# Main entrypoint
# -----------------------------------------------------------------------------------------------
if __name__ == "__main__":
    user_data = user_app_callback_class()
    app = GStreamerDetectionApp(app_callback, user_data)
    app.run()
