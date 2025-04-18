import threading
import time
import cv2
import serial
import numpy as np
from ultralytics import YOLO
from picamera2 import Picamera2

try:
    serial_port = serial.Serial(port='/dev/ttyACM0', baudrate=9600, timeout=1)
    print("Serial port opened successfully.")
except Exception as e:
    print("Failed to open serial port:", e)
    serial_port = None

class PiCameraStream:
    def __init__(self, resolution=(640, 480), target_fps=60):
        self.camera = Picamera2()
        config = self.camera.create_video_configuration(
            main={"format": "XRGB8888", "size": resolution}
        )
        self.camera.configure(config)
        # Request a higher frame rate (ensure the sensor mode supports this)
        self.camera.set_controls({"FrameRate": target_fps})
        self.camera.start()
        self.frame = None
        self.stopped = False
        self.lock = threading.Lock()

    def update(self):
        while not self.stopped:
            frame_bgra = self.camera.capture_array()
            frame = cv2.cvtColor(frame_bgra, cv2.COLOR_BGRA2BGR)
            with self.lock:
                self.frame = frame

    def read(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.stopped = True
        self.camera.stop()
        

def center_bound(xmin, ymin, xmax, ymax, frame_x, frame_y):
    box_x = (xmin + xmax)/2
    box_y = (ymin + ymax)/2
    
    #creating a var for frame allows us to get center and change res
    frame_x_center = frame_x/2
    frame_y_center = frame_y/2
        
    offset_x = frame_x_center - box_x
    offset_y = frame_y_center - box_y
    
    #return and cast as int for pwm values
    return int(offset_x), int(offset_y)

def get_quadrant(x, y, frame_width, frame_height):
    """
    Determine which quadrant a point (x, y) is in:
    1: Top-Left
    2: Top-Right
    3: Bottom-Left
    4: Bottom-Right
    """
    mid_x = frame_width // 2
    mid_y = frame_height // 2
    
    if x < mid_x and y < mid_y:
        return 1  # Top-Left
    elif x >= mid_x and y < mid_y:
        return 2  # Top-Right
    elif x < mid_x and y >= mid_y:
        return 3  # Bottom-Left
    else:
        return 4  # Bottom-Right

def draw_quadrants(frame):
    """Draw quadrant lines on the frame."""
    height, width = frame.shape[:2]
    mid_x = width // 2
    mid_y = height // 2
    
    # Draw vertical line
    cv2.line(frame, (mid_x, 0), (mid_x, height), (255, 255, 255), 1)
    # Draw horizontal line
    cv2.line(frame, (0, mid_y), (width, mid_y), (255, 255, 255), 1)
    
    # Label quadrants
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1
    font_color = (255, 255, 255)
    
    # Quadrant 1: Top-Left
    cv2.putText(frame, "Q1", (mid_x//2 - 10, mid_y//2), font, font_scale, font_color, font_thickness)
    # Quadrant 2: Top-Right
    cv2.putText(frame, "Q2", (mid_x + mid_x//2 - 10, mid_y//2), font, font_scale, font_color, font_thickness)
    # Quadrant 3: Bottom-Left
    cv2.putText(frame, "Q3", (mid_x//2 - 10, mid_y + mid_y//2), font, font_scale, font_color, font_thickness)
    # Quadrant 4: Bottom-Right
    cv2.putText(frame, "Q4", (mid_x + mid_x//2 - 10, mid_y + mid_y//2), font, font_scale, font_color, font_thickness)
    
def send_and_wait_for_echo(offset_x, offset_y, quadrant=None):
    """
    Send offset values to the NucleoF401RE and wait for echo.
    Now includes optional quadrant parameter for future use.
    """
    # 2) Build the ASCII message "(a,b)\r\n"
    expected = f"({offset_x},{offset_y})"
    msg = expected + "\r\n"

    if not (serial_port and serial_port.is_open):
        print("No serial port; would send:", expected)
        return

    # 3) Clear any old input, send and flush
    serial_port.reset_input_buffer()
    serial_port.write(msg.encode('ascii'))
    serial_port.flush()
    print("Sent:", expected)

    # 4) Block here until we get exactly that same line back
    while True:
        line = serial_port.readline()              # waits for '\n'
        if not line:
            # nothing received yet, keep waiting
            continue
        echo = line.decode('ascii', errors='ignore').strip()
        print("Echo:", echo)
        if echo == expected:
            # got the correct echo, break out and allow the next pair
            break
        # otherwise loop again (you could log a warning here)
        
def inference_loop(get_frame_func, model, labels, threshold, bbox_colors):
    while True:
        frame = get_frame_func()
        if frame is None:
            continue
            
        # Draw quadrant lines
        draw_quadrants(frame)
        
        # Run inference
        results = model(frame, verbose=False)
        detections = results[0].boxes
        # Draw detections
        for detection in detections:
            xyxy_tensor = detection.xyxy.cpu()
            xyxy = xyxy_tensor.numpy().squeeze()  # [xmin, ymin, xmax, ymax]
            xmin, ymin, xmax, ymax = xyxy.astype(int)
            #print("offset x:", {my_offset_x}, "\n offset y:", {my_offset_y})
            classidx = int(detection.cls.item())
            classname = labels[classidx]
            conf = detection.conf.item()
            if conf > threshold:
                color = bbox_colors[classidx % len(bbox_colors)]
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
                label = f'{classname}: {int(conf * 100)}%'
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                label_ymin = max(ymin, labelSize[1] + 10)
                cv2.rectangle(frame, (xmin, label_ymin - labelSize[1] - 10),
                              (xmin + labelSize[0], label_ymin + baseLine - 10), color, cv2.FILLED)
                cv2.putText(frame, label, (xmin, label_ymin - 7),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                
                if classname.lower() == 'person':
                    frame_height, frame_width = frame.shape[:2]
                    offset_x, offset_y = center_bound(xmin, ymin, xmax, ymax, frame_width, frame_height)
                    
                    center_x = (xmin + xmax) / 2
                    center_y = (ymin + ymax) / 2
                    
                    # Determine which quadrant the person is in
                    quadrant = get_quadrant(int(center_x), int(center_y), frame_width, frame_height)
                    
                    # Display the center point of the person
                    cv2.circle(frame, (int(center_x), int(center_y)), 5, (0, 255, 0), -1)
                    
                    # Display quadrant info on the frame
                    quadrant_text = f"Q{quadrant}"
                    cv2.putText(frame, quadrant_text, (int(center_x) + 10, int(center_y) + 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    print(f"Person detected. Offset (x, y): ({offset_x}, {offset_y}), Quadrant: {quadrant}")
                    
                    # Pass quadrant info to the send function for future use
                    send_and_wait_for_echo(offset_x, offset_y, quadrant)
                    
        cv2.imshow('YOLO detection results', frame)
        if cv2.waitKey(1) in [ord('q'), ord('Q')]:
            break

def main():
    # Set up the PiCamera stream with high target FPS
    stream = PiCameraStream(resolution=(640, 480), target_fps=60)
    capture_thread = threading.Thread(target=stream.update, daemon=True)
    capture_thread.start()

    # Load YOLO model; adjust the model path as needed
    model_path = "/home/XenaPi/yolo/yolov8n_ncnn_model"
    model = YOLO(model_path, task='detect')
    labels = model.names
    threshold = 0.5
    bbox_colors = [
        (164, 120, 87), (68, 148, 228), (93, 97, 209), (178, 182, 133), (88, 159, 106),
        (96, 202, 231), (159, 124, 168), (169, 162, 241), (98, 118, 150), (172, 176, 184)
    ]

    # Start inference in a dedicated thread so capture and inference run concurrently.
    inference_thread = threading.Thread(
        target=inference_loop,
        args=(stream.read, model, labels, threshold, bbox_colors),
        daemon=True
    )
    inference_thread.start()

    # Keep the main thread alive until user quits
    try:
        while inference_thread.is_alive():
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass

    stream.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
